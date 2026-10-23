"""桌面 verl V1 的 Retail 扩展；GPU 依赖只在新入口中导入。"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import NonTensorData, NonTensorStack
from torch.utils.data import Dataset

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker, DictConfigWrap,
)
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.logging_utils import configure_verl_logging
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from retail_agent.customer import DeepSeekCustomer, GOAL_POLICY
from retail_agent.environment import RetailSession
from retail_agent.grpo_rollout import RolloutError
from retail_agent.observability import StageTimer
from retail_agent.training_reward import ready_training_rows
from retail_agent.verl_agent_loop import (
    complete_group, probability_diagnostics, run_episode, selected_training_rows, shared_opening,
)


class RetailDataset(Dataset):
    """只将公开提示和任务 ID 交给调度器；目标与金标准不进入模型 prompt。"""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        self.rows = selected_training_rows(data_files, max_samples)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"raw_prompt": RetailSession(row).agent_messages(), "index": index,
                "extra_info": {"task_id": row["id"]}, "data_source": "retail",
                "reward_model": {"style": "rule", "ground_truth": ""}}


class RetailAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        result = await run_episode(kwargs["row"], self.server_manager, self.tokenizer,
                                 kwargs["customer_factory"], kwargs["opening"], kwargs["retail_config"],
                                 sampling_params, kwargs["timing"], kwargs["request_id"])
        return AgentLoopOutput(
            prompt_ids=result["prompt_ids"], response_ids=result["completion_ids"],
            response_mask=result["assistant_mask"], response_logprobs=result["logprobs"],
            reward_score=float(result["scores"][kwargs["retail_config"]["reward_mode"]]),
            num_turns=result["assistant_actions"], metrics=AgentLoopMetrics(),
            extra_fields={"retail_trace": result})


class RetailGroupWorker(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.retail = OmegaConf.to_container(self.config.retail, resolve=True)
        self.rows = {r["id"]: r for r in ready_training_rows(
            Path(self.retail["train_tasks"]), self.retail["review_manifest"])}
        self.timing = StageTimer(self.retail["output_dir"])
        self.agent = RetailAgentLoop(
            trainer_config=DictConfigWrap(config=self.config), server_manager=self.llm_client,
            tokenizer=self.tokenizer, processor=None, dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(config=self.config.data), hf_model_type=self.hf_model_type)

    def customer_factory(self, scenario):
        return DeepSeekCustomer(scenario, {**self.retail["customer"], "goal_policy": GOAL_POLICY},
                                limit_cny=None, ledger_path=Path(self.retail["output_dir"]) / "customer_costs.jsonl")

    async def generate_sequences(self, batch):
        # 同步阻塞到整组发布或抛错；不让默认 ReplayBuffer 消费残缺失败组。
        for i in range(len(batch)):
            prompt = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    prompt[key] = value[i].item() if value[i].ndim == 0 else value[i]
                elif isinstance(value, NonTensorStack):
                    prompt[key] = value[i].data
                elif isinstance(value, NonTensorData):
                    prompt[key] = value.data
                else:
                    raise TypeError(f"未支持的 TransferQueue 字段：{key}: {type(value)}")
            if prompt.get("validate", False):
                raise ValueError("首版只验收 TRAIN 闭环，dev/test 评估不自动启动")
            await self._run_group(prompt)

    async def _run_group(self, prompt):
        uid, step = str(prompt["uid"]), int(prompt["global_steps"])
        row = self.rows[prompt["extra_info"]["task_id"]]
        await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "running"})
        try:
            opening = await shared_opening(row, self.customer_factory, self.timing)
            initial = RetailSession(row)
            initial.add_user_message(opening)
            prompt = {**prompt, "raw_prompt": initial.agent_messages()}
            rollout = self.config.actor_rollout_ref.rollout
            params = dict(temperature=rollout.temperature, top_p=rollout.top_p,
                          top_k=rollout.top_k, repetition_penalty=1.0, logprobs=True)

            async def run_one(session_id, text):
                output = await self.agent.run(params, row=row, customer_factory=self.customer_factory,
                    opening=text, retail_config=self.retail, timing=self.timing,
                    request_id=f"{uid}-{session_id}")
                result = output.extra_fields["retail_trace"]
                self.timing.record("rollout_trace", uid=uid, session_id=session_id,
                                   sampled_policy_step=result["policy_version"], trace=result)
                return result

            # fit 从训练步 1 开始；初始化同步版本 0，每次 on_step_end 再同步一次。
            results = await complete_group(row, opening, run_one, int(rollout.n), step - 1)
            self.timing.observe_group(results, self.retail["reward_mode"], task_businesses=row["businesses"])
            if step == 1 and self.retail["require_nonzero_gradient"] and len({
                    r["scores"][self.retail["reward_mode"]] for r in results}) == 1:
                raise RolloutError("首组奖励相同，没有有效学习信号；保留轨迹，未更新参数或自动补采")
            await self._publish(uid, step, prompt, results)
            await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "finished"})
        except Exception as error:
            self.timing.record("group_failed", uid=uid, step=step, error_type=type(error).__name__)
            await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "failure"})
            raise

    async def _publish(self, uid, step, prompt, results):
        fields, tags = [], []
        for result in results:
            score = float(result["scores"][self.retail["reward_mode"]])
            if not math.isfinite(score):
                raise RolloutError("奖励非有限")
            output = AgentLoopOutput(
                prompt_ids=result["prompt_ids"], response_ids=result["completion_ids"],
                response_mask=result["assistant_mask"], response_logprobs=result["logprobs"],
                reward_score=score, num_turns=result["assistant_actions"], metrics=AgentLoopMetrics(),
                extra_fields={"min_global_steps": result["policy_version"],
                              "max_global_steps": result["policy_version"],
                              "reward_extra_info": {"reward_a": score}})
            field = output.as_dict()
            field.update(prompt)
            field["input_ids"] = torch.cat([field["prompts"], field["responses"]])
            field["attention_mask"] = torch.ones_like(field["input_ids"])
            field["position_ids"] = torch.arange(len(field["input_ids"]), dtype=torch.int64)
            field["loss_mask"] = field["response_mask"]
            field["multi_modal_inputs"] = {}
            fields.append(field)
            tags.append({"status": "success", "global_steps": step,
                         "min_global_steps": result["policy_version"],
                         "max_global_steps": result["policy_version"], "prompt_len": len(result["prompt_ids"]),
                         "response_len": len(result["completion_ids"]), "seq_len": len(field["input_ids"])})
        await tq.async_kv_batch_put(keys=[f"{uid}_{i}_0" for i in range(len(results))],
            fields=list_of_dict_to_tensordict(fields), tags=tags, partition_id="train")


class RetailAgentLoopManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(RetailGroupWorker)


class RetailPPOTrainer(PPOTrainerSync):
    @contextmanager
    def phase(self, name):
        path = Path(self.config.retail.output_dir) / "phases.jsonl"
        def record(event):
            with path.open("a") as stream:
                stream.write(json.dumps({"phase": name, "event": event, "step": self.global_steps,
                                         "unix_time": time.time()}) + "\n")
        record("begin")
        try:
            yield
        finally:
            record("end")

    def _compute_old_log_prob(self, batch, metrics):
        with self.phase("old_logprob"):
            batch = super()._compute_old_log_prob(batch, metrics)
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                select_fields=["old_log_probs", "rollout_log_probs", "response_mask"]).to_padded_tensor()
            mask = data["response_mask"].bool()
            metrics.update(probability_diagnostics(data["rollout_log_probs"][mask].tolist(),
                                                   data["old_log_probs"][mask].tolist()))
            return batch

    def _update_actor(self, batch, metrics):
        if self.config.retail.calibration_only:
            metrics["retail/calibration_only"] = 1
            return batch
        with self.phase("actor_update"):
            result = super()._update_actor(batch, metrics)
            grad_norm = float(metrics["actor/grad_norm"])
            if not math.isfinite(grad_norm):
                raise RolloutError("梯度范数非有限；本次更新不能验收")
            if self.config.retail.require_nonzero_gradient and self.global_steps == 1 and grad_norm == 0:
                raise RolloutError("本组没有非零梯度，不能作为有效更新验收；未修改奖励或自动补采")
            return result

    def _compute_reward_colocate(self, batch, metrics=None):
        # 奖励已在真实 Retail 环境判分，不能再用默认文本 reward 覆盖。
        scores = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                select_fields=["rm_scores"])
        if len(scores) != len(batch):
            raise RolloutError("采样阶段缺少环境奖励")
        return batch

    def prepare_step(self):
        with self.phase("rollout"):
            return super().prepare_step()

    def on_step_end(self):
        if self.config.retail.calibration_only:
            return
        with self.phase("weight_sync"):
            super().on_step_end()

    def on_sample_end(self):
        with self.phase("rollout_sleep"):
            super().on_sample_end()


@ray.remote
class RetailTaskRunner:
    def run(self, config):
        configure_verl_logging()
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = RetailPPOTrainer(config=config)
            trainer.init()
            manager = RetailAgentLoopManager.create(config=config, llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(), reward_loop_worker_handles=trainer.get_reward_handles())
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()
