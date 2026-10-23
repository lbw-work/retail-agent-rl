"""单进程串行基线的阶段计时；嵌套耗时分别报告 inclusive/self，不能重复相加。"""

import json
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


class StageTimer:
    def __init__(self, output_dir=None, synchronize=None):
        self.run_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.synchronize = synchronize
        self.stack, self.stages, self.groups, self.api_records = [], {}, [], []
        self.directory = Path(output_dir) / "observations" if output_dir is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, event, **fields):
        if event == "customer_api":
            self.api_records.append(fields.get("api") or {})
        if self.directory is not None:
            with (self.directory / f"{self.run_id}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"run_id": self.run_id, "event": event,
                    "offset_seconds": time.perf_counter() - self.started, **fields}, ensure_ascii=False) + "\n")

    @contextmanager
    def measure(self, stage, *, gpu=False, **metadata):
        if gpu and self.synchronize is not None:
            self.synchronize()
        frame = {"stage": stage, "started": time.perf_counter(), "children": 0.0}
        self.stack.append(frame)
        status, error = "completed", None
        try:
            yield
            if gpu and self.synchronize is not None:
                self.synchronize()
        except BaseException as exception:
            status, error = "failed", type(exception).__name__
            raise
        finally:
            elapsed = time.perf_counter() - frame["started"]
            assert self.stack.pop() is frame, "计时范围必须按嵌套顺序结束"
            if self.stack:
                self.stack[-1]["children"] += elapsed
            own = max(0.0, elapsed - frame["children"])
            stats = self.stages.setdefault(stage, {"calls": 0, "failed_calls": 0,
                "inclusive_seconds": 0.0, "self_seconds": 0.0, "max_seconds": 0.0})
            stats["calls"] += 1
            stats["failed_calls"] += status == "failed"
            stats["inclusive_seconds"] += elapsed
            stats["self_seconds"] += own
            stats["max_seconds"] = max(stats["max_seconds"], elapsed)
            self.record("stage", stage=stage, status=status, error_type=error,
                inclusive_seconds=elapsed, self_seconds=own,
                cuda_boundary_synchronized=gpu and self.synchronize is not None and status == "completed", **metadata)
            if not self.stack:
                self.save()

    def observe_group(self, group, reward_mode, *, task_businesses):
        rewards = [r["scores"][reward_mode] for r in group]
        item = {"task_id": group[0]["task_id"], "businesses": task_businesses,
            "rewards": rewards, "effective": len(set(rewards)) > 1,
            "terminations": dict(Counter(r["termination"] for r in group)),
            "assistant_tokens": [sum(r["assistant_mask"]) for r in group],
            "feedback_tokens": [len(r["completion_ids"]) - sum(r["assistant_mask"]) for r in group],
            "trajectory_tokens": [len(r["prompt_ids"]) + len(r["completion_ids"]) for r in group],
            "assistant_actions": [r["assistant_actions"] for r in group],
            "tool_calls": [sum(m.get("role") == "tool" for m in r["trajectory"]) for r in group],
            "tool_errors": [sum(m.get("role") == "tool" and bool(m.get("error"))
                for m in r["trajectory"]) for r in group]}
        self.groups.append(item)
        self.record("rollout_group", **item)
        self.save()

    def summary(self):
        wall = time.perf_counter() - self.started
        effective = sum(g["effective"] for g in self.groups)
        sampled = sum(len(g["rewards"]) for g in self.groups)
        tokens = sum(sum(g["assistant_tokens"]) for g in self.groups)
        generation = self.stages.get("assistant_generation", {}).get("inclusive_seconds", 0)
        businesses = sorted({b for g in self.groups for b in g["businesses"]})
        known_api = [r for r in self.api_records if "peak_cost_upper_cny" in r]
        return {"run_id": self.run_id, "wall_seconds": wall,
            "timing_method": "perf_counter_wall_with_optional_cuda_boundary_sync",
            "unaccounted_or_active_seconds": max(0.0, wall - sum(s["self_seconds"] for s in self.stages.values())),
            "active_stages": [s["stage"] for s in self.stack], "stages": self.stages,
            "customer_api": {"returned_calls": len(self.api_records), "usage_known_calls": len(known_api),
                "completed_usage_cost_upper_cny": sum(r["peak_cost_upper_cny"] for r in known_api),
                "billing_note": "completed_returns_only; failures/retries/reserves remain in API ledger"},
            "groups": {"total": len(self.groups), "effective": effective,
                "effective_rate": effective / len(self.groups) if self.groups else None,
                "all_zero": sum(all(r == 0 for r in g["rewards"]) for g in self.groups),
                "all_one": sum(all(r == 1 for r in g["rewards"]) for g in self.groups),
                "sampled_episodes": sampled, "assistant_tokens": tokens,
                "assistant_tokens_per_generation_second": tokens / generation if generation and tokens else None,
                "wall_seconds_per_effective_group": wall / effective if effective else None,
                "by_business": {b: {"groups": sum(b in g["businesses"] for g in self.groups),
                    "effective_groups": sum(g["effective"] for g in self.groups if b in g["businesses"]),
                    "sampled_episodes": sum(len(g["rewards"]) for g in self.groups if b in g["businesses"]),
                    "whole_task_successes": sum(sum(r == 1 for r in g["rewards"])
                        for g in self.groups if b in g["businesses"])}
                    for b in businesses}}}

    def save(self):
        if self.directory is not None:
            target = self.directory / f"{self.run_id}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(target)


def timed(stage, *, gpu=False):
    """给已有 Trainer 方法加观测，不改变参数、返回值或异常。"""
    def decorate(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            with self.timing.measure(stage, gpu=gpu,
                    step=getattr(getattr(self, "state", None), "global_step", None)):
                return function(self, *args, **kwargs)
        return wrapper
    return decorate
