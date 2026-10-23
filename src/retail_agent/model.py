"""固定 Qwen3-4B 的本机量化推理，可加载已训练的 PEFT LoRA。"""

import json
import re
import time


class ContextLimit(RuntimeError):
    pass


def parse_action(text: str) -> dict:
    text = text.strip()
    if not text or "<think>" in text or "</think>" in text:
        raise ValueError("客服输出为空或包含思考标签")
    if "<tool_call>" in text or "</tool_call>" in text:
        match = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
        if not match:
            raise ValueError("客服回合必须只有一个工具调用，不能混入文字")
        call = json.loads(match[1])
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("工具调用必须包含 name 和 arguments")
        if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise ValueError("工具调用名称或参数类型无效")
        return {"kind": "tool", **call}
    return {"kind": "message", "text": text}


class QwenAgent:
    def __init__(self, path, max_context_tokens=16384, max_new_tokens=512, seed=42,
                 adapter_path=None, use_prompt_cache=True):
        import mlx.core as mx
        from mlx_lm import load
        mx.random.seed(seed)
        self.model, self.tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False})
        self.adapter_info = None
        if adapter_path is not None:
            from retail_agent.adapter import load_peft_adapter
            self.adapter_info = load_peft_adapter(self.model, adapter_path)
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        # 复用 MLX 官方服务端的精确 token 前缀缓存，每个模型只保留一份。
        from mlx_lm.server import LRUPromptCache
        self.prompt_cache = LRUPromptCache(max_size=1) if use_prompt_cache else None

    def generate(self, messages, tools):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        if len(prompt) + self.max_new_tokens > self.max_context_tokens:
            raise ContextLimit("上下文超过本次推理上限；没有静默截断")
        from mlx_lm.models.cache import make_prompt_cache
        cache, rest = (self.prompt_cache.fetch_nearest_cache("agent", prompt)
                       if self.prompt_cache is not None else (None, prompt))
        if cache is None:
            cache = make_prompt_cache(self.model)
        tokens, logprobs, pieces = [], [], []
        started = time.monotonic()
        for response in stream_generate(self.model, self.tokenizer, rest, max_tokens=self.max_new_tokens,
                                        sampler=make_sampler(temp=0.0), prefill_step_size=256,
                                        prompt_cache=cache):
            tokens.append(response.token)
            logprobs.append(float(response.logprobs[response.token].item()))
            pieces.append(response.text)
        if self.prompt_cache is not None:
            self.prompt_cache.insert_cache("agent", prompt + tokens, cache)
        result = {"raw_text": "".join(pieces), "prompt_token_ids": prompt,
                  "generated_token_ids": tokens, "generated_logprobs": logprobs,
                  "finish_reason": response.finish_reason, "prompt_tokens": len(prompt),
                  "processed_prompt_tokens": len(rest), "cached_prompt_tokens": len(prompt) - len(rest),
                  "generated_tokens": len(tokens), "peak_memory_gb": response.peak_memory,
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "backend": "mlx_local_4bit", "enable_thinking": False,
                  "sampling_temperature": 0.0, "logprob_kind": "model_log_softmax_not_grpo_policy"}
        mx.clear_cache()
        return result
