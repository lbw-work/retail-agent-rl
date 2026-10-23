"""固定MMOPD完整SFT快照、零增量LoRA和原生思考适配；不改旧实验协议。"""

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

from retail_agent.environment import ROOT
from retail_agent.model import ContextLimit

REPO = 'MMOPD/Qwen3-4B-OT3-tau2-2ep'
REVISION = 'd2712252ebb411b9a68a92474c1d868469357789'
PROFILE = ROOT / 'configs/verl/mmopd.toml'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def profile():
    config = tomllib.loads(PROFILE.read_text())
    if (config['model_id'], config['model_revision'], config['enable_thinking']) != (REPO, REVISION, True):
        raise ValueError('MMOPD仓库、revision或原生思考模式改变，须另立实验')
    for key in ['model_path', 'sft_adapter_dir']:
        path = (ROOT / config[key]).resolve()
        if not path.is_relative_to((ROOT / 'models').resolve()):
            raise ValueError('新模型和初始化adapter必须在本项目models中')
    if config['model_path'] == config['sft_adapter_dir']:
        raise ValueError('完整SFT模型与LoRA初始化目录必须分开')
    return config


def verify_snapshot(require_adapter=False):
    config = profile()
    model = ROOT / config['model_path']
    manifest = json.loads((model / 'SOURCE_MANIFEST.json').read_text())
    if (manifest['repo_id'], manifest['revision']) != (REPO, REVISION):
        raise ValueError('模型快照身份不符')
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json',
                'chat_template.jinja', 'model.safetensors.index.json'}
    if not required <= manifest['files'].keys():
        raise ValueError('模型清单不完整')
    weights = set(json.loads((model / 'model.safetensors.index.json').read_text())['weight_map'].values())
    if not weights or not weights <= manifest['files'].keys():
        raise ValueError('模型分片缺失')
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('模型清单含非法路径')
        path = model / name
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('新模型文件校验失败：' + name)
        if name in weights and expected.get('upstream_lfs_sha256') != expected['sha256']:
            raise ValueError('权重未与上游LFS SHA256核对：' + name)
    report = {'model_id': REPO, 'model_revision': REVISION,
              'model_manifest_sha256': digest(model / 'SOURCE_MANIFEST.json'),
              'experiment_profile': config['experiment_profile'], 'enable_thinking': True,
              'model_path': config['model_path'], 'data_exposure': config['data_exposure']}
    if require_adapter:
        adapter = ROOT / config['sft_adapter_dir']
        init = json.loads((adapter / 'initialization.json').read_text())
        if init['model_manifest_sha256'] != report['model_manifest_sha256']:
            raise ValueError('LoRA初始化不属于当前完整SFT快照')
        if init['profile_sha256'] != digest(PROFILE) or not init['identity_adapter']:
            raise ValueError('LoRA初始化协议不符')
        for name, expected in init['files'].items():
            if Path(name).name != name or digest(adapter / name) != expected:
                raise ValueError('LoRA初始化文件不一致：' + name)
        if set(init['files']) != {'adapter_config.json', 'adapter_model.safetensors'}:
            raise ValueError('LoRA初始化清单不完整')
        report['initial_adapter_sha256'] = init['files']
    return report


def download():
    from huggingface_hub import HfApi, snapshot_download

    config = profile()
    model = ROOT / config['model_path']
    if (model / 'SOURCE_MANIFEST.json').exists():
        return verify_snapshot()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('上游revision不一致')
    names = [f.rfilename for f in info.siblings if f.rfilename != '.gitattributes'
             and Path(f.rfilename).name == f.rfilename
             and (f.rfilename.endswith(('.json', '.jinja', '.safetensors', '.txt'))
                  or f.rfilename in {'README.md', 'LICENSE'})]
    snapshot_download(REPO, revision=REVISION, local_dir=model, allow_patterns=names)
    files = {}
    for remote in info.siblings:
        name = remote.rfilename
        if name not in names:
            continue
        path = model / name
        sha = digest(path)
        record = {'bytes': path.stat().st_size, 'sha256': sha}
        if remote.lfs is not None:
            if sha != remote.lfs.sha256:
                raise ValueError('下载的LFS内容不完整：' + name)
            record['upstream_lfs_sha256'] = remote.lfs.sha256
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != remote.blob_id:
                raise ValueError('下载的Git对象不完整：' + name)
        files[name] = record
    (model / 'SOURCE_MANIFEST.json').write_text(json.dumps(
        {'repo_id': REPO, 'revision': REVISION, 'files': files}, indent=2) + '\n')
    return verify_snapshot()


def init_adapter():
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    report = verify_snapshot()
    config = profile()
    target = ROOT / config['sft_adapter_dir']
    if target.exists():
        return verify_snapshot(require_adapter=True)
    torch.manual_seed(config['seed'])
    base = AutoModelForCausalLM.from_pretrained(ROOT / config['model_path'],
        torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa')
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32,
        lora_dropout=0.0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))
    target.mkdir(parents=True)
    model.save_pretrained(target, safe_serialization=True, save_embedding_layers=False)
    weights = load_file(str(target / 'adapter_model.safetensors'))
    if not weights or not any('lora_B' in k for k in weights):
        raise ValueError('初始化未产生LoRA参数')
    if any(not torch.isfinite(t).all() or ('lora_B' in k and torch.count_nonzero(t))
           for k, t in weights.items()):
        raise ValueError('LoRA必须有限且B为零，才能作为完整SFT的恒等增量起点')
    (target / 'initialization.json').write_text(json.dumps({
        'model_manifest_sha256': report['model_manifest_sha256'], 'profile_sha256': digest(PROFILE),
        'identity_adapter': True, 'training_started': False, 'seed': config['seed'],
        'files': {n: digest(target / n) for n in ['adapter_config.json', 'adapter_model.safetensors']},
    }, indent=2) + '\n')
    return verify_snapshot(require_adapter=True)


def visible_action(text):
    stripped = text.strip()
    if not stripped.startswith('<think>'):
        return text
    end = stripped.find('</think>')
    if end < 0 or '<think>' in stripped[len('<think>'):end]:
        return text
    # 不闭合/重复标签仍交给原解析器拒绝，不从非法输出中捞取工具调用。
    return stripped[end + len('</think>'):].strip()


class ThinkingTokenizer:
    """只改变模板模式和动作视图，原始采样token/probability完全不动。"""

    def __init__(self, tokenizer):
        self.original = tokenizer

    def __getattr__(self, name):
        return getattr(self.original, name)

    def apply_chat_template(self, *args, **kwargs):
        return self.original.apply_chat_template(*args, **dict(kwargs, enable_thinking=True))

    def decode(self, *args, **kwargs):
        return visible_action(self.original.decode(*args, **kwargs))


class NativeAgent:
    def __init__(self, model, tokenizer, config, journal=None):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.tool_protocol = config.get('tool_protocol')
        self.journal = journal
        self.reset()

    def reset(self):
        self.ids = None
        self.message_count = 0

    def generate(self, messages, tools):
        import torch
        from transformers import GenerationConfig

        if self.ids is None:
            self.ids = list(self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                add_generation_prompt=True, enable_thinking=True))
        else:
            feedback_messages = [messages[-1]]
            if self.tool_protocol in {'mmopd_native_multi_tool_v1', 'mmopd_native_multi_tool_v2'}:
                feedback_messages = [m for m in messages[self.message_count:] if m['role'] in {'user', 'tool'}]
                if not feedback_messages:
                    raise ValueError('多工具模式缺少新增顾客或工具反馈')
            feedback = '\n' + self.tokenizer.apply_chat_template(feedback_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=True)
            self.ids.extend(self.tokenizer.encode(feedback, add_special_tokens=False))
        self.message_count = len(messages)
        ids = list(self.ids)
        remaining = self.config['max_trajectory_tokens'] - len(ids)
        if remaining < 1:
            raise ContextLimit('思考模式上下文已满，不裁剪历史')
        inputs = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        eos = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        generation = GenerationConfig(do_sample=True, temperature=self.config['temperature'],
            top_p=1.0, top_k=0, repetition_penalty=1.0,
            max_new_tokens=min(self.config['max_new_tokens_per_action'], remaining),
            eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=generation, use_model_defaults=False, disable_compile=True,
                logits_to_keep=1)
        tokens = output[0, len(ids):].tolist()
        self.ids.extend(tokens)
        stopped = bool(tokens) and tokens[-1] == eos
        raw = self.tokenizer.decode(tokens[:-1] if stopped else tokens, skip_special_tokens=False)
        result = {'raw_text': visible_action(raw), 'raw_model_text': raw,
                'prompt_token_ids': ids, 'generated_token_ids': tokens,
                'finish_reason': 'stop' if stopped else 'length',
                'backend': 'mmopd_cuda_bf16_native_think', 'enable_thinking': True}
        if self.journal is not None:
            with Path(self.journal).open('a') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--download', action='store_true')
    mode.add_argument('--init-adapter', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = download() if args.download else init_adapter() if args.init_adapter else verify_snapshot(True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
