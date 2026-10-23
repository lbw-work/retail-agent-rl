#!/usr/bin/env bash
# 只建立独立环境，不下载模型、不调用顾客 API、不启动训练。
set -euo pipefail
RETAIL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_ROOT="${1:?用法：bash scripts/setup_verl.sh /absolute/path/to/verl-main}"
VERL_ENV="$RETAIL_ROOT/.venv-verl"
if [[ ! -f "$VERL_ROOT/uv.lock" ]]; then
  echo "缺少已冻结的 verl 源码和 uv.lock" >&2
  exit 1
fi
if [[ "$(uname -s)" != Linux ]]; then
  echo "完整 FSDP/vLLM 环境仅在 Linux CUDA 服务器建立；Mac 运行本地专项测试。" >&2
  exit 1
fi
UV_PROJECT_ENVIRONMENT="$VERL_ENV" uv sync --project "$VERL_ROOT" --frozen --extra fsdp --extra vllm
uv pip install --python "$VERL_ENV/bin/python" -r "$RETAIL_ROOT/requirements/verl-retail.txt"
uv pip install --python "$VERL_ENV/bin/python" --no-deps --editable "$RETAIL_ROOT"
uv pip check --python "$VERL_ENV/bin/python"
"$VERL_ENV/bin/python" -m retail_agent.train_verl --preflight --verl-root "$VERL_ROOT"
