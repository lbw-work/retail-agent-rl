#!/usr/bin/env bash
# 各阶段显式执行；不自动跨过失败，也不修改旧模型或实验。
set -euo pipefail
RETAIL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$(dirname "$RETAIL_ROOT")/verl-main}"
PY="$RETAIL_ROOT/.venv-verl/bin/python"
TAG="${TAG:-mmopd_20261020_01}"
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'TAG只能包含字母、数字、下划线、横线'; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/root/retail_verl_cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$RETAIL_ROOT"
mkdir -p outputs/logs
mode="${1:-help}"
case_dir="$RETAIL_ROOT/outputs/mmopd_case/${TAG}_174"
cal_dir="$RETAIL_ROOT/outputs/verl/${TAG}_calibration"
smoke_dir="$RETAIL_ROOT/outputs/verl/${TAG}_smoke"
formal_dir="$RETAIL_ROOT/outputs/verl/${TAG}_full94"
selection_dir="$RETAIL_ROOT/outputs/verl/${TAG}_selection"

run_logged() {
  "$@" 2>&1 | tee "outputs/logs/${TAG}_${mode}_$(date +%Y%m%d_%H%M%S).log"
}

case "$mode" in
  prepare)
    "$PY" -m retail_agent.mmopd --download
    "$PY" -m retail_agent.mmopd --init-adapter
    "$PY" -m retail_agent.train_verl --profile mmopd --preflight --verl-root "$VERL_ROOT"
    "$PY" -m retail_agent.train_verl --profile mmopd --dry-run --max-steps 94 \
      --verl-root "$VERL_ROOT" --output-dir "$formal_dir" > "outputs/logs/${TAG}_full94_config.yaml"
    ;;
  case)
    run_logged "$PY" -m retail_agent.run_mmopd_case --run --task-id retail_174 --output-dir "$case_dir"
    ;;
  calibrate)
    run_logged "$PY" -m retail_agent.train_verl --profile mmopd --calibrate --task-id retail_105 \
      --max-steps 1 --verl-root "$VERL_ROOT" --output-dir "$cal_dir"
    ;;
  smoke)
    run_logged "$PY" -m retail_agent.train_verl --profile mmopd --run \
      --task-id retail_105 --task-id retail_112 --max-steps 2 \
      --verl-root "$VERL_ROOT" --output-dir "$smoke_dir"
    "$PY" -m retail_agent.audit_verl --profile mmopd \
      --checkpoint "$smoke_dir/checkpoints/global_step_2/actor" \
      --output-dir "$smoke_dir/reloaded_adapter" --reload-cuda
    ;;
  formal)
    "$PY" - "$smoke_dir" "$cal_dir" <<'PY'
import json, sys
from pathlib import Path
from retail_agent.mmopd import verify_snapshot
smoke, calibration = map(Path, sys.argv[1:])
run = json.loads((smoke / 'run.json').read_text())
audit = json.loads((smoke / 'reloaded_adapter/audit.json').read_text())
cal = json.loads((calibration / 'run.json').read_text())
assets = verify_snapshot(require_adapter=True)
if (run['status'] != 'training_returned_pending_reload_acceptance' or run['max_steps'] != 2
    or run.get('experiment_profile') != assets['experiment_profile']
    or cal['status'] != 'calibration_recorded_no_optimizer_update'
    or audit['status'] != 'cuda_reload_verified'
    or audit.get('initial_adapter_sha256') != assets['initial_adapter_sha256']
    or run.get('model_manifest_sha256') != assets['model_manifest_sha256']):
    raise SystemExit('尚未通过本模型校准、两步smoke及独立CUDA重载，停止正式训练')
print('验收通过：正式训练从同一完整SFT和零增量LoRA重新开始，不接续smoke权重。')
PY
    run_logged "$PY" -m retail_agent.train_verl --profile mmopd --run --max-steps 94 \
      --verl-root "$VERL_ROOT" --output-dir "$formal_dir"
    ;;
  select-preflight)
    "$PY" -m retail_agent.eval_verl --preflight --run-dir "$formal_dir" \
      --output-dir "$selection_dir" --expected-steps 94 --api-limit-cny 0
    ;;
  select|select-resume)
    extra=()
    [[ "$mode" == select-resume ]] && extra=(--resume)
    run_logged "$PY" -m retail_agent.eval_verl --run-dir "$formal_dir" \
      --output-dir "$selection_dir" --expected-steps 94 --api-limit-cny 0 "${extra[@]}"
    ;;
  status)
    "$PY" - "$case_dir" "$cal_dir" "$smoke_dir" "$formal_dir" "$selection_dir" <<'PY'
import json, sys
from pathlib import Path
for name in sys.argv[1:]:
    folder = Path(name)
    print('\n' + folder.name)
    for relative in ['run.json', 'summary.json', 'reloaded_adapter/audit.json', 'best.json']:
        path = folder / relative
        if path.exists():
            obj = json.loads(path.read_text())
            fields = ['status', 'task_id', 'scores', 'termination', 'max_steps', 'gpu',
                      'wall_seconds', 'api_calls', 'api_cost_upper_cny', 'step', 'success_rate',
                      'candidate_count', 'checkpoint_selection_scope', 'best_adapter']
            print(relative, json.dumps({k: obj[k] for k in fields if k in obj}, ensure_ascii=False))
    checkpoints = list((folder / 'checkpoints').glob('global_step_*'))
    if checkpoints:
        print('latest_saved_step', max(int(p.name.rsplit('_', 1)[1]) for p in checkpoints))
PY
    ;;
  *)
    echo '用法：bash scripts/mmopd_server.sh prepare|case|calibrate|smoke|formal|select-preflight|select|select-resume|status'
    [[ "$mode" == help ]] || exit 1
    ;;
esac
