#!/usr/bin/env bash
# ultralytics_run.sh — Ultralytics YOLO 训练启动脚本
# 统一接口: bash ultralytics_run.sh <run_dir> <config_path> <log_dir> <logger_type>
#   run_dir      : 训练输出目录（weights 等）
#   config_path  : YAML 配置文件（含 task, model, data, epochs, imgsz 等）
#   log_dir      : 日志目录
#   logger_type  : local | wandb | swanlab
set -euo pipefail

# ── 参数校验 ──────────────────────────────────────────────
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <run_dir> <config_path> <log_dir> <logger_type>" >&2
  exit 1
fi

RUN_DIR="$1"
CONFIG_PATH="$2"
LOG_DIR="$3"
LOGGER_TYPE="$4"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ultralytics] config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

LOG_FILE="${LOG_DIR}/train.log"

# ── logger 透传 ───────────────────────────────────────────
case "$LOGGER_TYPE" in
  wandb)
    echo "[ultralytics] wandb logger requested — set WANDB_API_KEY before running"
    ;;
  swanlab)
    echo "[ultralytics] swanlab logger requested — set SWANLAB_API_KEY before running"
    ;;
  local|*)
    : # ultralytics 默认 local 日志
    ;;
esac

# 从 YAML 配置中提取参数并构造 yolo CLI 参数
YOLO_ARGS=$(python -c "
import yaml
with open('$CONFIG_PATH') as f:
    cfg = yaml.safe_load(f) or {}
parts = []
for k, v in cfg.items():
    if v is None:
        continue
    if isinstance(v, bool):
        parts.append(f'{k}={str(v).lower()}')
    elif isinstance(v, (int, float)):
        parts.append(f'{k}={v}')
    else:
        parts.append(f'{k}={v}')
print(' '.join(parts))
" 2>/dev/null)

if [[ -z "$YOLO_ARGS" ]]; then
  echo "[ultralytics] failed to parse config or config empty" >&2
  exit 1
fi

echo "[ultralytics] run_dir=$RUN_DIR config=$CONFIG_PATH log=$LOG_FILE logger=$LOGGER_TYPE"
echo "[ultralytics] yolo args: $YOLO_ARGS"

# ── 启动训练 ──────────────────────────────────────────────
# ultralytics 会自动在 runs/ 下创建输出，通过 project 参数指定输出目录
yolo train project="$RUN_DIR" $YOLO_ARGS 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[ultralytics] training finished, exit_code=$EXIT_CODE"
exit "$EXIT_CODE"
