#!/usr/bin/env bash
# trl_run.sh — HuggingFace TRL 训练启动脚本
# 统一接口: bash trl_run.sh <run_dir> <config_path> <log_dir> <logger_type>
#   run_dir      : 训练输出目录（checkpoints 等）
#   config_path  : TRL YAML 训练配置文件（含 sft/dpo/grpo 字段）
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
  echo "[trl] config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

LOG_FILE="${LOG_DIR}/train.log"

# ── logger 透传 ───────────────────────────────────────────
case "$LOGGER_TYPE" in
  wandb)
    export TRL_REPORT_TO="wandb"
    ;;
  swanlab)
    export TRL_REPORT_TO="swanlab"
    ;;
  local|*)
    export TRL_REPORT_TO="none"
    ;;
esac

# 从配置中推断子命令（sft/dpo/grpo），默认 sft
SUBCMD=$(python -c "
import yaml, sys
with open('$CONFIG_PATH') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('trl_subcommand', 'sft'))
" 2>/dev/null || echo "sft")

echo "[trl] subcmd=$SUBCMD run_dir=$RUN_DIR config=$CONFIG_PATH log=$LOG_FILE logger=$LOGGER_TYPE"

# ── 启动训练 ──────────────────────────────────────────────
trl "$SUBCMD" \
  --config "$CONFIG_PATH" \
  --output_dir "$RUN_DIR" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[trl] training finished, exit_code=$EXIT_CODE"
exit "$EXIT_CODE"
