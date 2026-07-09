#!/usr/bin/env bash
# llamafactory_run.sh — LLaMA-Factory 训练启动脚本
# 统一接口: bash llamafactory_run.sh <run_dir> <config_path> <log_dir> <logger_type>
#   run_dir      : 训练输出目录（checkpoints 等）
#   config_path  : LLaMA-Factory YAML 训练配置文件
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
  echo "[llamafactory] config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

LOG_FILE="${LOG_DIR}/train.log"

# ── logger 透传 ───────────────────────────────────────────
case "$LOGGER_TYPE" in
  wandb)
    export REPORT_TO="wandb"
    ;;
  swanlab)
    export REPORT_TO="swanlab"
    ;;
  local|*)
    export REPORT_TO="none"
    ;;
esac

echo "[llamafactory] run_dir=$RUN_DIR config=$CONFIG_PATH log=$LOG_FILE logger=$LOGGER_TYPE"

# ── 启动训练 ──────────────────────────────────────────────
llamafactory-cli train \
  --output_dir "$RUN_DIR" \
  "$CONFIG_PATH" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[llamafactory] training finished, exit_code=$EXIT_CODE"
exit "$EXIT_CODE"
