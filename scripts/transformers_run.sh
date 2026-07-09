#!/usr/bin/env bash
# transformers_run.sh — Transformers Trainer 训练启动脚本
# 统一接口: bash transformers_run.sh <run_dir> <config_path> <log_dir> <logger_type>
#   run_dir      : 训练输出目录（checkpoints 等）
#   config_path  : YAML 配置文件（含 model_name_or_path, dataset_dir, hyperparams 等）
#   log_dir      : 日志目录
#   logger_type  : local | wandb | swanlab
#
# 本脚本通过 accelerate launch 驱动一个内联 Python 训练脚本，
# 该脚本读取 YAML 配并使用 transformers.Trainer 执行训练。
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
  echo "[transformers] config not found: $CONFIG_PATH" >&2
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

echo "[transformers] run_dir=$RUN_DIR config=$CONFIG_PATH log=$LOG_FILE logger=$LOGGER_TYPE"

# ── 内联训练脚本 ──────────────────────────────────────────
TRAIN_SCRIPT="${LOG_DIR}/_transformers_train.py"

cat > "$TRAIN_SCRIPT" << 'PYEOF'
import os, sys, yaml
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)
from datasets import load_dataset

cfg_path   = sys.argv[1]
output_dir = sys.argv[2]
report_to  = os.environ.get("REPORT_TO", "none")

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

model_name = cfg["model_name_or_path"]
dataset_path = cfg.get("dataset_dir", cfg.get("dataset_path", ""))
max_steps  = cfg.get("max_steps", 10)
lr         = cfg.get("learning_rate", 2e-5)
batch_size = cfg.get("per_device_train_batch_size", 1)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

ds = load_dataset("text", data_files={"train": dataset_path})["train"]

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tokenize_fn, batched=True)

args = TrainingArguments(
    output_dir=output_dir,
    max_steps=max_steps,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    save_steps=max_steps,
    report_to=[report_to] if report_to != "none" else "none",
    save_strategy="no",
)

trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)
trainer.train()
trainer.save_model(output_dir)
print("[transformers] training done")
PYEOF

# ── 启动训练 ──────────────────────────────────────────────
accelerate launch "$TRAIN_SCRIPT" "$CONFIG_PATH" "$RUN_DIR" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[transformers] training finished, exit_code=$EXIT_CODE"
exit "$EXIT_CODE"
