#!/usr/bin/env bash
# verl_run.sh — verl 训练启动脚本（占位）
# 统一接口: bash verl_run.sh <run_dir> <config_path> <log_dir> <logger_type>
#
# 当前环境未安装 verl，此脚本仅输出提示信息并以非零退出码退出。
# 用户后续可 pip install verl 后替换占位逻辑为真实训练命令。
set -euo pipefail

echo "[verl] verl is not installed in the current environment." >&2
echo "[verl] To enable verl training, please run: pip install verl" >&2
echo "[verl] Then replace this placeholder script with the actual verl launch command." >&2
exit 2
