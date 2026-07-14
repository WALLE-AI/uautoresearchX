#!/usr/bin/env bash
# CLI命令功能验收测试脚本：用真实opencode引擎 + SVRDD_YOLO路面病害检测数据集
# 小规模跑通一次训练闭环，验证 uautoresearchx run/list/status/logs/cancel/resume
# 是否可用。这是CLI机制的验收测试，不是追求最终模型精度——因此只抽样一小部分
# 数据、限制epoch数，让训练阶段几分钟内跑完。
#
# 用法：
#   bash test_cli_svrdd_opencode.sh
#
# 建议配合第二个终端一起看效果（本脚本的Phase 3会打印可以在另一个终端里
# 手动尝试的命令，比如训练进行中查看 list/status/logs -f，或Ctrl-C后resume）。
#
# 前置假设（已核实）：
#   - opencode CLI (1.14.40) 已安装且可用
#   - 数据集: /home/dataset1/gaojing/xibeiyuan/datasets/SVRDD/SVRDD_YOLO
#     （YOLO txt格式，7类路面病害，train 6000/val 1000/test 1000张，1024x1024）
#   - 本地YOLO系列权重: /home/dataset1/gaojing/xibeiyuan/models/{yolo26n,yolo11x}/*.pt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_ROOT="/home/dataset1/gaojing/xibeiyuan/datasets/SVRDD/SVRDD_YOLO"
MODELS_ROOT="/home/dataset1/gaojing/xibeiyuan/models"
RUN_ID="run_svrdd_opencode_$(date +%H%M%S)"

# 当前环境全局设了http_proxy/https_proxy但没有给127.0.0.1/localhost加豁免，
# 会导致opencode连接本地vLLM(127.0.0.1:8087, Qwen3.5-35B-A3B)的请求被错误
# 转发到外部代理，进而在`opencode acp`的session/new握手阶段直接失败
# （JSON-RPC -32603 Internal error）。这里显式豁免，不依赖调用方提前手动export。
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

echo "================================================================"
echo "Phase 0: 环境检查（只读）"
echo "================================================================"
if ! command -v opencode >/dev/null 2>&1; then
    echo "错误: 未找到 opencode CLI，请先确认已安装并在PATH中。" >&2
    exit 1
fi
opencode --version
# scripts/ultralytics_run.sh直接调用系统PATH里的`yolo` CLI，不经过`uv run`的
# 项目venv（该venv未装ultralytics属预期），因此这里检查`yolo`而非`uv run python
# -c "import ultralytics"`（后者在本环境下必然报ModuleNotFoundError，属于
# 无意义的假警报）。
if command -v yolo >/dev/null 2>&1; then
    yolo version
else
    echo "警告: 未在PATH中找到yolo CLI，训练阶段的scripts/ultralytics_run.sh会失败" >&2
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>&1 || echo "（nvidia-smi不可用，忽略）"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "错误: 数据集目录不存在: $DATASET_ROOT" >&2
    exit 1
fi
if [ ! -d "$MODELS_ROOT" ]; then
    echo "错误: 模型目录不存在: $MODELS_ROOT" >&2
    exit 1
fi

echo
echo "================================================================"
echo "Phase 1: 把 configs/agents.yaml 全部10个Agent切到 opencode 引擎"
echo "================================================================"
if [ ! -f configs/agents.yaml.bak ]; then
    cp configs/agents.yaml configs/agents.yaml.bak
    echo "已备份原配置到 configs/agents.yaml.bak"
else
    echo "configs/agents.yaml.bak 已存在，跳过备份（说明之前已经切换过）"
fi
sed -i 's/engine: claude/engine: opencode/; s/engine: codex/engine: opencode/' configs/agents.yaml
echo "当前各Agent engine配置："
grep -E "^\S|engine:" configs/agents.yaml | grep -B1 "engine:" | grep -v "^--"
echo
echo "（测试结束后如需恢复原配置: cp configs/agents.yaml.bak configs/agents.yaml）"

echo
echo "================================================================"
echo "Phase 2: 准备小规模抽样数据集（80张train + 20张val，非全量）"
echo "================================================================"
if [ ! -f tools/svrdd_dataset_records.json ]; then
    echo "未找到完整records.json，先生成（遍历7000张图片读取尺寸，可能需要几分钟）..."
    uv run python tools/prepare_svrdd_records.py
fi
uv run python tools/sample_svrdd_records.py --train 80 --val 20

DATASET_SAMPLE_TEXT="$(cat tools/svrdd_dataset_sample_summary.txt)"

echo
echo "================================================================"
echo "Phase 3: 发起训练闭环（前台运行，实时TUI默认开启）"
echo "================================================================"
echo "run_id 将固定为: $RUN_ID"
echo
echo ">>> 可以在训练进行中打开另一个终端，尝试以下命令观察CLI管理能力 <<<"
echo "  uv run uautoresearchx list"
echo "  uv run uautoresearchx status $RUN_ID"
echo "  uv run uautoresearchx logs $RUN_ID --follow"
echo ">>> 也可以在此终端里直接 Ctrl-C 中断本次run（训练子进程若已启动不会被"
echo "    连带杀死，因为已独立于CLI进程组），然后用以下命令验证resume能否"
echo "    正确接管继续跑，而不是从头重来: <<<"
echo "  uv run uautoresearchx resume $RUN_ID"
echo ">>> 如果想彻底终止这次运行（含仍在跑的训练进程）: <<<"
echo "  uv run uautoresearchx cancel $RUN_ID"
echo

TASK_DESCRIPTION="在SVRDD道路路面病害检测数据集上训练一个目标检测模型，识别纵向裂缝/横向裂缝/龟裂/坑槽/纵向修补/横向修补/井盖共7类路面病害。目标是后续可持续迭代提高检测精度（mAP）。本地可用YOLO系列基础模型权重（无网络下载新权重的假设，请从下列本地路径中选择起点权重）：${MODELS_ROOT}/yolo26n/yolo26n.pt (YOLO26n)、${MODELS_ROOT}/yolo11x/yolo11x.pt (YOLO11x)。本次是uautoresearchx CLI命令功能验收测试（验证run/list/status/logs/cancel/resume是否可用），不是追求最终模型精度，请使用较小分辨率与尽量少的epoch数（1-3）快速跑完当前阶段；training_plan中可以说明后续换回全量数据集+更多训练轮次以持续提升精度的迭代思路，但当前阶段只需给出一个能快速跑完的最小方案。"

INDICATORS="全流程无未处理异常跑到DONE；产出至少一个有效checkpoint文件；mAP非0即可，不要求收敛"
RESOURCE_CONSTRAINTS="仅限使用1张GPU，训练阶段总时长不超过20分钟，本次目的是CLI功能验证而非模型质量"
AVAILABLE_RESOURCES="1x NVIDIA A100-SXM4-40GB"

uv run uautoresearchx run \
    --run-id "$RUN_ID" \
    --task-description "$TASK_DESCRIPTION" \
    --dataset-path "$DATASET_ROOT" \
    --dataset-sample "$DATASET_SAMPLE_TEXT" \
    --dataset-records-file tools/svrdd_dataset_records_sample.json \
    --indicators "$INDICATORS" \
    --resource-constraints "$RESOURCE_CONSTRAINTS" \
    --available-resources "$AVAILABLE_RESOURCES" \
    --logger-type local \
    --interval-minutes 1
RUN_EXIT_CODE=$?

echo
echo "================================================================"
echo "Phase 4: run结束（退出码=$RUN_EXIT_CODE），产物核对建议"
echo "================================================================"
echo "uv run uautoresearchx status $RUN_ID"
echo "uv run uautoresearchx list"
echo "cat runs/$RUN_ID/training_plan.md"
echo "cat logs/$RUN_ID/local/train.log        # 若已进入训练阶段"
echo "ls runs/$RUN_ID/monitor_reports/ 2>/dev/null"
echo
echo "恢复原始CLI引擎配置（若不再需要opencode）："
echo "  cp configs/agents.yaml.bak configs/agents.yaml"

exit "$RUN_EXIT_CODE"
