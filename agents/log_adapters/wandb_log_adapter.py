"""解析wandb离线同步(`WANDB_MODE=offline`)在本地磁盘产出的`wandb-summary.json`。

实现说明（未在本次会话中实测，见下方"未验证声明"）：
wandb离线模式下，每次run会在`<WANDB_DIR>/wandb/`下创建一个
`offline-run-<timestamp>-<run_id>/`（或在线模式下`run-<timestamp>-<run_id>/`）
目录，其中`files/wandb-summary.json`是一份随训练过程持续更新的扁平JSON，
key为metric名（如`loss`/`train/loss`/`epoch`/`eval/loss`等，具体key名由训练
脚本调用`wandb.log(...)`时传入的dict决定），value为最新一次记录的标量值。
本适配器glob该目录下所有`files/wandb-summary.json`，取修改时间最新的一份解析。

**未验证声明**：本模块在实现阶段本应先用`wandb.init(mode="offline")`跑一次
最小demo核实上述路径/字段名（见`多智能体训练框架-架构审阅与修订计划-v1.1.md`
中关于日志适配器格式需要实测的记录），但受限于当前会话中Bash工具暂时不可用，
未能完成这一验证步骤。以上路径/字段假设基于wandb公开文档与既有版本的通用行为，
**生产使用前必须用当前环境实际安装的wandb版本重新核实**，不应视为已实测结论。
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.log_adapters.base_log_adapter import BaseLogAdapter, NormalizedMetrics

_LOSS_KEYS = ("loss", "train/loss", "train_loss")
_EPOCH_KEYS = ("epoch", "train/epoch")
_METRIC_KEYS = ("eval/loss", "eval_loss", "eval/accuracy", "eval_accuracy")
_SPEED_KEYS = ("train/train_samples_per_second", "train_samples_per_second", "samples_per_second")
_MEMORY_KEYS = ("system/gpu.0.memoryAllocated", "gpu_memory", "train/gpu_memory_gb")
_CHECKPOINT_KEYS = ("checkpoint_path", "output_dir")


def _first_present(data: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class WandbLogAdapter(BaseLogAdapter):
    def read_latest_metrics(self, log_dir: Path) -> NormalizedMetrics:
        wandb_root = Path(log_dir) / "wandb"
        if not wandb_root.exists():
            return NormalizedMetrics()

        summary_files = list(wandb_root.glob("**/files/wandb-summary.json"))
        if not summary_files:
            return NormalizedMetrics()

        latest_file = max(summary_files, key=lambda p: p.stat().st_mtime)
        try:
            data = json.loads(latest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return NormalizedMetrics()
        if not isinstance(data, dict):
            return NormalizedMetrics()

        checkpoint = _first_present(data, _CHECKPOINT_KEYS)
        return NormalizedMetrics(
            epoch=_to_float(_first_present(data, _EPOCH_KEYS)),
            loss=_to_float(_first_present(data, _LOSS_KEYS)),
            metric=_to_float(_first_present(data, _METRIC_KEYS)),
            speed=_to_float(_first_present(data, _SPEED_KEYS)),
            memory=_to_float(_first_present(data, _MEMORY_KEYS)),
            checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        )
