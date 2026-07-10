"""解析SwanLab本地/离线模式在磁盘产出的同步文件。

实现说明：SwanLab（本地模式，`swanlab.init(mode="local")`或离线同步）会在
`<logdir>/<project>/<experiment_id>/`下为每个被记录的metric创建一份
`logs/<tag>.log`（JSONL，每行形如`{"index": n, "data": value}`，`tag`即
`swanlab.log({...})`调用时dict的key名，如`loss`/`epoch`/`eval/accuracy`）。
本适配器glob该目录结构，为每个已知六字段候选key名查找对应`<tag>.log`并取
最后一行的`data`值。

**未验证声明**：与`wandb_log_adapter.py`同理，本模块在实现阶段应先用真实
SwanLab本地模式跑一次最小demo核实实际目录/文件结构，但受限于当前会话中
Bash工具暂时不可用，未能完成验证。以上结构基于SwanLab公开文档描述的通用
行为归纳，**生产使用前必须用当前环境实际安装的swanlab版本重新核实**，若
实测发现结构不同，只需替换本文件内的glob pattern与解析逻辑，不影响上层
`BaseLogAdapter`接口。
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.log_adapters.base_log_adapter import BaseLogAdapter, NormalizedMetrics

_LOSS_TAGS = ("loss", "train/loss", "train_loss")
_EPOCH_TAGS = ("epoch", "train/epoch")
_METRIC_TAGS = ("eval/loss", "eval_loss", "eval/accuracy", "eval_accuracy")
_SPEED_TAGS = ("train/samples_per_second", "samples_per_second")
_MEMORY_TAGS = ("gpu_memory", "train/gpu_memory_gb")


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_last_value(experiment_dir: Path, tags: tuple[str, ...]) -> float | None:
    for tag in tags:
        log_file = experiment_dir / "logs" / f"{tag.replace('/', '_')}.log"
        if not log_file.exists():
            continue
        last_line: str | None = None
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                last_line = line
        if last_line is None:
            continue
        try:
            record = json.loads(last_line)
        except json.JSONDecodeError:
            continue
        value = record.get("data") if isinstance(record, dict) else None
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _find_latest_experiment_dir(swanlab_root: Path) -> Path | None:
    candidates = [p for p in swanlab_root.glob("**/logs") if p.is_dir()]
    if not candidates:
        return None
    latest_logs_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest_logs_dir.parent


class SwanlabLogAdapter(BaseLogAdapter):
    def read_latest_metrics(self, log_dir: Path) -> NormalizedMetrics:
        swanlab_root = Path(log_dir) / "swanlab"
        if not swanlab_root.exists():
            return NormalizedMetrics()

        experiment_dir = _find_latest_experiment_dir(swanlab_root)
        if experiment_dir is None:
            return NormalizedMetrics()

        return NormalizedMetrics(
            epoch=_read_last_value(experiment_dir, _EPOCH_TAGS),
            loss=_read_last_value(experiment_dir, _LOSS_TAGS),
            metric=_read_last_value(experiment_dir, _METRIC_TAGS),
            speed=_read_last_value(experiment_dir, _SPEED_TAGS),
            memory=_read_last_value(experiment_dir, _MEMORY_TAGS),
            checkpoint_path=None,
        )
