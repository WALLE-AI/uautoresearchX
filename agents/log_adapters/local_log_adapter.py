"""解析`scripts/<engine>_run.sh`统一产出的本地日志文件：`<log_dir>/train.log`
（各训练脚本把训练框架原始stdout通过`tee`写入该文件，不是设计文档最初设想的
六字段结构化格式）。

本适配器采用best-effort策略从尾部按行扫描，尝试匹配两类已知的常见输出：
    1. HuggingFace Trainer / TRL 风格的字典日志行，如：
       {'loss': 1.234, 'learning_rate': 2e-05, 'epoch': 0.5}
    2. Ultralytics 风格的训练进度表格行，如：
         1/10      2.1G      1.234      0.567      0.891         16        640
不属于以上两种已知格式的日志（如llamafactory的自定义输出）目前无法可靠抽取，
对应字段留空——这是有意的降级行为而非bug，抽取不到的信息仍可由Monitor Agent
在prompt中直接看到日志尾部原文自行判断。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from agents.log_adapters.base_log_adapter import BaseLogAdapter, NormalizedMetrics

_MAX_TAIL_LINES = 200

_DICT_LINE_RE = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
_ULTRALYTICS_ROW_RE = re.compile(
    r"^\s*\d+/\d+\s+([\d.]+)\w?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+"
)
_CHECKPOINT_RE = re.compile(r"(?:checkpoint|saved to|save_model)[^\n]*?([\w./\\-]+\.(?:pt|bin|safetensors)|checkpoint-\d+)", re.IGNORECASE)


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


def _try_parse_hf_dict_line(lines: list[str]) -> NormalizedMetrics | None:
    for line in reversed(lines):
        match = _DICT_LINE_RE.search(line)
        if not match:
            continue
        try:
            data = ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            continue
        if not isinstance(data, dict):
            continue
        return NormalizedMetrics(
            epoch=_to_float(data.get("epoch")),
            loss=_to_float(data.get("loss")),
            metric=_to_float(data.get("eval_loss") or data.get("eval_accuracy")),
            speed=_to_float(data.get("train_samples_per_second") or data.get("samples_per_second")),
            memory=None,
            checkpoint_path=None,
        )
    return None


def _try_parse_ultralytics_row(lines: list[str]) -> NormalizedMetrics | None:
    for line in reversed(lines):
        match = _ULTRALYTICS_ROW_RE.match(line)
        if not match:
            continue
        gpu_mem, box_loss, cls_loss, dfl_loss = match.groups()
        return NormalizedMetrics(
            epoch=None,
            loss=_to_float(box_loss),
            metric=None,
            speed=None,
            memory=_to_float(gpu_mem),
            checkpoint_path=None,
        )
    return None


def _find_checkpoint_path(lines: list[str]) -> str | None:
    for line in reversed(lines):
        match = _CHECKPOINT_RE.search(line)
        if match:
            return match.group(1)
    return None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LocalLogAdapter(BaseLogAdapter):
    def read_latest_metrics(self, log_dir: Path) -> NormalizedMetrics:
        log_file = Path(log_dir) / "train.log"
        if not log_file.exists():
            return NormalizedMetrics()

        lines = _tail_lines(log_file, _MAX_TAIL_LINES)
        if not lines:
            return NormalizedMetrics()

        result = _try_parse_hf_dict_line(lines) or _try_parse_ultralytics_row(lines) or NormalizedMetrics()
        result.checkpoint_path = _find_checkpoint_path(lines)
        return result
