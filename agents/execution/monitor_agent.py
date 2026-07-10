"""Monitor Agent（LLM驱动，只读，不可修改配置）。

每轮`poll_once()`：从对应`log_adapters`读六字段归一化指标 + 尝试`nvidia-smi`
获取GPU状态 → 追加一行到`runs/<run_id>/metrics.csv`（趋势由本Agent自己逐轮
积累，不依赖wandb/swanlab自带历史）→ 把归一化数据+最近几轮趋势喂给LLM做
全面分析 → 写`runs/<run_id>/monitor_reports/report_<seq>.md`。
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.execution.schemas import MonitorReportOutput
from agents.log_adapters.base_log_adapter import BaseLogAdapter, NormalizedMetrics
from agents.log_adapters.local_log_adapter import LocalLogAdapter
from agents.log_adapters.swanlab_log_adapter import SwanlabLogAdapter
from agents.log_adapters.wandb_log_adapter import WandbLogAdapter
from agents.planning.io_utils import write_markdown
from agents.planning.prompt_utils import format_kv_block, schema_instruction

_ADAPTER_BY_LOGGER: dict[str, type[BaseLogAdapter]] = {
    "local": LocalLogAdapter,
    "wandb": WandbLogAdapter,
    "swanlab": SwanlabLogAdapter,
}

_METRICS_CSV_HEADER = [
    "seq",
    "epoch",
    "loss",
    "metric",
    "speed",
    "memory",
    "checkpoint_path",
    "gpu_util",
    "gpu_mem_used_mb",
]


def read_gpu_status() -> dict[str, float | None]:
    """尽力调用`nvidia-smi`获取GPU利用率/显存占用，不可用时返回全None（不抛异常）。"""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"gpu_util": None, "gpu_mem_used_mb": None}

    if proc.returncode != 0 or not proc.stdout.strip():
        return {"gpu_util": None, "gpu_mem_used_mb": None}

    first_line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) != 2:
        return {"gpu_util": None, "gpu_mem_used_mb": None}
    try:
        return {"gpu_util": float(parts[0]), "gpu_mem_used_mb": float(parts[1])}
    except ValueError:
        return {"gpu_util": None, "gpu_mem_used_mb": None}


def _next_seq(run_dir: Path) -> int:
    reports_dir = run_dir / "monitor_reports"
    if not reports_dir.exists():
        return 1
    existing = list(reports_dir.glob("report_*.md"))
    return len(existing) + 1


def _append_metrics_row(
    run_dir: Path, seq: int, metrics: NormalizedMetrics, gpu_status: dict[str, float | None]
) -> None:
    metrics_csv = run_dir / "metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not metrics_csv.exists()
    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(_METRICS_CSV_HEADER)
        writer.writerow(
            [
                seq,
                metrics.epoch,
                metrics.loss,
                metrics.metric,
                metrics.speed,
                metrics.memory,
                metrics.checkpoint_path,
                gpu_status.get("gpu_util"),
                gpu_status.get("gpu_mem_used_mb"),
            ]
        )


def _read_recent_trend(run_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        return []
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:]


def _render_report_markdown(seq: int, report: MonitorReportOutput) -> str:
    return (
        f"# Monitor Report #{seq}\n\n"
        f"**风险等级**: {report.risk_level}\n"
        f"**崩溃检测**: {report.crash_detected}\n\n"
        f"## GPU运行情况\n{report.gpu_observation}\n\n"
        f"## Loss下降趋势\n{report.loss_trend}\n\n"
        f"## 早熟/过拟合迹象\n{report.overfitting_signal}\n\n"
        f"## 验证精度\n{report.validation_accuracy}\n\n"
        f"## 其他异常\n{report.other_anomalies or '无'}\n\n"
        f"## 建议措施\n{report.recommendation}\n"
    )


class MonitorAgent(BaseAgent):
    agent_id = "monitor"
    output_schema = MonitorReportOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名训练监控分析专家。基于给定的归一化指标（当前值+最近几轮趋势）"
            "与GPU状态，从以下维度分析：(1)GPU运行情况（利用率/显存/异常波动）；"
            "(2)Loss下降趋势（平稳下降/停滞/发散）；(3)早熟/过拟合迹象（train/val "
            "差距扩大、验证指标不再提升但训练loss仍下降等）；(4)验证精度（与目标"
            "差距、变化趋势）；(5)其他可观测异常。给出risk_level（Normal/Warning/"
            "Critical），若判断为明显早熟、loss发散、GPU显存持续异常等严重问题，"
            "risk_level应为Critical；若日志内容显示训练已实质性崩溃（而非仅仅是"
            "指标不理想），crash_detected应为true。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        metrics: NormalizedMetrics = kwargs["metrics"]
        gpu_status: dict[str, float | None] = kwargs["gpu_status"]
        trend: list[dict[str, Any]] = kwargs["trend"]
        indicators: str = kwargs.get("indicators", "无特殊要求")

        context = format_kv_block(
            "本轮监控输入",
            {
                "本轮归一化指标": metrics.__dict__,
                "GPU状态": gpu_status,
                "最近几轮趋势": trend,
                "用户目标指标": indicators,
            },
        )
        return f"{context}\n\n请给出本轮监控分析结论。\n\n{schema_instruction(self.output_schema)}"

    def poll_once(
        self,
        run_id: str,
        log_dir: Path,
        logger_type: str,
        indicators: str = "无特殊要求",
        runs_root: Path = Path("runs"),
    ) -> MonitorReportOutput:
        adapter_cls = _ADAPTER_BY_LOGGER.get(logger_type)
        if adapter_cls is None:
            raise ValueError(f"未知logger_type: {logger_type!r}")

        run_dir = runs_root / run_id
        metrics = adapter_cls().read_latest_metrics(Path(log_dir))
        gpu_status = read_gpu_status()

        seq = _next_seq(run_dir)
        _append_metrics_row(run_dir, seq, metrics, gpu_status)
        trend = _read_recent_trend(run_dir)

        result = self.run(metrics=metrics, gpu_status=gpu_status, trend=trend, indicators=indicators)
        assert result.structured_output is not None
        report = MonitorReportOutput(**result.structured_output)

        report_path = run_dir / "monitor_reports" / f"report_{seq}.md"
        write_markdown(report_path, _render_report_markdown(seq, report))
        return report
