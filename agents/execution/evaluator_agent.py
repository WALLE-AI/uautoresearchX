"""Evaluator Agent：解析归一化指标+Monitor报告，判定PASS/FAIL，维护
`improve_guide.md`逐轮追加。

"崩溃"判定采用双信号：训练子进程真实退出码（硬信号，由调用方检测后传入
`process_exit_code`）或Monitor报告的`crash_detected`（软信号）任一为真，直接
判FAIL，不必等待/依赖LLM去"发现"进程已经死了；否则走LLM判定PASS/FAIL。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.execution.schemas import EvaluatorOutput
from agents.planning.io_utils import append_markdown_log
from agents.planning.prompt_utils import format_kv_block, schema_instruction


def _read_recent_monitor_reports(run_dir: Path, limit: int = 3) -> list[str]:
    reports_dir = run_dir / "monitor_reports"
    if not reports_dir.exists():
        return []
    files = sorted(reports_dir.glob("report_*.md"), key=lambda p: p.stat().st_mtime)
    return [f.read_text(encoding="utf-8") for f in files[-limit:]]


def _render_improve_guide_section(
    iteration: int, hyperparams_snapshot: dict[str, Any], result: EvaluatorOutput
) -> str:
    suggestions = "\n".join(f"- {s}" for s in result.improvement_suggestions) or "- 无"
    timestamp = datetime.now(timezone.utc).isoformat()
    return (
        f"## 迭代 #{iteration} - {timestamp}\n\n"
        f"**超参快照**: {hyperparams_snapshot}\n"
        f"**评测结论**: {result.verdict}\n"
        f"**与目标差距分析**: {result.gap_analysis}\n"
        f"**改进建议**:\n{suggestions}\n"
        f"**是否需要回退规划**: {'是' if result.needs_replanning else '否'}\n"
    )


class EvaluatorAgent(BaseAgent):
    agent_id = "evaluator"
    output_schema = EvaluatorOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名训练评测专家。基于最近几轮的归一化指标趋势与Monitor Agent的"
            "分析报告，判定本轮训练是否达到用户目标指标：verdict=PASS表示达标，"
            "FAIL表示未达标。FAIL时给出gap_analysis（差距分析）与"
            "improvement_suggestions（超参/数据/模型层面的具体改进建议）。"
            "needs_replanning用于区分FAIL的根源：若是超参级问题（可通过调整学习率"
            "/batch size/epoch等重训解决），应为false，优先回退给Trainer重训；"
            "若判断为规划层问题（模型选型错误、资源明显不足、任务定义与数据不"
            "匹配等重训无法解决），应为true，需回退到Planning重新规划。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        trend: list[dict[str, Any]] = kwargs.get("trend", [])
        recent_reports: list[str] = kwargs.get("recent_reports", [])
        indicators: str = kwargs.get("indicators", "无特殊要求")

        context = format_kv_block(
            "评测输入",
            {
                "用户目标指标": indicators,
                "最近几轮归一化指标趋势": trend,
                "最近Monitor分析报告数": len(recent_reports),
            },
        )
        reports_block = "\n---\n".join(recent_reports) if recent_reports else "（暂无Monitor报告）"
        return (
            f"{context}\n\n最近Monitor分析报告全文：\n{reports_block}\n\n"
            f"请给出评测结论。\n\n{schema_instruction(self.output_schema)}"
        )

    def evaluate(
        self,
        run_id: str,
        iteration: int,
        indicators: str,
        hyperparams_snapshot: dict[str, Any],
        trend: list[dict[str, Any]] | None = None,
        process_exit_code: int | None = None,
        crash_detected: bool = False,
        runs_root: Path = Path("runs"),
    ) -> EvaluatorOutput:
        run_dir = runs_root / run_id

        if (process_exit_code is not None and process_exit_code != 0) or crash_detected:
            result = EvaluatorOutput(
                verdict="FAIL",
                gap_analysis=(
                    f"训练进程异常终止(exit_code={process_exit_code})或Monitor检测到崩溃，"
                    "未能产出有效指标。"
                ),
                improvement_suggestions=["检查配置/资源是否正确，修正后重试当前阶段"],
                needs_replanning=False,
            )
        else:
            recent_reports = _read_recent_monitor_reports(run_dir)
            trend_data = trend if trend is not None else []
            llm_result = self.run(
                trend=trend_data, recent_reports=recent_reports, indicators=indicators
            )
            assert llm_result.structured_output is not None
            result = EvaluatorOutput(**llm_result.structured_output)

        append_markdown_log(
            run_dir / "improve_guide.md",
            _render_improve_guide_section(iteration, hyperparams_snapshot, result),
        )
        return result
