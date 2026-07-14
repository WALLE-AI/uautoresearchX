"""可选的人工确认层：`StateMachine`在三类关键判定点——Plan Review结论、
Evaluator FAIL后的回退方向、达到最大重试次数——会调用`HumanGate`。

默认实现`AutoHumanGate`原样采纳LLM的判定，`StateMachine`行为与"支持CLI模式"
之前完全自动化的流程零差异；只有显式传入`InteractiveHumanGate`（`run
--interactive`）时才会真正暂停等待终端输入，这是一个纯附加的能力，不影响
默认路径。
"""

from __future__ import annotations

from typing import Literal, Protocol

from rich.console import Console
from rich.prompt import Confirm, Prompt

from agents.execution.schemas import EvaluatorOutput
from agents.planning.schemas import PlanReviewOutput, TrainingPlanOutput

PlanReviewDecision = Literal["accept_llm_verdict", "force_approve", "force_reject"]
StageFailDecision = Literal["accept_llm_verdict", "retry", "replan", "abort"]
RetryLimitDecision = Literal["abort", "extend_retries"]


class HumanGate(Protocol):
    def review_plan(self, plan: TrainingPlanOutput, review: PlanReviewOutput) -> PlanReviewDecision: ...

    def on_stage_fail(self, evaluator_output: EvaluatorOutput) -> StageFailDecision: ...

    def on_max_retries_exceeded(self, context: str) -> RetryLimitDecision: ...


class AutoHumanGate:
    """默认实现：始终采纳LLM原有判定。"""

    def review_plan(self, plan: TrainingPlanOutput, review: PlanReviewOutput) -> PlanReviewDecision:
        return "accept_llm_verdict"

    def on_stage_fail(self, evaluator_output: EvaluatorOutput) -> StageFailDecision:
        return "accept_llm_verdict"

    def on_max_retries_exceeded(self, context: str) -> RetryLimitDecision:
        return "abort"


class InteractiveHumanGate:
    """`run --interactive`接入：在终端打印决策上下文并等待用户输入。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def review_plan(self, plan: TrainingPlanOutput, review: PlanReviewOutput) -> PlanReviewDecision:
        self.console.print("\n[bold]— Plan Review 结论 —[/bold]")
        self.console.print(f"LLM判定: {'通过' if review.approved else '拒绝'}；{review.summary}")
        for issue in review.issues:
            self.console.print(f"  - [{issue.category}] {issue.description}")
        if Confirm.ask("是否采纳LLM判定？", default=True):
            return "accept_llm_verdict"
        return "force_reject" if review.approved else "force_approve"

    def on_stage_fail(self, evaluator_output: EvaluatorOutput) -> StageFailDecision:
        self.console.print("\n[bold]— 训练阶段FAIL —[/bold]")
        self.console.print(f"差距分析: {evaluator_output.gap_analysis}")
        for suggestion in evaluator_output.improvement_suggestions:
            self.console.print(f"  - 建议: {suggestion}")
        self.console.print(
            f"LLM判定{'需要' if evaluator_output.needs_replanning else '不需要'}回退重新规划"
        )
        choice = Prompt.ask(
            "如何处理？(accept=采纳LLM判定 / retry=重试当前阶段 / replan=回退重新规划 / abort=终止运行)",
            choices=["accept", "retry", "replan", "abort"],
            default="accept",
        )
        mapping: dict[str, StageFailDecision] = {
            "accept": "accept_llm_verdict",
            "retry": "retry",
            "replan": "replan",
            "abort": "abort",
        }
        return mapping[choice]

    def on_max_retries_exceeded(self, context: str) -> RetryLimitDecision:
        self.console.print(f"\n[bold red]— 已达最大重试次数 —[/bold red]\n{context}")
        if Confirm.ask("是否延长一次重试次数继续尝试？", default=False):
            return "extend_retries"
        return "abort"
