"""orchestrator/human_gate.py单元测试：`AutoHumanGate`默认行为 +
`InteractiveHumanGate`的终端交互分支（用monkeypatch替换rich的Prompt/Confirm，
不需要真正的tty）。
"""

from __future__ import annotations

from agents.execution.schemas import EvaluatorOutput
from agents.planning.schemas import (
    DataFormatSpec,
    PlanReviewOutput,
    ReviewIssue,
    TrainingPlanOutput,
)
from orchestrator.human_gate import AutoHumanGate, InteractiveHumanGate

_PLAN = TrainingPlanOutput(
    markdown="# plan",
    resource_plan={},
    pipeline_stages=[],
    data_format=DataFormatSpec(target_format="ShareGPT", rationale="test", field_mapping=[]),
)


def test_auto_human_gate_always_accepts_llm_verdict() -> None:
    gate = AutoHumanGate()
    review = PlanReviewOutput(approved=False, issues=[], summary="拒绝")
    evaluator_output = EvaluatorOutput(
        verdict="FAIL", gap_analysis="差距", improvement_suggestions=[], needs_replanning=False
    )

    assert gate.review_plan(_PLAN, review) == "accept_llm_verdict"
    assert gate.on_stage_fail(evaluator_output) == "accept_llm_verdict"
    assert gate.on_max_retries_exceeded("已达上限") == "abort"


def test_interactive_human_gate_review_plan_accepts_by_default(monkeypatch) -> None:
    monkeypatch.setattr("orchestrator.human_gate.Confirm.ask", lambda *a, **k: True)
    gate = InteractiveHumanGate()
    review = PlanReviewOutput(approved=True, issues=[], summary="通过")
    assert gate.review_plan(_PLAN, review) == "accept_llm_verdict"


def test_interactive_human_gate_review_plan_force_rejects_when_user_declines() -> None:
    class _Gate(InteractiveHumanGate):
        pass

    gate = _Gate()
    gate.console.print = lambda *a, **k: None  # noqa: ARG005 - 静默输出，不测渲染内容

    import orchestrator.human_gate as hg_module

    original_ask = hg_module.Confirm.ask
    hg_module.Confirm.ask = staticmethod(lambda *a, **k: False)  # type: ignore[method-assign]
    try:
        review = PlanReviewOutput(
            approved=True, issues=[ReviewIssue(category="计划参数", description="x")], summary="通过"
        )
        assert gate.review_plan(_PLAN, review) == "force_reject"

        review_rejected = PlanReviewOutput(approved=False, issues=[], summary="拒绝")
        assert gate.review_plan(_PLAN, review_rejected) == "force_approve"
    finally:
        hg_module.Confirm.ask = original_ask


def test_interactive_human_gate_on_stage_fail_maps_choices() -> None:
    import orchestrator.human_gate as hg_module

    gate = InteractiveHumanGate()
    gate.console.print = lambda *a, **k: None  # noqa: ARG005
    evaluator_output = EvaluatorOutput(
        verdict="FAIL", gap_analysis="差距", improvement_suggestions=["建议"], needs_replanning=False
    )

    original_ask = hg_module.Prompt.ask
    for choice, expected in [
        ("accept", "accept_llm_verdict"),
        ("retry", "retry"),
        ("replan", "replan"),
        ("abort", "abort"),
    ]:
        hg_module.Prompt.ask = staticmethod(lambda *a, choice=choice, **k: choice)  # type: ignore[method-assign]
        assert gate.on_stage_fail(evaluator_output) == expected
    hg_module.Prompt.ask = original_ask


def test_interactive_human_gate_on_max_retries_exceeded() -> None:
    import orchestrator.human_gate as hg_module

    gate = InteractiveHumanGate()
    gate.console.print = lambda *a, **k: None  # noqa: ARG005

    original_ask = hg_module.Confirm.ask
    hg_module.Confirm.ask = staticmethod(lambda *a, **k: True)  # type: ignore[method-assign]
    try:
        assert gate.on_max_retries_exceeded("上限") == "extend_retries"
    finally:
        hg_module.Confirm.ask = original_ask
