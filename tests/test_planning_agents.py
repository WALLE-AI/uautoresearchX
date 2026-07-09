"""Planning阶段六个Agent的集成测试。

用 `tests/fakes/scripted_engine.ScriptedEngine` 逐一注入每个Agent的预设
结构化输出，串联跑通 scenario -> dataset -> model_selection ->
training_plan -> plan_reviewer -> report_writer 全流程，验证：
- 各Agent能正确产出结构化输出并在prompt中带入上游上下文
- training_plan.md / analysis_report.md 能正确写入文件系统
- Plan Reviewer拒绝时的分层回退目标判定正确
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.engines.base_engine import AgentResult
from agents.planning.dataset_analysis_agent import DatasetAnalysisAgent
from agents.planning.io_utils import append_markdown_log, write_markdown
from agents.planning.model_selection_agent import ModelSelectionAgent
from agents.planning.plan_reviewer_agent import PlanReviewerAgent, determine_rollback_target
from agents.planning.report_writer_agent import ReportWriterAgent
from agents.planning.scenario_analysis_agent import ScenarioAnalysisAgent
from agents.planning.schemas import (
    AnalysisReportOutput,
    DataFormatSpec,
    DatasetAnalysisOutput,
    ModelSelectionOutput,
    PipelineStage,
    PlanReviewOutput,
    ScenarioAnalysisOutput,
    TrainingPlanOutput,
)
from agents.planning.training_plan_generator import TrainingPlanGeneratorAgent
from tests.fakes.scripted_engine import ScriptedEngine


def _result_for(model_instance: Any) -> AgentResult:
    dumped = model_instance.model_dump()
    return AgentResult(text=json.dumps(dumped, ensure_ascii=False), structured_output=dumped)


_SCENARIO_FIXTURE = ScenarioAnalysisOutput(
    task_type="llm-sft",
    industry="客服问答",
    difficulty="medium",
    latency_constraint="无",
    priority_suggestion="高优先级",
    risks=["领域术语覆盖不足"],
    citations=["https://example.com/llm-sft-best-practice"],
)

_DATASET_FIXTURE = DatasetAnalysisOutput(
    num_samples=5000,
    spec="平均对话长度200 tokens",
    num_classes=None,
    class_distribution=None,
    quality_score=7.5,
    augmentation_suggestions=["回译增强"],
    candidate_formats=["ShareGPT", "Alpaca"],
    field_mapping_hints=["原始question/answer字段映射到ShareGPT的human/gpt"],
    confidence="medium",
    citations=["https://example.com/similar-dataset-benchmark"],
)

_MODEL_SELECTION_FIXTURE = ModelSelectionOutput(
    recommended_model="Qwen2.5-7B-Instruct",
    alternative_models=["Llama-3-8B"],
    rationale="中文客服场景效果与成本平衡",
    gpu_requirement="4x A100-40GB",
    estimated_metric="人工评估满意度≥85%",
    estimated_training_duration="6小时",
    data_format_requirements=["需ShareGPT多轮对话格式"],
    citations=["https://example.com/qwen2.5-benchmark"],
)

_TRAINING_PLAN_FIXTURE = TrainingPlanOutput(
    markdown=(
        "# 客服问答任务 - 训练计划\n\n## TL;DR（人类速览）\n**训练目标**: 客服问答SFT\n\n"
        "## 资源规划\n| 项目 | 配置 |\n| --- | --- |\n| GPU | 4x A100-40GB |\n\n"
        "## Pipeline Stages（训练流程）\n| 阶段 | 起点权重 | 训练目标 | 引擎 | 关键超参 | 预计耗时 |\n"
        "| --- | --- | --- | --- | --- | --- |\n| 1. SFT | 基础模型 | 指令遵循 | llamafactory | lr=2e-5 | 6h |\n\n"
        "## 数据格式\n**目标格式**: ShareGPT\n\n## 训练日历（分阶段）\n- 0~3 Epoch: SFT\n\n"
        "## 验证与达标标准\n- 目标指标: 满意度≥85%\n\n## 决策依据与引用来源\n- https://example.com/qwen2.5-benchmark"
    ),
    resource_plan={"GPU": "4x A100-40GB", "Batch Size": "16"},
    pipeline_stages=[
        PipelineStage(
            name="SFT",
            start_from="基础模型",
            goal="指令遵循能力对齐",
            engine="llamafactory",
            key_hyperparams="lr=2e-5, epoch=3",
            estimated_duration="6h",
        )
    ],
    data_format=DataFormatSpec(
        target_format="ShareGPT",
        rationale="Model-Selection硬性要求需多轮对话格式",
        field_mapping=[],
    ),
    decision_references=["https://example.com/qwen2.5-benchmark"],
)

_PLAN_REVIEW_APPROVED_FIXTURE = PlanReviewOutput(approved=True, issues=[], summary="计划合理，通过评审")

_ANALYSIS_REPORT_FIXTURE = AnalysisReportOutput(
    markdown="# 分析报告\n\n## 第一章 场景需求深度分析\n...\n## 第四章 训练计划总览\n...",
)


def test_full_planning_pipeline_writes_training_plan_and_report(tmp_path: Path) -> None:
    scenario_engine = ScriptedEngine([_result_for(_SCENARIO_FIXTURE)])
    scenario_agent = ScenarioAnalysisAgent(engine=scenario_engine)
    scenario_result = scenario_agent.run(task_description="构建客服问答机器人")
    assert scenario_result.structured_output is not None
    assert "客服问答机器人" in scenario_agent.build_user_prompt(task_description="构建客服问答机器人")

    dataset_engine = ScriptedEngine([_result_for(_DATASET_FIXTURE)])
    dataset_agent = DatasetAnalysisAgent(engine=dataset_engine)
    dataset_result = dataset_agent.run(
        task_description="构建客服问答机器人",
        scenario_summary=json.dumps(scenario_result.structured_output, ensure_ascii=False),
    )
    assert dataset_result.structured_output is not None

    model_engine = ScriptedEngine([_result_for(_MODEL_SELECTION_FIXTURE)])
    model_agent = ModelSelectionAgent(engine=model_engine)
    model_result = model_agent.run(
        task_description="构建客服问答机器人",
        scenario_summary=json.dumps(scenario_result.structured_output, ensure_ascii=False),
        dataset_summary=json.dumps(dataset_result.structured_output, ensure_ascii=False),
    )
    assert model_result.structured_output is not None

    plan_engine = ScriptedEngine([_result_for(_TRAINING_PLAN_FIXTURE)])
    plan_agent = TrainingPlanGeneratorAgent(engine=plan_engine)
    plan_result = plan_agent.run(
        task_description="构建客服问答机器人",
        task_type="llm-sft",
        scenario_output=json.dumps(scenario_result.structured_output, ensure_ascii=False),
        dataset_output=json.dumps(dataset_result.structured_output, ensure_ascii=False),
        model_selection_output=json.dumps(model_result.structured_output, ensure_ascii=False),
    )
    assert plan_result.structured_output is not None
    plan_markdown = plan_result.structured_output["markdown"]
    assert "Pipeline Stages" in plan_markdown

    training_plan_path = tmp_path / "training_plan.md"
    write_markdown(training_plan_path, plan_markdown)
    assert training_plan_path.read_text(encoding="utf-8") == plan_markdown

    review_engine = ScriptedEngine([_result_for(_PLAN_REVIEW_APPROVED_FIXTURE)])
    review_agent = PlanReviewerAgent(engine=review_engine)
    review_result = review_agent.run(training_plan_markdown=plan_markdown)
    assert review_result.structured_output is not None
    assert review_result.structured_output["approved"] is True
    assert determine_rollback_target(review_result.structured_output["issues"]) is None

    review_log_path = tmp_path / "plan_review_log.md"
    append_markdown_log(
        review_log_path,
        f"## 评审记录\n{review_result.structured_output['summary']}",
    )
    assert "通过评审" in review_log_path.read_text(encoding="utf-8")

    report_engine = ScriptedEngine([_result_for(_ANALYSIS_REPORT_FIXTURE)])
    report_agent = ReportWriterAgent(engine=report_engine)
    report_result = report_agent.run(
        scenario_output=json.dumps(scenario_result.structured_output, ensure_ascii=False),
        dataset_output=json.dumps(dataset_result.structured_output, ensure_ascii=False),
        model_selection_output=json.dumps(model_result.structured_output, ensure_ascii=False),
        training_plan_output=json.dumps(plan_result.structured_output, ensure_ascii=False),
    )
    assert report_result.structured_output is not None
    report_markdown = report_result.structured_output["markdown"]

    report_path = tmp_path / "analysis_report.md"
    write_markdown(report_path, report_markdown)
    assert "分析报告" in report_path.read_text(encoding="utf-8")

    for agent in (scenario_agent, dataset_agent, model_agent, plan_agent, review_agent, report_agent):
        agent.stop()


def test_plan_reviewer_rejection_rolls_back_to_deepest_root_cause() -> None:
    rejected = PlanReviewOutput(
        approved=False,
        issues=[
            {"category": "计划参数", "description": "Batch Size过大"},
            {"category": "数据理解", "description": "类别分布估算有误"},
        ],
        summary="存在数据理解与计划参数问题",
    )
    target = determine_rollback_target([issue.model_dump() for issue in rejected.issues])
    assert target == "dataset_analysis"


def test_plan_reviewer_rejection_single_category_rolls_back_correctly() -> None:
    issues = [{"category": "选型判断", "description": "模型选型资源估算不足"}]
    assert determine_rollback_target(issues) == "model_selection"

    issues_param_only = [{"category": "计划参数", "description": "学习率设置不合理"}]
    assert determine_rollback_target(issues_param_only) == "training_plan"


def test_plan_review_log_appends_multiple_sections_in_order(tmp_path: Path) -> None:
    log_path = tmp_path / "plan_review_log.md"
    append_markdown_log(log_path, "## 第1次评审\n拒绝：Batch Size过大")
    append_markdown_log(log_path, "## 第2次评审\n通过")

    content = log_path.read_text(encoding="utf-8")
    assert content.index("第1次评审") < content.index("第2次评审")
