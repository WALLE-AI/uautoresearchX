"""orchestrator/state_machine.py集成测试：驱动完整闭环走过三条关键路径，全部
用`ScriptedEngine`+`fake_train_script.py`模拟，不依赖真实CLI/训练框架。

每个Agent持有各自独立的`ScriptedEngine`实例，因此只需保证"该Agent在本场景
下被调用的次数与顺序"与脚本列表一致，不需要考虑跨Agent的全局调用顺序。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from agents.engines.base_engine import AgentResult
from agents.execution.evaluator_agent import EvaluatorAgent
from agents.execution.monitor_agent import MonitorAgent
from agents.execution.schemas import EvaluatorOutput, MonitorReportOutput, StageConfigOutput
from agents.execution.trainer_agent import TrainerAgent
from agents.knowledge.knowledge_update_agent import KnowledgeUpdateAgent
from agents.knowledge.schemas import KnowledgeCardOutput
from agents.planning.dataset_analysis_agent import DatasetAnalysisAgent
from agents.planning.model_selection_agent import ModelSelectionAgent
from agents.planning.plan_reviewer_agent import PlanReviewerAgent
from agents.planning.report_writer_agent import ReportWriterAgent
from agents.planning.scenario_analysis_agent import ScenarioAnalysisAgent
from agents.planning.schemas import (
    AnalysisReportOutput,
    DataFormatSpec,
    DatasetAnalysisOutput,
    ModelSelectionOutput,
    PipelineStage,
    PlanReviewOutput,
    ReviewIssue,
    ScenarioAnalysisOutput,
    TrainingPlanOutput,
)
from agents.planning.training_plan_generator import TrainingPlanGeneratorAgent
from orchestrator.state_machine import PipelineState, RunContext, StateMachine, StateMachineConfig
from tests.fakes.scripted_engine import ScriptedEngine

_FAKE_TRAIN_SCRIPT = Path(__file__).parent / "fakes" / "fake_train_script.py"


def _r(model_instance: Any) -> AgentResult:
    dumped = model_instance.model_dump()
    return AgentResult(text=json.dumps(dumped, ensure_ascii=False), structured_output=dumped)


def _fake_resolver(engine: str) -> list[str]:
    return [sys.executable, str(_FAKE_TRAIN_SCRIPT)]


_SCENARIO = ScenarioAnalysisOutput(
    task_type="llm-sft",
    industry="客服问答",
    difficulty="medium",
    latency_constraint="无",
    priority_suggestion="高",
    risks=[],
    citations=["https://example.com"],
)
_DATASET = DatasetAnalysisOutput(
    num_samples=100,
    spec="短对话",
    num_classes=None,
    class_distribution=None,
    quality_score=8.0,
    augmentation_suggestions=[],
    candidate_formats=["ShareGPT"],
    field_mapping_hints=[],
    confidence="medium",
    citations=["https://example.com"],
)
_MODEL = ModelSelectionOutput(
    recommended_model="Qwen2.5-7B-Instruct",
    alternative_models=[],
    rationale="效果成本平衡",
    gpu_requirement="4x A100-40GB",
    estimated_metric="满意度>=85%",
    estimated_training_duration="4h",
    data_format_requirements=["ShareGPT多轮对话"],
    citations=["https://example.com"],
)


def _plan(engine: str = "llamafactory") -> TrainingPlanOutput:
    return TrainingPlanOutput(
        markdown="# 客服问答 - 训练计划\n\n## Pipeline Stages\n| 阶段 | ... |\n",
        resource_plan={"GPU": "4x A100-40GB", "Batch Size": "16"},
        pipeline_stages=[
            PipelineStage(
                name="SFT",
                start_from="基础模型",
                goal="指令遵循能力对齐",
                engine=engine,
                key_hyperparams="lr=2e-5, epoch=3",
                estimated_duration="4h",
            )
        ],
        data_format=DataFormatSpec(target_format="ShareGPT", rationale="多轮对话", field_mapping=[]),
        decision_references=["https://example.com"],
    )


_APPROVED_REVIEW = PlanReviewOutput(approved=True, issues=[], summary="计划合理")
_REJECTED_REVIEW = PlanReviewOutput(
    approved=False,
    issues=[ReviewIssue(category="计划参数", description="Batch Size偏大")],
    summary="超参不合理",
)
_ANALYSIS_REPORT = AnalysisReportOutput(markdown="# 分析报告\n...")
_STAGE_CONFIG = StageConfigOutput(yaml_content="model_name_or_path: Qwen2.5-7B-Instruct\n")
_MONITOR_NORMAL = MonitorReportOutput(
    risk_level="Normal",
    gpu_observation="正常",
    loss_trend="下降",
    overfitting_signal="无",
    validation_accuracy="符合预期",
    recommendation="继续",
)
_KNOWLEDGE_CARD = KnowledgeCardOutput(
    task_summary="客服问答SFT",
    dataset_stats_summary="100条样本",
    model_and_hyperparams_summary="Qwen2.5-7B-Instruct, lr=2e-5",
    final_metrics_summary="满意度87%",
    lessons_learned=[],
    reused_pipeline_stages=[],
    task_types=["llm-sft"],
)


def _make_context(run_id: str) -> RunContext:
    return RunContext(
        run_id=run_id,
        task_description="构建客服问答机器人",
        dataset_records=[{"instruction": "你好", "input": "", "output": "你好，有什么可以帮您？"}],
        indicators="满意度>=85%",
    )


def _make_config(tmp_path: Path) -> StateMachineConfig:
    return StateMachineConfig(
        poll_interval_seconds=0,
        max_polls_per_stage=1,
        logger_type="local",
        runs_root=tmp_path / "runs",
        logs_root=tmp_path / "logs",
        knowledge_base_root=tmp_path / "knowledge_base",
        run_script_resolver=_fake_resolver,
    )


def test_plan_rejected_once_then_approved_and_full_pipeline_passes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_TRAIN_MODE", "normal")

    sm = StateMachine(
        context=_make_context("run_reject_then_approve"),
        config=_make_config(tmp_path),
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([_r(_SCENARIO)])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([_r(_DATASET)])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([_r(_MODEL)])),
        plan_agent=TrainingPlanGeneratorAgent(
            engine=ScriptedEngine([_r(_plan()), _r(_plan())])
        ),
        review_agent=PlanReviewerAgent(
            engine=ScriptedEngine([_r(_REJECTED_REVIEW), _r(_APPROVED_REVIEW)])
        ),
        report_agent=ReportWriterAgent(engine=ScriptedEngine([_r(_ANALYSIS_REPORT)])),
        trainer_agent=TrainerAgent(engine=ScriptedEngine([_r(_STAGE_CONFIG)])),
        monitor_agent=MonitorAgent(engine=ScriptedEngine([_r(_MONITOR_NORMAL)])),
        evaluator_agent=EvaluatorAgent(
            engine=ScriptedEngine(
                [_r(EvaluatorOutput(verdict="PASS", gap_analysis="达标", improvement_suggestions=[], needs_replanning=False))]
            )
        ),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([_r(_KNOWLEDGE_CARD)])),
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE
    run_dir = tmp_path / "runs" / "run_reject_then_approve"
    assert (run_dir / "training_plan.md").exists()
    assert (run_dir / "plan_review_log.md").exists()
    review_log = (run_dir / "plan_review_log.md").read_text(encoding="utf-8")
    assert "第1次评审" in review_log and "第2次评审" in review_log
    assert (run_dir / "analysis_report.md").exists()
    assert (run_dir / "monitor_reports" / "report_1.md").exists()
    assert (run_dir / "improve_guide.md").exists()
    assert sm.knowledge_card_id == "card_run_reject_then_approve"
    kb_index = json.loads((tmp_path / "knowledge_base" / "index.json").read_text(encoding="utf-8"))
    assert kb_index["entries"][0]["card_id"] == "card_run_reject_then_approve"


def test_stage_fail_hyperparam_retry_then_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_TRAIN_MODE", "diverge")

    fail_then_pass_evaluator = ScriptedEngine(
        [
            _r(
                EvaluatorOutput(
                    verdict="FAIL",
                    gap_analysis="loss发散",
                    improvement_suggestions=["降低学习率"],
                    needs_replanning=False,
                )
            ),
            _r(
                EvaluatorOutput(
                    verdict="PASS", gap_analysis="达标", improvement_suggestions=[], needs_replanning=False
                )
            ),
        ]
    )

    class _ModeAwareEvaluatorAgent(EvaluatorAgent):
        def evaluate(self, *args: Any, **kwargs: Any) -> EvaluatorOutput:
            if kwargs.get("iteration") == 2:
                os.environ["FAKE_TRAIN_MODE"] = "normal"
            return super().evaluate(*args, **kwargs)

    sm = StateMachine(
        context=_make_context("run_stage_retry"),
        config=_make_config(tmp_path),
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([_r(_SCENARIO)])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([_r(_DATASET)])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([_r(_MODEL)])),
        plan_agent=TrainingPlanGeneratorAgent(engine=ScriptedEngine([_r(_plan())])),
        review_agent=PlanReviewerAgent(engine=ScriptedEngine([_r(_APPROVED_REVIEW)])),
        report_agent=ReportWriterAgent(engine=ScriptedEngine([_r(_ANALYSIS_REPORT)])),
        trainer_agent=TrainerAgent(engine=ScriptedEngine([_r(_STAGE_CONFIG), _r(_STAGE_CONFIG)])),
        monitor_agent=MonitorAgent(
            engine=ScriptedEngine([_r(_MONITOR_NORMAL), _r(_MONITOR_NORMAL)])
        ),
        evaluator_agent=_ModeAwareEvaluatorAgent(engine=fail_then_pass_evaluator),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([_r(_KNOWLEDGE_CARD)])),
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE
    run_dir = tmp_path / "runs" / "run_stage_retry"
    improve_guide = (run_dir / "improve_guide.md").read_text(encoding="utf-8")
    assert "迭代 #1" in improve_guide and "迭代 #2" in improve_guide
    assert "FAIL" in improve_guide and "PASS" in improve_guide


def test_stage_fail_triggers_replanning_then_passes(tmp_path: Path, monkeypatch) -> None:
    call_count = {"n": 0}

    class _ModeSwitchingEvaluatorAgent(EvaluatorAgent):
        def evaluate(self, *args: Any, **kwargs: Any) -> EvaluatorOutput:
            call_count["n"] += 1
            if call_count["n"] == 1:
                os.environ["FAKE_TRAIN_MODE"] = "normal"
                return EvaluatorOutput(
                    verdict="FAIL",
                    gap_analysis="模型选型明显资源不足",
                    improvement_suggestions=["更换更小模型"],
                    needs_replanning=True,
                )
            return EvaluatorOutput(
                verdict="PASS", gap_analysis="达标", improvement_suggestions=[], needs_replanning=False
            )

    monkeypatch.setenv("FAKE_TRAIN_MODE", "crash")

    sm = StateMachine(
        context=_make_context("run_replan"),
        config=_make_config(tmp_path),
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([_r(_SCENARIO), _r(_SCENARIO)])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([_r(_DATASET), _r(_DATASET)])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([_r(_MODEL), _r(_MODEL)])),
        plan_agent=TrainingPlanGeneratorAgent(engine=ScriptedEngine([_r(_plan()), _r(_plan())])),
        review_agent=PlanReviewerAgent(
            engine=ScriptedEngine([_r(_APPROVED_REVIEW), _r(_APPROVED_REVIEW)])
        ),
        report_agent=ReportWriterAgent(
            engine=ScriptedEngine([_r(_ANALYSIS_REPORT), _r(_ANALYSIS_REPORT)])
        ),
        trainer_agent=TrainerAgent(engine=ScriptedEngine([_r(_STAGE_CONFIG), _r(_STAGE_CONFIG)])),
        monitor_agent=MonitorAgent(
            engine=ScriptedEngine([_r(_MONITOR_NORMAL), _r(_MONITOR_NORMAL)])
        ),
        evaluator_agent=_ModeSwitchingEvaluatorAgent(engine=ScriptedEngine([])),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([_r(_KNOWLEDGE_CARD)])),
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE
    assert "PLANNING->PLAN_REVIEW" in sm.transitions
    replanning_transitions = [t for t in sm.transitions if t == "TRAINING->PLANNING"]
    assert len(replanning_transitions) == 1


def test_planning_exhausts_retries_and_fails(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.max_plan_review_retries = 1

    always_rejected = ScriptedEngine([_r(_REJECTED_REVIEW)] * 3)

    sm = StateMachine(
        context=_make_context("run_planning_fails"),
        config=config,
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([_r(_SCENARIO)])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([_r(_DATASET)])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([_r(_MODEL)])),
        plan_agent=TrainingPlanGeneratorAgent(
            engine=ScriptedEngine([_r(_plan()), _r(_plan())])
        ),
        review_agent=PlanReviewerAgent(engine=always_rejected),
        report_agent=ReportWriterAgent(engine=ScriptedEngine([])),
        trainer_agent=TrainerAgent(engine=ScriptedEngine([])),
        monitor_agent=MonitorAgent(engine=ScriptedEngine([])),
        evaluator_agent=EvaluatorAgent(engine=ScriptedEngine([])),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([])),
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.FAILED
    review_log = (tmp_path / "runs" / "run_planning_fails" / "plan_review_log.md").read_text(
        encoding="utf-8"
    )
    assert "需人工介入" in review_log
