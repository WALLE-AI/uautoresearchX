"""orchestrator/state_machine.py集成测试：驱动完整闭环走过三条关键路径，全部
用`ScriptedEngine`+`fake_train_script.py`模拟，不依赖真实CLI/训练框架。

每个Agent持有各自独立的`ScriptedEngine`实例，因此只需保证"该Agent在本场景
下被调用的次数与顺序"与脚本列表一致，不需要考虑跨Agent的全局调用顺序。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
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
from orchestrator.state_machine import (
    PipelineState,
    RunContext,
    StateMachine,
    StateMachineConfig,
    _serializable_config,
)
from orchestrator.run_registry import RunManifest
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


# ----------------------------------------------------------------------
# resume：模拟CLI进程崩溃后从manifest恢复
# ----------------------------------------------------------------------


def test_resume_reattaches_to_still_running_training_pid_and_completes(tmp_path: Path) -> None:
    """模拟"CLI进程在训练子进程仍在跑时被杀"场景：不调用`sm.run()`，直接手工
    构造一份对应"跑到MONITORING阶段一半"的`RunManifest`（`training_pid`指向一个
    真实存活的进程），再用`StateMachine.resume_from_manifest()`重建一个全新的
    `StateMachine`并调用`resume()`，断言：(1) 不会重新调用Trainer生成config
    （TrainerAgent的ScriptedEngine给空脚本列表，若重新调用会因`pop from empty
    list`报错）；(2) 轮询到该pid自然退出后能正确完成评测并推进到DONE。
    """
    run_id = "run_resume_reattach"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / run_id

    # 用真实sleep子进程模拟"仍在跑的训练"，不经过launch_stage的bash包装。
    # pytest进程本身就是它的父进程，若不主动reap会一直是僵尸进程（对`os.kill(pid,
    # 0)`而言僵尸依然"存活"），与真实场景（原进程已死、被init自动reap）不同，
    # 因此起一个后台线程调用`proc.wait()`及时回收，让`is_pid_alive`能正确感知
    # 进程已退出。
    import threading

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    threading.Thread(target=proc.wait, daemon=True).start()
    exit_code_path = run_dir / "exit_code.txt"
    exit_code_path.parent.mkdir(parents=True, exist_ok=True)
    exit_code_path.write_text("0", encoding="utf-8")

    context = _make_context(run_id)
    config = _make_config(tmp_path)
    config.poll_interval_seconds = 0.1
    config.max_polls_per_stage = 50

    plan = _plan()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    manifest = RunManifest(
        run_id=run_id,
        status="RUNNING",
        pipeline_state="MONITORING",
        context=asdict(context),
        config=_serializable_config(config),
        scenario_output=_SCENARIO.model_dump(),
        dataset_output=_DATASET.model_dump(),
        model_output=_MODEL.model_dump(),
        stage_index=1,
        stage_iteration=1,
        training_pid=proc.pid,
        training_exit_code_path=str(exit_code_path),
    )

    sm = StateMachine.resume_from_manifest(
        manifest,
        run_script_resolver=_fake_resolver,
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([])),
        plan_agent=TrainingPlanGeneratorAgent(engine=ScriptedEngine([])),
        review_agent=PlanReviewerAgent(engine=ScriptedEngine([])),
        report_agent=ReportWriterAgent(engine=ScriptedEngine([])),
        # 空脚本：证明resume没有重新调用build_stage_config()。
        trainer_agent=TrainerAgent(engine=ScriptedEngine([])),
        monitor_agent=MonitorAgent(engine=ScriptedEngine([_r(_MONITOR_NORMAL)] * 30)),
        evaluator_agent=EvaluatorAgent(
            engine=ScriptedEngine(
                [_r(EvaluatorOutput(verdict="PASS", gap_analysis="达标", improvement_suggestions=[], needs_replanning=False))]
            )
        ),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([_r(_KNOWLEDGE_CARD)])),
    )

    final_state = sm.resume()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE
    assert sm.manifest.training_pid is None
    improve_guide = (run_dir / "improve_guide.md").read_text(encoding="utf-8")
    assert "PASS" in improve_guide


def test_resume_refuses_terminal_status(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="run_done",
        status="DONE",
        pipeline_state="DONE",
        context=asdict(_make_context("run_done")),
        config=_serializable_config(_make_config(tmp_path)),
    )
    with pytest.raises(ValueError):
        StateMachine.resume_from_manifest(manifest)


# ----------------------------------------------------------------------
# human_gate：--interactive场景下人工决策覆盖LLM原有判定
# ----------------------------------------------------------------------


class _ScriptedHumanGate:
    """按脚本顺序返回决策，记录被调用时收到的LLM原始判定，供断言。"""

    def __init__(
        self,
        review_decisions: list[str] | None = None,
        stage_fail_decisions: list[str] | None = None,
        retry_decisions: list[str] | None = None,
    ) -> None:
        self._review = list(review_decisions or [])
        self._stage_fail = list(stage_fail_decisions or [])
        self._retry = list(retry_decisions or [])
        self.review_calls: list[bool] = []
        self.stage_fail_calls: list[str] = []
        self.retry_calls: list[str] = []

    def review_plan(self, plan: Any, review: PlanReviewOutput) -> str:
        self.review_calls.append(review.approved)
        return self._review.pop(0) if self._review else "accept_llm_verdict"

    def on_stage_fail(self, evaluator_output: EvaluatorOutput) -> str:
        self.stage_fail_calls.append(evaluator_output.verdict)
        return self._stage_fail.pop(0) if self._stage_fail else "accept_llm_verdict"

    def on_max_retries_exceeded(self, context: str) -> str:
        self.retry_calls.append(context)
        return self._retry.pop(0) if self._retry else "abort"


def test_human_gate_force_reject_overrides_llm_approval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_TRAIN_MODE", "normal")
    human_gate = _ScriptedHumanGate(review_decisions=["force_reject"])

    sm = StateMachine(
        context=_make_context("run_force_reject"),
        config=_make_config(tmp_path),
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([_r(_SCENARIO)])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([_r(_DATASET)])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([_r(_MODEL)])),
        plan_agent=TrainingPlanGeneratorAgent(engine=ScriptedEngine([_r(_plan()), _r(_plan())])),
        review_agent=PlanReviewerAgent(
            engine=ScriptedEngine([_r(_APPROVED_REVIEW), _r(_APPROVED_REVIEW)])
        ),
        report_agent=ReportWriterAgent(engine=ScriptedEngine([_r(_ANALYSIS_REPORT)])),
        trainer_agent=TrainerAgent(engine=ScriptedEngine([_r(_STAGE_CONFIG)])),
        monitor_agent=MonitorAgent(engine=ScriptedEngine([_r(_MONITOR_NORMAL)])),
        evaluator_agent=EvaluatorAgent(
            engine=ScriptedEngine(
                [
                    _r(
                        EvaluatorOutput(
                            verdict="PASS", gap_analysis="达标", improvement_suggestions=[], needs_replanning=False
                        )
                    )
                ]
            )
        ),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([_r(_KNOWLEDGE_CARD)])),
        human_gate=human_gate,
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE
    # LLM两次都判定通过（True），但第一次被人工强制拒绝，第二次人工采纳。
    assert human_gate.review_calls == [True, True]
    review_log = (tmp_path / "runs" / "run_force_reject" / "plan_review_log.md").read_text(
        encoding="utf-8"
    )
    assert "第1次评审\n拒绝" in review_log
    assert "第2次评审\n通过" in review_log


def test_human_gate_abort_on_stage_fail_stops_immediately(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_TRAIN_MODE", "diverge")
    human_gate = _ScriptedHumanGate(stage_fail_decisions=["abort"])

    sm = StateMachine(
        context=_make_context("run_human_abort"),
        config=_make_config(tmp_path),
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([_r(_SCENARIO)])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([_r(_DATASET)])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([_r(_MODEL)])),
        plan_agent=TrainingPlanGeneratorAgent(engine=ScriptedEngine([_r(_plan())])),
        review_agent=PlanReviewerAgent(engine=ScriptedEngine([_r(_APPROVED_REVIEW)])),
        report_agent=ReportWriterAgent(engine=ScriptedEngine([_r(_ANALYSIS_REPORT)])),
        trainer_agent=TrainerAgent(engine=ScriptedEngine([_r(_STAGE_CONFIG)])),
        monitor_agent=MonitorAgent(engine=ScriptedEngine([_r(_MONITOR_NORMAL)])),
        evaluator_agent=EvaluatorAgent(
            engine=ScriptedEngine(
                [
                    _r(
                        EvaluatorOutput(
                            verdict="FAIL",
                            gap_analysis="loss发散",
                            improvement_suggestions=["降低学习率"],
                            needs_replanning=False,
                        )
                    )
                ]
            )
        ),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([])),
        human_gate=human_gate,
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.FAILED
    assert human_gate.stage_fail_calls == ["FAIL"]


def test_human_gate_extend_retries_bypasses_stage_retry_limit(tmp_path: Path, monkeypatch) -> None:
    """回归测试：`_run_single_stage`此前用`for iteration in range(...,
    max_stage_retries+2)`固定边界，循环体内`max_stage_retries += 1`并不会真正
    延长循环（range对象在创建时就已固定），这里验证改成`while True`+手动
    自增后，人工选择`extend_retries`确实能多跑一轮并最终PASS。
    """
    monkeypatch.setenv("FAKE_TRAIN_MODE", "normal")
    human_gate = _ScriptedHumanGate(retry_decisions=["extend_retries"])

    config = _make_config(tmp_path)
    config.max_stage_retries = 0

    sm = StateMachine(
        context=_make_context("run_extend_retry"),
        config=config,
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
        evaluator_agent=EvaluatorAgent(
            engine=ScriptedEngine(
                [
                    _r(
                        EvaluatorOutput(
                            verdict="FAIL",
                            gap_analysis="欠拟合",
                            improvement_suggestions=["提高学习率"],
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
        ),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([_r(_KNOWLEDGE_CARD)])),
        human_gate=human_gate,
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE
    assert human_gate.retry_calls, "on_max_retries_exceeded应该被调用过一次"
    assert config.max_stage_retries == 1


# ----------------------------------------------------------------------
# on_event / on_transition：TUI等展示层依赖的回调接线
# ----------------------------------------------------------------------


def test_on_event_and_on_transition_callbacks_fire_during_full_run(
    tmp_path: Path, monkeypatch
) -> None:
    """验证`StateMachine(on_event=..., on_transition=...)`真的接到了全部9处
    `*_agent.run()`/`poll_once`/`build_stage_config`/`run_and_save`调用（`
    ScriptedEngine`现在会在收到on_event时回放text_delta+done），以及每次
    `_transition()`都触发了`on_transition`。这是`orchestrator/tui.py`得以
    工作的前提。
    """
    monkeypatch.setenv("FAKE_TRAIN_MODE", "normal")

    event_types: list[str] = []
    transitions: list[tuple[str, str]] = []

    scenario_engine = ScriptedEngine([_r(_SCENARIO)])
    dataset_engine = ScriptedEngine([_r(_DATASET)])
    model_engine = ScriptedEngine([_r(_MODEL)])
    plan_engine = ScriptedEngine([_r(_plan())])
    review_engine = ScriptedEngine([_r(_APPROVED_REVIEW)])
    report_engine = ScriptedEngine([_r(_ANALYSIS_REPORT)])
    trainer_engine = ScriptedEngine([_r(_STAGE_CONFIG)])
    monitor_engine = ScriptedEngine([_r(_MONITOR_NORMAL)])
    evaluator_engine = ScriptedEngine(
        [_r(EvaluatorOutput(verdict="PASS", gap_analysis="达标", improvement_suggestions=[], needs_replanning=False))]
    )
    knowledge_engine = ScriptedEngine([_r(_KNOWLEDGE_CARD)])

    sm = StateMachine(
        context=_make_context("run_callbacks"),
        config=_make_config(tmp_path),
        scenario_agent=ScenarioAnalysisAgent(engine=scenario_engine),
        dataset_agent=DatasetAnalysisAgent(engine=dataset_engine),
        model_agent=ModelSelectionAgent(engine=model_engine),
        plan_agent=TrainingPlanGeneratorAgent(engine=plan_engine),
        review_agent=PlanReviewerAgent(engine=review_engine),
        report_agent=ReportWriterAgent(engine=report_engine),
        trainer_agent=TrainerAgent(engine=trainer_engine),
        monitor_agent=MonitorAgent(engine=monitor_engine),
        evaluator_agent=EvaluatorAgent(engine=evaluator_engine),
        knowledge_agent=KnowledgeUpdateAgent(engine=knowledge_engine),
        on_event=lambda event: event_types.append(event.type),
        on_transition=lambda old, new: transitions.append((old, new)),
    )

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE
    # 9个LLM调用点每次都应回放text_delta+done，即至少18个事件。
    assert event_types.count("text_delta") >= 9
    assert event_types.count("done") >= 9

    # 每个engine都应实际收到了同一个non-None回调（而不是被静默丢弃）。
    for engine in (
        scenario_engine,
        dataset_engine,
        model_engine,
        plan_engine,
        review_engine,
        report_engine,
        trainer_engine,
        monitor_engine,
        evaluator_engine,
        knowledge_engine,
    ):
        assert engine.on_event_calls and all(cb is not None for cb in engine.on_event_calls)

    # on_transition应与sm.transitions记录的转移一一对应。
    assert [f"{old}->{new}" for old, new in transitions] == sm.transitions
