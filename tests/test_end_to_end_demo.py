"""T8端到端回归（全部走模拟）。

用一个小型示例任务驱动`StateMachine`跑通完整闭环，验证`runs/<run_id>/`下
产出的文件契约齐全、`knowledge_base/`正确更新；另外用
`fake_train_script.py`的"崩溃"模式验证硬崩溃信号能被Evaluator正确判FAIL并
在耗尽阶段重试次数后使整条流水线终止为FAILED。

不在本文件范围内：真实调用`codex`/`claude`/`opencode`二进制、真实调用
`llamafactory-cli`/`trl`/`yolo`等训练框架——按计划留给下一轮bug修复阶段处理。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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


def _build_full_pipeline_state_machine(tmp_path: Path) -> StateMachine:
    scenario = ScenarioAnalysisOutput(
        task_type="llm-sft",
        industry="客服问答",
        difficulty="medium",
        latency_constraint="无",
        priority_suggestion="高",
        risks=[],
        citations=["https://example.com"],
    )
    dataset = DatasetAnalysisOutput(
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
    model = ModelSelectionOutput(
        recommended_model="Qwen2.5-7B-Instruct",
        alternative_models=[],
        rationale="效果成本平衡",
        gpu_requirement="4x A100-40GB",
        estimated_metric="满意度>=85%",
        estimated_training_duration="4h",
        data_format_requirements=["ShareGPT多轮对话"],
        citations=["https://example.com"],
    )
    plan = TrainingPlanOutput(
        markdown="# 客服问答 - 训练计划\n\n## Pipeline Stages\n| SFT | ... |\n",
        resource_plan={"GPU": "4x A100-40GB"},
        pipeline_stages=[
            PipelineStage(
                name="SFT",
                start_from="基础模型",
                goal="指令遵循能力对齐",
                engine="llamafactory",
                key_hyperparams="lr=2e-5, epoch=3",
                estimated_duration="4h",
            )
        ],
        data_format=DataFormatSpec(target_format="ShareGPT", rationale="多轮对话", field_mapping=[]),
        decision_references=["https://example.com"],
    )
    review = PlanReviewOutput(approved=True, issues=[], summary="计划合理")
    report = AnalysisReportOutput(markdown="# 分析报告\n第一章...第四章...")
    stage_config = StageConfigOutput(yaml_content="model_name_or_path: Qwen2.5-7B-Instruct\n")
    monitor_report = MonitorReportOutput(
        risk_level="Normal",
        gpu_observation="正常",
        loss_trend="下降",
        overfitting_signal="无",
        validation_accuracy="符合预期",
        recommendation="继续",
    )
    evaluator_pass = EvaluatorOutput(
        verdict="PASS", gap_analysis="达标", improvement_suggestions=[], needs_replanning=False
    )
    knowledge_card = KnowledgeCardOutput(
        task_summary="客服问答SFT",
        dataset_stats_summary="100条样本",
        model_and_hyperparams_summary="Qwen2.5-7B-Instruct, lr=2e-5",
        final_metrics_summary="满意度87%",
        lessons_learned=["长尾问题需数据增强"],
        reused_pipeline_stages=plan.pipeline_stages,
        task_types=["llm-sft"],
    )

    context = RunContext(
        run_id="demo_run",
        task_description="构建客服问答机器人",
        dataset_records=[{"instruction": "你好", "input": "", "output": "你好，有什么可以帮您？"}],
        indicators="满意度>=85%",
    )
    config = StateMachineConfig(
        poll_interval_seconds=0,
        max_polls_per_stage=1,
        logger_type="local",
        runs_root=tmp_path / "runs",
        logs_root=tmp_path / "logs",
        knowledge_base_root=tmp_path / "knowledge_base",
        run_script_resolver=_fake_resolver,
    )

    return StateMachine(
        context=context,
        config=config,
        scenario_agent=ScenarioAnalysisAgent(engine=ScriptedEngine([_r(scenario)])),
        dataset_agent=DatasetAnalysisAgent(engine=ScriptedEngine([_r(dataset)])),
        model_agent=ModelSelectionAgent(engine=ScriptedEngine([_r(model)])),
        plan_agent=TrainingPlanGeneratorAgent(engine=ScriptedEngine([_r(plan)])),
        review_agent=PlanReviewerAgent(engine=ScriptedEngine([_r(review)])),
        report_agent=ReportWriterAgent(engine=ScriptedEngine([_r(report)])),
        trainer_agent=TrainerAgent(engine=ScriptedEngine([_r(stage_config)])),
        monitor_agent=MonitorAgent(engine=ScriptedEngine([_r(monitor_report)])),
        evaluator_agent=EvaluatorAgent(engine=ScriptedEngine([_r(evaluator_pass)])),
        knowledge_agent=KnowledgeUpdateAgent(engine=ScriptedEngine([_r(knowledge_card)])),
    )


def test_end_to_end_demo_produces_all_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_TRAIN_MODE", "normal")
    sm = _build_full_pipeline_state_machine(tmp_path)

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.DONE

    run_dir = tmp_path / "runs" / "demo_run"
    expected_artifacts = [
        run_dir / "training_plan.md",
        run_dir / "training_plan.json",
        run_dir / "plan_review_log.md",
        run_dir / "analysis_report.md",
        run_dir / "data",
        run_dir / "stage_1_sft" / "config.yaml",
        run_dir / "metrics.csv",
        run_dir / "monitor_reports" / "report_1.md",
        run_dir / "improve_guide.md",
    ]
    for path in expected_artifacts:
        assert path.exists(), f"missing artifact: {path}"

    train_log = tmp_path / "logs" / "demo_run" / "local" / "train.log"
    assert train_log.exists()
    assert "loss" in train_log.read_text(encoding="utf-8")

    kb_root = tmp_path / "knowledge_base"
    assert (kb_root / "cards" / "card_demo_run.json").exists()
    index = json.loads((kb_root / "index.json").read_text(encoding="utf-8"))
    assert any(e["card_id"] == "card_demo_run" for e in index["entries"])


def test_end_to_end_demo_crash_mode_exhausts_retries_and_fails(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_TRAIN_MODE", "crash")
    sm = _build_full_pipeline_state_machine(tmp_path)
    sm.config.max_stage_retries = 1
    # 崩溃路径由退出码硬信号短路判定，不调用LLM；训练配置生成仍需调用一次LLM
    # 每次重试都要生成一次config，因此trainer engine需要2个脚本槽位。
    sm.trainer_agent.engine.script.append(sm.trainer_agent.engine.script[0])
    # monitor每次stage尝试轮询1次，2次尝试共需2个脚本槽位。
    sm.monitor_agent.engine.script.append(sm.monitor_agent.engine.script[0])
    # evaluate()在崩溃时短路，不消费evaluator脚本槽位；knowledge_update不会被调用。
    sm.evaluator_agent.engine.script.clear()
    sm.knowledge_agent.engine.script.clear()

    final_state = sm.run()
    sm.stop_all_agents()

    assert final_state == PipelineState.FAILED

    run_dir = tmp_path / "runs" / "demo_run"
    improve_guide = (run_dir / "improve_guide.md").read_text(encoding="utf-8")
    assert improve_guide.count("FAIL") == 2
    assert "exit_code=137" in improve_guide or "崩溃" in improve_guide
