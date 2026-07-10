"""三阶段+核心循环状态机：`orchestrator/run_pipeline.py`的CLI入口通过本模块
从用户输入驱动到闭环结束。

状态流转：
    PLANNING -> PLAN_REVIEW -> (拒绝，回退P4/P3/P2重新规划，最多
    `max_plan_review_retries`次) | (通过 -> REPORT -> TRAINING)
    TRAINING 内部逐阶段循环 MONITORING -> EVALUATING：
        PASS            -> 进入下一阶段，全部阶段PASS后跳出训练循环
        FAIL(超参级)     -> 回退Trainer重试当前阶段（限`max_stage_retries`次）
        FAIL(需重新规划)  -> 回退整个PLANNING阶段重新规划（限
                            `max_replanning_attempts`次），成功后重新跑一遍
                            TRAINING循环
    全部阶段PASS -> KNOWLEDGE_UPDATE -> DONE
    任一环节超过重试上限 -> FAILED（需要人工介入）

`poll_interval_seconds`/`max_polls_per_stage`两个配置项共同控制Monitor轮询
节奏：生产模式下按`configs/agents.yaml`的`interval_minutes`真实sleep，
`max_polls_per_stage`给一个较大的软上限；测试/demo模式下把
`poll_interval_seconds`设为0、`max_polls_per_stage`设为一个小的确定值，
使得每个stage每次尝试恰好产生可预期数量的Monitor调用，避免依赖真实wall-clock
的不确定性。
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agents.execution.evaluator_agent import EvaluatorAgent
from agents.execution.monitor_agent import MonitorAgent
from agents.execution.schemas import EvaluatorOutput
from agents.execution.trainer_agent import TrainerAgent, stage_dir
from agents.knowledge.knowledge_update_agent import KnowledgeUpdateAgent
from agents.planning.dataset_analysis_agent import DatasetAnalysisAgent
from agents.planning.io_utils import append_markdown_log, write_markdown
from agents.planning.model_selection_agent import ModelSelectionAgent
from agents.planning.plan_reviewer_agent import PlanReviewerAgent, determine_rollback_target
from agents.planning.report_writer_agent import ReportWriterAgent
from agents.planning.scenario_analysis_agent import ScenarioAnalysisAgent
from agents.planning.schemas import PlanReviewOutput, TrainingPlanOutput
from agents.planning.training_plan_generator import TrainingPlanGeneratorAgent


class PipelineState(str, Enum):
    PLANNING = "PLANNING"
    PLAN_REVIEW = "PLAN_REVIEW"
    REPORT = "REPORT"
    TRAINING = "TRAINING"
    MONITORING = "MONITORING"
    EVALUATING = "EVALUATING"
    KNOWLEDGE_UPDATE = "KNOWLEDGE_UPDATE"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class RunContext:
    """一次完整训练闭环运行所需的用户输入。"""

    run_id: str
    task_description: str
    dataset_path: str = "未提供，请基于描述估算"
    dataset_sample: str = ""
    dataset_records: list[dict[str, Any]] = field(default_factory=list)
    indicators: str = "无特殊要求"
    resource_constraints: str = "无特殊约束"
    available_resources: str = "8x NVIDIA A100-SXM4-40GB"


@dataclass
class StateMachineConfig:
    max_plan_review_retries: int = 3
    max_stage_retries: int = 3
    max_replanning_attempts: int = 2
    poll_interval_seconds: float = 300.0
    max_polls_per_stage: int = 100000
    logger_type: str = "local"
    runs_root: Path = Path("runs")
    logs_root: Path = Path("logs")
    knowledge_base_root: Path = Path("knowledge_base")
    run_script_resolver: Callable[[str], Sequence[str]] | None = None


class StateMachine:
    def __init__(
        self,
        context: RunContext,
        config: StateMachineConfig | None = None,
        *,
        scenario_agent: ScenarioAnalysisAgent | None = None,
        dataset_agent: DatasetAnalysisAgent | None = None,
        model_agent: ModelSelectionAgent | None = None,
        plan_agent: TrainingPlanGeneratorAgent | None = None,
        review_agent: PlanReviewerAgent | None = None,
        report_agent: ReportWriterAgent | None = None,
        trainer_agent: TrainerAgent | None = None,
        monitor_agent: MonitorAgent | None = None,
        evaluator_agent: EvaluatorAgent | None = None,
        knowledge_agent: KnowledgeUpdateAgent | None = None,
    ) -> None:
        self.ctx = context
        self.config = config or StateMachineConfig()
        self.state = PipelineState.PLANNING
        self.transitions: list[str] = []

        agent_log_dir = self.config.logs_root / self.ctx.run_id / "agents"

        self.scenario_agent = scenario_agent or ScenarioAnalysisAgent(log_dir=agent_log_dir)
        self.dataset_agent = dataset_agent or DatasetAnalysisAgent(log_dir=agent_log_dir)
        self.model_agent = model_agent or ModelSelectionAgent(log_dir=agent_log_dir)
        self.plan_agent = plan_agent or TrainingPlanGeneratorAgent(log_dir=agent_log_dir)
        self.review_agent = review_agent or PlanReviewerAgent(log_dir=agent_log_dir)
        self.report_agent = report_agent or ReportWriterAgent(log_dir=agent_log_dir)
        self.trainer_agent = trainer_agent or TrainerAgent(log_dir=agent_log_dir)
        self.monitor_agent = monitor_agent or MonitorAgent(log_dir=agent_log_dir)
        self.evaluator_agent = evaluator_agent or EvaluatorAgent(log_dir=agent_log_dir)
        self.knowledge_agent = knowledge_agent or KnowledgeUpdateAgent(log_dir=agent_log_dir)

        self._scenario_output: dict[str, Any] | None = None
        self._dataset_output: dict[str, Any] | None = None
        self._model_output: dict[str, Any] | None = None
        self.training_plan: TrainingPlanOutput | None = None
        self.knowledge_card_id: str | None = None

    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self.config.runs_root / self.ctx.run_id

    def _transition(self, new_state: PipelineState) -> None:
        self.transitions.append(f"{self.state.value}->{new_state.value}")
        self.state = new_state

    def run(self) -> PipelineState:
        """驱动整个闭环直到DONE或FAILED，返回最终状态。"""
        self._run_planning_loop()
        if self.state == PipelineState.FAILED:
            return self.state

        self._run_report()
        self._run_training_loop()
        if self.state == PipelineState.FAILED:
            return self.state

        self._run_knowledge_update()
        self._transition(PipelineState.DONE)
        return self.state

    # ------------------------------------------------------------------
    # Planning阶段
    # ------------------------------------------------------------------
    def _call_scenario(self) -> None:
        result = self.scenario_agent.run(
            task_description=self.ctx.task_description,
            indicators=self.ctx.indicators,
            resource_constraints=self.ctx.resource_constraints,
        )
        assert result.structured_output is not None
        self._scenario_output = result.structured_output

    def _call_dataset(self) -> None:
        assert self._scenario_output is not None
        result = self.dataset_agent.run(
            task_description=self.ctx.task_description,
            dataset_path=self.ctx.dataset_path,
            dataset_sample=self.ctx.dataset_sample,
            scenario_summary=json.dumps(self._scenario_output, ensure_ascii=False),
        )
        assert result.structured_output is not None
        self._dataset_output = result.structured_output

    def _call_model_selection(self) -> None:
        assert self._scenario_output is not None and self._dataset_output is not None
        result = self.model_agent.run(
            task_description=self.ctx.task_description,
            scenario_summary=json.dumps(self._scenario_output, ensure_ascii=False),
            dataset_summary=json.dumps(self._dataset_output, ensure_ascii=False),
            resource_constraints=self.ctx.resource_constraints,
        )
        assert result.structured_output is not None
        self._model_output = result.structured_output

    def _call_training_plan(self, review_feedback: str | None = None) -> None:
        assert (
            self._scenario_output is not None
            and self._dataset_output is not None
            and self._model_output is not None
        )
        result = self.plan_agent.run(
            task_description=self.ctx.task_description,
            task_type=self._scenario_output.get("task_type", ""),
            indicators=self.ctx.indicators,
            resource_constraints=self.ctx.resource_constraints,
            scenario_output=json.dumps(self._scenario_output, ensure_ascii=False),
            dataset_output=json.dumps(self._dataset_output, ensure_ascii=False),
            model_selection_output=json.dumps(self._model_output, ensure_ascii=False),
            review_feedback=review_feedback or "",
        )
        assert result.structured_output is not None
        plan = TrainingPlanOutput(**result.structured_output)
        self.training_plan = plan

        write_markdown(self.run_dir / "training_plan.md", plan.markdown)
        (self.run_dir / "training_plan.json").parent.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "training_plan.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )

    def _call_plan_review(self) -> PlanReviewOutput:
        assert self.training_plan is not None
        result = self.review_agent.run(
            training_plan_markdown=self.training_plan.markdown,
            available_resources=self.ctx.available_resources,
            indicators=self.ctx.indicators,
        )
        assert result.structured_output is not None
        return PlanReviewOutput(**result.structured_output)

    def _run_planning_loop(self) -> None:
        self._transition(PipelineState.PLANNING)
        review_log_path = self.run_dir / "plan_review_log.md"

        self._call_scenario()
        self._call_dataset()
        self._call_model_selection()
        self._call_training_plan()

        attempt = 0
        while True:
            attempt += 1
            self._transition(PipelineState.PLAN_REVIEW)
            review = self._call_plan_review()

            if review.approved:
                append_markdown_log(
                    review_log_path, f"## 第{attempt}次评审\n通过。{review.summary}"
                )
                return

            issues = [issue.model_dump() for issue in review.issues]
            append_markdown_log(
                review_log_path,
                f"## 第{attempt}次评审\n拒绝。{review.summary}\n问题清单: {issues}",
            )

            if attempt > self.config.max_plan_review_retries:
                append_markdown_log(
                    review_log_path, "## 终止\n已达最大评审重试次数，需人工介入。"
                )
                self._transition(PipelineState.FAILED)
                return

            review_feedback = f"评审结论: {review.summary}\n问题清单: {issues}"
            rollback_target = determine_rollback_target(issues)
            self._transition(PipelineState.PLANNING)
            if rollback_target == "dataset_analysis":
                self._call_dataset()
                self._call_model_selection()
                self._call_training_plan(review_feedback=review_feedback)
            elif rollback_target == "model_selection":
                self._call_model_selection()
                self._call_training_plan(review_feedback=review_feedback)
            else:
                self._call_training_plan(review_feedback=review_feedback)

    def _run_report(self) -> None:
        assert self.training_plan is not None
        self._transition(PipelineState.REPORT)
        result = self.report_agent.run(
            scenario_output=json.dumps(self._scenario_output, ensure_ascii=False),
            dataset_output=json.dumps(self._dataset_output, ensure_ascii=False),
            model_selection_output=json.dumps(self._model_output, ensure_ascii=False),
            training_plan_output=self.training_plan.model_dump_json(),
        )
        assert result.structured_output is not None
        write_markdown(self.run_dir / "analysis_report.md", result.structured_output["markdown"])

    # ------------------------------------------------------------------
    # Execution阶段
    # ------------------------------------------------------------------
    def _run_training_loop(self) -> None:
        assert self.training_plan is not None
        self._transition(PipelineState.TRAINING)

        replanning_attempts = 0
        while True:
            data_path = self.trainer_agent.prepare_data(
                self.training_plan.data_format,
                self.ctx.dataset_records,
                self.ctx.run_id,
                runs_root=self.config.runs_root,
            )
            start_from_path = (
                (self._model_output or {}).get("recommended_model") or "base_model"
            )

            needs_replanning = False
            for stage_index, stage in enumerate(self.training_plan.pipeline_stages, start=1):
                outcome = self._run_single_stage(stage, stage_index, start_from_path, data_path)
                if outcome is None:
                    self._transition(PipelineState.FAILED)
                    return
                if outcome.needs_replanning:
                    needs_replanning = True
                    break
                start_from_path = str(
                    stage_dir(self.config.runs_root, self.ctx.run_id, stage_index, stage.name)
                    / "checkpoints"
                )

            self._transition(PipelineState.TRAINING)
            if not needs_replanning:
                return

            replanning_attempts += 1
            if replanning_attempts > self.config.max_replanning_attempts:
                self._transition(PipelineState.FAILED)
                return

            self._run_planning_loop()
            if self.state == PipelineState.FAILED:
                return
            self._run_report()
            self._transition(PipelineState.TRAINING)

    def _run_single_stage(
        self,
        stage: Any,
        stage_index: int,
        start_from_path: str,
        data_path: Path,
    ) -> EvaluatorOutput | None:
        for iteration in range(1, self.config.max_stage_retries + 2):
            config_path = self.trainer_agent.build_stage_config(
                stage=stage,
                resource_plan=self.training_plan.resource_plan if self.training_plan else {},
                start_from_path=start_from_path,
                run_id=self.ctx.run_id,
                stage_index=stage_index,
                dataset_path=str(data_path),
                runs_root=self.config.runs_root,
            )
            proc = self.trainer_agent.launch_stage(
                stage=stage,
                config_path=config_path,
                run_id=self.ctx.run_id,
                stage_index=stage_index,
                logger_type=self.config.logger_type,
                runs_root=self.config.runs_root,
                logs_root=self.config.logs_root,
                run_script_resolver=self.config.run_script_resolver,
            )

            exit_code, crash_detected = self._monitor_until_stage_done(proc)

            self._transition(PipelineState.EVALUATING)
            result = self.evaluator_agent.evaluate(
                run_id=self.ctx.run_id,
                iteration=iteration,
                indicators=self.ctx.indicators,
                hyperparams_snapshot={"stage": stage.name, "key_hyperparams": stage.key_hyperparams},
                process_exit_code=exit_code,
                crash_detected=crash_detected,
                runs_root=self.config.runs_root,
            )

            if result.verdict == "PASS":
                self._transition(PipelineState.TRAINING)
                return result
            if result.needs_replanning:
                return result
            if iteration > self.config.max_stage_retries:
                return None
            self._transition(PipelineState.TRAINING)
        return None

    def _monitor_until_stage_done(self, proc: subprocess.Popen) -> tuple[int | None, bool]:
        self._transition(PipelineState.MONITORING)
        log_dir = self.config.logs_root / self.ctx.run_id / self.config.logger_type

        crash_detected = False
        polls = 0
        while polls < self.config.max_polls_per_stage:
            if proc.poll() is not None:
                break
            if self.config.poll_interval_seconds > 0:
                time.sleep(self.config.poll_interval_seconds)

            report = self.monitor_agent.poll_once(
                run_id=self.ctx.run_id,
                log_dir=log_dir,
                logger_type=self.config.logger_type,
                indicators=self.ctx.indicators,
                runs_root=self.config.runs_root,
            )
            polls += 1
            if report.crash_detected or report.risk_level == "Critical":
                crash_detected = report.crash_detected
                break
            if proc.poll() is not None:
                break

        if proc.poll() is None:
            try:
                proc.wait(timeout=max(self.config.poll_interval_seconds * 2, 5))
            except subprocess.TimeoutExpired:
                pass

        return proc.poll(), crash_detected

    # ------------------------------------------------------------------
    # Knowledge阶段
    # ------------------------------------------------------------------
    def _run_knowledge_update(self) -> None:
        assert self.training_plan is not None
        self._transition(PipelineState.KNOWLEDGE_UPDATE)

        analysis_report_path = self.run_dir / "analysis_report.md"
        improve_guide_path = self.run_dir / "improve_guide.md"
        monitor_reports_dir = self.run_dir / "monitor_reports"

        monitor_reports_text = ""
        if monitor_reports_dir.exists():
            monitor_reports_text = "\n---\n".join(
                p.read_text(encoding="utf-8") for p in sorted(monitor_reports_dir.glob("report_*.md"))
            )

        self.knowledge_card_id = self.knowledge_agent.run_and_save(
            run_id=self.ctx.run_id,
            training_plan_markdown=self.training_plan.markdown,
            analysis_report_markdown=(
                analysis_report_path.read_text(encoding="utf-8")
                if analysis_report_path.exists()
                else ""
            ),
            improve_guide_markdown=(
                improve_guide_path.read_text(encoding="utf-8") if improve_guide_path.exists() else ""
            ),
            monitor_reports_text=monitor_reports_text,
            knowledge_base_root=self.config.knowledge_base_root,
        )

    # ------------------------------------------------------------------
    def stop_all_agents(self) -> None:
        for agent in (
            self.scenario_agent,
            self.dataset_agent,
            self.model_agent,
            self.plan_agent,
            self.review_agent,
            self.report_agent,
            self.trainer_agent,
            self.monitor_agent,
            self.evaluator_agent,
            self.knowledge_agent,
        ):
            agent.stop()
