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

**运行状态持久化与resume**：每次`_transition()`及Planning阶段每个中间产出
落盘后都会同步写一份`runs/<run_id>/manifest.json`（`orchestrator/
run_registry.py`），CLI进程被杀后可用`StateMachine.resume_from_manifest()`
从最后一次成功的状态转换点继续跑，而不是从头重来。resume的粒度是"重跑最后一次
未完成的单个操作"（如某个Planning Agent的LLM调用被打断，resume后会完整重新
调用一次该Agent，而不是恢复半截的文本流）；若被杀时训练子进程仍在跑（`
trainer_agent.launch_stage()`用`start_new_session=True`使其独立于CLI进程组），
resume会重新接管该pid继续监控，不会重启训练。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agents.engines.base_engine import AgentEvent
from agents.execution.evaluator_agent import EvaluatorAgent
from agents.execution.monitor_agent import MonitorAgent
from agents.execution.schemas import EvaluatorOutput
from agents.execution.trainer_agent import TrainerAgent, read_exit_code, stage_dir
from agents.knowledge.knowledge_update_agent import KnowledgeUpdateAgent
from agents.planning.dataset_analysis_agent import DatasetAnalysisAgent
from agents.planning.io_utils import append_markdown_log, write_markdown
from agents.planning.model_selection_agent import ModelSelectionAgent
from agents.planning.plan_reviewer_agent import PlanReviewerAgent, determine_rollback_target
from agents.planning.report_writer_agent import ReportWriterAgent
from agents.planning.scenario_analysis_agent import ScenarioAnalysisAgent
from agents.planning.schemas import PlanReviewOutput, ReviewIssue, TrainingPlanOutput
from agents.planning.training_plan_generator import TrainingPlanGeneratorAgent
from orchestrator.human_gate import AutoHumanGate, HumanGate
from orchestrator.run_registry import RunManifest, RunStatus, is_pid_alive, save_manifest


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


# resume时可以从这些状态继续跑；DONE/FAILED是终态，CANCELLED只存在于manifest.status。
_RESUMABLE_STATES = {
    PipelineState.PLANNING,
    PipelineState.PLAN_REVIEW,
    PipelineState.REPORT,
    PipelineState.TRAINING,
    PipelineState.MONITORING,
    PipelineState.EVALUATING,
    PipelineState.KNOWLEDGE_UPDATE,
}


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


def _serializable_config(config: StateMachineConfig) -> dict[str, Any]:
    """`run_script_resolver`是callable不可JSON序列化，其余字段原样落盘。

    resume时`run_script_resolver`需要由调用方显式重新传入（生产场景本就没有
    这个字段，只有测试/demo用它注入替身脚本；这类场景通常也是在同一进程内直接
    构造`StateMachine`验证，不走跨进程resume路径）。
    """
    return {
        "max_plan_review_retries": config.max_plan_review_retries,
        "max_stage_retries": config.max_stage_retries,
        "max_replanning_attempts": config.max_replanning_attempts,
        "poll_interval_seconds": config.poll_interval_seconds,
        "max_polls_per_stage": config.max_polls_per_stage,
        "logger_type": config.logger_type,
        "runs_root": str(config.runs_root),
        "logs_root": str(config.logs_root),
        "knowledge_base_root": str(config.knowledge_base_root),
    }


def _config_from_dict(
    data: dict[str, Any], *, run_script_resolver: Callable[[str], Sequence[str]] | None = None
) -> StateMachineConfig:
    defaults = StateMachineConfig()
    return StateMachineConfig(
        max_plan_review_retries=data.get("max_plan_review_retries", defaults.max_plan_review_retries),
        max_stage_retries=data.get("max_stage_retries", defaults.max_stage_retries),
        max_replanning_attempts=data.get("max_replanning_attempts", defaults.max_replanning_attempts),
        poll_interval_seconds=data.get("poll_interval_seconds", defaults.poll_interval_seconds),
        max_polls_per_stage=data.get("max_polls_per_stage", defaults.max_polls_per_stage),
        logger_type=data.get("logger_type", defaults.logger_type),
        runs_root=Path(data.get("runs_root", defaults.runs_root)),
        logs_root=Path(data.get("logs_root", defaults.logs_root)),
        knowledge_base_root=Path(data.get("knowledge_base_root", defaults.knowledge_base_root)),
        run_script_resolver=run_script_resolver,
    )


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
        on_event: Callable[[AgentEvent], None] | None = None,
        on_transition: Callable[[str, str], None] | None = None,
        human_gate: HumanGate | None = None,
    ) -> None:
        self.ctx = context
        self.config = config or StateMachineConfig()
        self.state = PipelineState.PLANNING
        self.transitions: list[str] = []
        self._on_event = on_event
        self._on_transition = on_transition
        self._human_gate: HumanGate = human_gate or AutoHumanGate()

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

        self._current_stage_index: int | None = None
        self._current_stage_iteration: int | None = None
        self._replanning_attempts: int = 0

        self.manifest = RunManifest(
            run_id=self.ctx.run_id,
            owner_pid=os.getpid(),
            task_description_preview=self.ctx.task_description[:200],
            context=asdict(self.ctx),
            config=_serializable_config(self.config),
        )
        self._save_manifest()

    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self.config.runs_root / self.ctx.run_id

    def mark_failed(self, error: BaseException) -> None:
        """把manifest标记为FAILED并记录错误信息，供调用方在`run()`/`resume()`
        抛出未被状态机自身捕获的异常（如`AgentRunError`重试耗尽、
        `TimeoutError`）时调用，避免manifest永远停留在`status=RUNNING`——
        否则进程已经崩溃退出，但`status`/`list`会一直显示"仍在运行"，
        `resume`也无法判断这是一次已经废弃的运行。
        """
        self.manifest.last_error = f"{type(error).__name__}: {error}"
        self._save_manifest(status="FAILED")

    def set_callbacks(
        self,
        *,
        on_event: Callable[[AgentEvent], None] | None = None,
        on_transition: Callable[[str, str], None] | None = None,
    ) -> None:
        """在构造完成后补设事件回调（如`resume_from_manifest()`场景下调用方
        需要先拿到`self.config`用于准备TUI的日志tail路径，再回过头接上回调）。
        """
        self._on_event = on_event
        self._on_transition = on_transition

    def _transition(self, new_state: PipelineState) -> None:
        old_state = self.state
        self.transitions.append(f"{self.state.value}->{new_state.value}")
        self.state = new_state
        self._save_manifest()
        if self._on_transition is not None:
            self._on_transition(old_state.value, new_state.value)

    def _save_manifest(self, *, status: RunStatus | None = None) -> None:
        """把当前已知的运行进度落盘到`runs/<run_id>/manifest.json`。

        在每次`_transition()`与Planning阶段每个中间产出之后调用，是`resume`/
        `status`/`list`子命令的唯一数据来源。
        """
        self.manifest.pipeline_state = self.state.value
        if status is not None:
            self.manifest.status = status
        elif self.state == PipelineState.DONE:
            self.manifest.status = "DONE"
        elif self.state == PipelineState.FAILED:
            self.manifest.status = "FAILED"
        self.manifest.scenario_output = self._scenario_output
        self.manifest.dataset_output = self._dataset_output
        self.manifest.model_output = self._model_output
        self.manifest.stage_index = self._current_stage_index
        self.manifest.stage_iteration = self._current_stage_iteration
        self.manifest.replanning_attempts = self._replanning_attempts
        self.manifest.knowledge_card_id = self.knowledge_card_id
        save_manifest(self.config.runs_root, self.manifest)

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
    # Resume
    # ------------------------------------------------------------------
    @classmethod
    def resume_from_manifest(
        cls,
        manifest: RunManifest,
        *,
        run_script_resolver: Callable[[str], Sequence[str]] | None = None,
        **agent_kwargs: Any,
    ) -> StateMachine:
        """从一份持久化的`RunManifest`恢复出一个可继续`resume()`的`StateMachine`。

        `agent_kwargs`透传给`__init__`，供测试注入fake agent；生产路径留空即可
        按`configs/agents.yaml`重新构造真实Agent（Agent对象本身不可跨进程
        序列化，resume本来就需要重新构造，这与现状代码"每次全新构造"的既有
        设计一致，不是新增的限制）。
        """
        if manifest.status in ("DONE", "FAILED", "CANCELLED"):
            raise ValueError(
                f"run_id={manifest.run_id!r}状态为{manifest.status}（终态），不可恢复，"
                "请发起一次新的run"
            )
        pipeline_state = PipelineState(manifest.pipeline_state)
        if pipeline_state not in _RESUMABLE_STATES:
            raise ValueError(f"run_id={manifest.run_id!r}当前状态{pipeline_state.value}不支持恢复")

        context = RunContext(**manifest.context)
        config = _config_from_dict(manifest.config, run_script_resolver=run_script_resolver)

        state_machine = cls(context=context, config=config, **agent_kwargs)
        state_machine._restore_from_manifest(manifest)
        return state_machine

    def _restore_from_manifest(self, manifest: RunManifest) -> None:
        self.manifest = manifest
        self.manifest.owner_pid = os.getpid()
        self.manifest.status = "RUNNING"
        self.state = PipelineState(manifest.pipeline_state)
        self._scenario_output = manifest.scenario_output
        self._dataset_output = manifest.dataset_output
        self._model_output = manifest.model_output
        self._current_stage_index = manifest.stage_index
        self._current_stage_iteration = manifest.stage_iteration
        self._replanning_attempts = manifest.replanning_attempts
        self.knowledge_card_id = manifest.knowledge_card_id

        plan_json_path = self.run_dir / "training_plan.json"
        if plan_json_path.exists():
            self.training_plan = TrainingPlanOutput(
                **json.loads(plan_json_path.read_text(encoding="utf-8"))
            )
        self._save_manifest()

    def resume(self) -> PipelineState:
        """从`resume_from_manifest()`恢复的状态继续跑完剩余闭环，返回最终状态。"""
        if self.state in (PipelineState.PLANNING, PipelineState.PLAN_REVIEW):
            self._run_planning_loop()
            if self.state == PipelineState.FAILED:
                return self.state
            self._run_report()
            self._run_training_loop()
        elif self.state == PipelineState.REPORT:
            self._run_report()
            self._run_training_loop()
        elif self.state in (PipelineState.TRAINING, PipelineState.MONITORING, PipelineState.EVALUATING):
            self._resume_training_loop()
        elif self.state == PipelineState.KNOWLEDGE_UPDATE:
            pass
        else:  # pragma: no cover - resume_from_manifest已校验过状态
            raise ValueError(f"不支持从状态{self.state.value}恢复")

        if self.state == PipelineState.FAILED:
            return self.state

        self._run_knowledge_update()
        self._transition(PipelineState.DONE)
        return self.state

    def _resume_training_loop(self) -> None:
        """恢复到某个正在训练/监控/评测中的stage：优先重新接管仍存活的训练pid。"""
        assert self.training_plan is not None
        stages = self.training_plan.pipeline_stages
        stage_index = self._current_stage_index or 1

        if stage_index > len(stages):
            self._transition(PipelineState.TRAINING)
            self._run_training_loop(start_stage_index=stage_index)
            return

        reattach: dict[str, Any] | None = None
        pid = self.manifest.training_pid
        exit_code_path = self.manifest.training_exit_code_path
        if pid is not None and exit_code_path is not None and is_pid_alive(pid):
            reattach = {
                "pid": pid,
                "exit_code_path": Path(exit_code_path),
                "iteration": self._current_stage_iteration or 1,
            }

        stage = stages[stage_index - 1]
        if stage_index > 1:
            prev_stage = stages[stage_index - 2]
            start_from_path = str(
                stage_dir(self.config.runs_root, self.ctx.run_id, stage_index - 1, prev_stage.name)
                / "checkpoints"
            )
        else:
            start_from_path = str((self._model_output or {}).get("recommended_model") or "base_model")

        data_path = self.trainer_agent.prepare_data(
            self.training_plan.data_format,
            self.ctx.dataset_records,
            self.ctx.run_id,
            runs_root=self.config.runs_root,
        )

        self._transition(PipelineState.TRAINING)
        outcome = self._run_single_stage(stage, stage_index, start_from_path, data_path, reattach=reattach)
        if outcome is None:
            self._transition(PipelineState.FAILED)
            return

        if outcome.needs_replanning:
            self._handle_stage_needs_replanning()
            if self.state == PipelineState.FAILED:
                return
            self._run_training_loop()
            return

        self._transition(PipelineState.TRAINING)
        self._run_training_loop(start_stage_index=stage_index + 1)

    # ------------------------------------------------------------------
    # Planning阶段
    # ------------------------------------------------------------------
    def _call_scenario(self) -> None:
        result = self.scenario_agent.run(
            task_description=self.ctx.task_description,
            indicators=self.ctx.indicators,
            resource_constraints=self.ctx.resource_constraints,
            on_event=self._on_event,
        )
        assert result.structured_output is not None
        self._scenario_output = result.structured_output
        self._save_manifest()

    def _call_dataset(self) -> None:
        assert self._scenario_output is not None
        result = self.dataset_agent.run(
            task_description=self.ctx.task_description,
            dataset_path=self.ctx.dataset_path,
            dataset_sample=self.ctx.dataset_sample,
            scenario_summary=json.dumps(self._scenario_output, ensure_ascii=False),
            on_event=self._on_event,
        )
        assert result.structured_output is not None
        self._dataset_output = result.structured_output
        self._save_manifest()

    def _call_model_selection(self) -> None:
        assert self._scenario_output is not None and self._dataset_output is not None
        result = self.model_agent.run(
            task_description=self.ctx.task_description,
            scenario_summary=json.dumps(self._scenario_output, ensure_ascii=False),
            dataset_summary=json.dumps(self._dataset_output, ensure_ascii=False),
            resource_constraints=self.ctx.resource_constraints,
            on_event=self._on_event,
        )
        assert result.structured_output is not None
        self._model_output = result.structured_output
        self._save_manifest()

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
            on_event=self._on_event,
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
            on_event=self._on_event,
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

            assert self.training_plan is not None
            decision = self._human_gate.review_plan(self.training_plan, review)
            if decision == "force_approve":
                review = review.model_copy(update={"approved": True})
            elif decision == "force_reject" and review.approved:
                review = review.model_copy(
                    update={
                        "approved": False,
                        "issues": [
                            ReviewIssue(category="计划参数", description="人工审阅后强制拒绝，需人工补充具体原因")
                        ],
                    }
                )
            # decision == "accept_llm_verdict"：review保持LLM原始判定不变。

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
                retry_decision = self._human_gate.on_max_retries_exceeded(
                    f"Plan Review已连续{attempt}次拒绝，超过最大重试次数"
                    f"{self.config.max_plan_review_retries}"
                )
                if retry_decision != "extend_retries":
                    append_markdown_log(
                        review_log_path, "## 终止\n已达最大评审重试次数，需人工介入。"
                    )
                    self._transition(PipelineState.FAILED)
                    return
                append_markdown_log(review_log_path, "## 人工决定：延长一次重试次数，继续评审循环。")
                self.config.max_plan_review_retries += 1

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
            on_event=self._on_event,
        )
        assert result.structured_output is not None
        write_markdown(self.run_dir / "analysis_report.md", result.structured_output["markdown"])

    # ------------------------------------------------------------------
    # Execution阶段
    # ------------------------------------------------------------------
    def _handle_stage_needs_replanning(self) -> None:
        self._replanning_attempts += 1
        self._save_manifest()
        if self._replanning_attempts > self.config.max_replanning_attempts:
            retry_decision = self._human_gate.on_max_retries_exceeded(
                f"Evaluator已判定需要重新规划{self._replanning_attempts}次，超过最大重规划次数"
                f"{self.config.max_replanning_attempts}"
            )
            if retry_decision != "extend_retries":
                self._transition(PipelineState.FAILED)
                return
            self.config.max_replanning_attempts += 1
        self._run_planning_loop()
        if self.state == PipelineState.FAILED:
            return
        self._run_report()
        self._transition(PipelineState.TRAINING)

    def _run_training_loop(
        self, start_stage_index: int = 1, start_from_path_override: str | None = None
    ) -> None:
        assert self.training_plan is not None
        self._transition(PipelineState.TRAINING)

        while True:
            data_path = self.trainer_agent.prepare_data(
                self.training_plan.data_format,
                self.ctx.dataset_records,
                self.ctx.run_id,
                runs_root=self.config.runs_root,
            )
            start_from_path = start_from_path_override or str(
                (self._model_output or {}).get("recommended_model") or "base_model"
            )
            # 只对紧接着resume之后的第一轮生效；若本轮触发重新规划需要从头再跑，
            # 下面会在for循环结束后把start_stage_index重置为1。
            start_from_path_override = None

            needs_replanning = False
            for stage_index, stage in enumerate(self.training_plan.pipeline_stages, start=1):
                if stage_index < start_stage_index:
                    continue
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

            start_stage_index = 1
            self._transition(PipelineState.TRAINING)
            if not needs_replanning:
                return

            self._handle_stage_needs_replanning()
            if self.state == PipelineState.FAILED:
                return

    def _run_single_stage(
        self,
        stage: Any,
        stage_index: int,
        start_from_path: str,
        data_path: Path,
        reattach: dict[str, Any] | None = None,
    ) -> EvaluatorOutput | None:
        iteration = reattach["iteration"] if reattach else 1
        while True:
            self._current_stage_index = stage_index
            self._current_stage_iteration = iteration
            self._save_manifest()

            if reattach is not None and iteration == reattach["iteration"]:
                exit_code, crash_detected = self._monitor_until_pid_done(
                    reattach["pid"], reattach["exit_code_path"]
                )
                reattach = None
            else:
                config_path = self.trainer_agent.build_stage_config(
                    stage=stage,
                    resource_plan=self.training_plan.resource_plan if self.training_plan else {},
                    start_from_path=start_from_path,
                    run_id=self.ctx.run_id,
                    stage_index=stage_index,
                    dataset_path=str(data_path),
                    runs_root=self.config.runs_root,
                    on_event=self._on_event,
                )
                proc, exit_code_path = self.trainer_agent.launch_stage(
                    stage=stage,
                    config_path=config_path,
                    run_id=self.ctx.run_id,
                    stage_index=stage_index,
                    logger_type=self.config.logger_type,
                    runs_root=self.config.runs_root,
                    logs_root=self.config.logs_root,
                    run_script_resolver=self.config.run_script_resolver,
                )
                self.manifest.training_pid = proc.pid
                self.manifest.training_exit_code_path = str(exit_code_path)
                self._save_manifest()

                exit_code, crash_detected = self._monitor_until_stage_done(proc)

            self.manifest.training_pid = None
            self._save_manifest()

            self._transition(PipelineState.EVALUATING)
            result = self.evaluator_agent.evaluate(
                run_id=self.ctx.run_id,
                iteration=iteration,
                indicators=self.ctx.indicators,
                hyperparams_snapshot={"stage": stage.name, "key_hyperparams": stage.key_hyperparams},
                process_exit_code=exit_code,
                crash_detected=crash_detected,
                runs_root=self.config.runs_root,
                on_event=self._on_event,
            )

            if result.verdict == "PASS":
                self._transition(PipelineState.TRAINING)
                return result

            fail_decision = self._human_gate.on_stage_fail(result)
            if fail_decision == "abort":
                return None
            if fail_decision == "retry":
                result = result.model_copy(update={"needs_replanning": False})
            elif fail_decision == "replan":
                result = result.model_copy(update={"needs_replanning": True})
            # fail_decision == "accept_llm_verdict"：result保持LLM原始判定不变。

            if result.needs_replanning:
                return result
            if iteration > self.config.max_stage_retries:
                retry_decision = self._human_gate.on_max_retries_exceeded(
                    f"训练阶段'{stage.name}'已连续FAIL {iteration}次，超过最大阶段重试次数"
                    f"{self.config.max_stage_retries}"
                )
                if retry_decision != "extend_retries":
                    return None
                self.config.max_stage_retries += 1
            self._transition(PipelineState.TRAINING)
            iteration += 1

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
                on_event=self._on_event,
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

    def _monitor_until_pid_done(self, pid: int, exit_code_path: Path) -> tuple[int | None, bool]:
        """resume场景下重新接管一个非本进程fork出来的训练pid，用存活探测代替
        `Popen.poll()`，退出码从`launch_stage()`写入的哨兵文件读取（见
        `agents/execution/trainer_agent.py:read_exit_code`）。
        """
        self._transition(PipelineState.MONITORING)
        log_dir = self.config.logs_root / self.ctx.run_id / self.config.logger_type

        crash_detected = False
        polls = 0
        while polls < self.config.max_polls_per_stage:
            if not is_pid_alive(pid):
                break
            if self.config.poll_interval_seconds > 0:
                time.sleep(self.config.poll_interval_seconds)

            report = self.monitor_agent.poll_once(
                run_id=self.ctx.run_id,
                log_dir=log_dir,
                logger_type=self.config.logger_type,
                indicators=self.ctx.indicators,
                runs_root=self.config.runs_root,
                on_event=self._on_event,
            )
            polls += 1
            if report.crash_detected or report.risk_level == "Critical":
                crash_detected = report.crash_detected
                break
            if not is_pid_alive(pid):
                break

        return read_exit_code(exit_code_path), crash_detected

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
            on_event=self._on_event,
        )
        self._save_manifest()

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
