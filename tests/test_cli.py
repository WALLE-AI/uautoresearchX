"""orchestrator/cli.py子命令单元测试：`list`/`status`/`cancel`/`logs`都是只读
或轻量操作，用`typer.testing.CliRunner`+手工构造的`manifest.json`验证。

`run`/`resume`两个会真正驱动`StateMachine`跑LLM调用的子命令，用
`monkeypatch`替换`orchestrator.cli.StateMachine`为一个不调用任何真实
Agent/engine的fake类，只验证CLI层面的参数透传/输出/退出码/回调接线是否正确——
真正的`StateMachine.run()`/`resume()`语义已由`tests/test_state_machine.py`
（含resume相关测试）与`tests/test_end_to_end_demo.py`覆盖。
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from orchestrator import cli as cli_module
from orchestrator.cli import app
from orchestrator.run_registry import RunManifest, save_manifest
from orchestrator.state_machine import PipelineState

runner = CliRunner()


def test_list_empty_runs_root(tmp_path: Path) -> None:
    result = runner.invoke(app, ["list", "--runs-root", str(tmp_path / "runs")])
    assert result.exit_code == 0
    assert "暂无运行记录" in result.stdout


def test_list_shows_manifests(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    save_manifest(
        runs_root,
        RunManifest(run_id="run_a", status="DONE", pipeline_state="DONE", task_description_preview="任务A"),
    )
    save_manifest(
        runs_root,
        RunManifest(
            run_id="run_b", status="RUNNING", pipeline_state="TRAINING", task_description_preview="任务B"
        ),
    )

    result = runner.invoke(app, ["list", "--runs-root", str(runs_root)])
    assert result.exit_code == 0
    assert "run_a" in result.stdout
    assert "run_b" in result.stdout


def test_status_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", "no-such-run", "--runs-root", str(tmp_path / "runs")])
    assert result.exit_code == 1
    assert "错误" in result.stdout


def test_status_reports_stage_and_pid_liveness(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    save_manifest(
        runs_root,
        RunManifest(
            run_id="run_c",
            status="RUNNING",
            pipeline_state="MONITORING",
            stage_index=2,
            stage_iteration=1,
            training_pid=999999999,  # 几乎不可能真实存在的pid，验证判活为dead
        ),
    )

    result = runner.invoke(app, ["status", "run_c", "--runs-root", str(runs_root)])
    assert result.exit_code == 0
    assert "pipeline_state: MONITORING" in result.stdout
    assert "stage_index: 2" in result.stdout
    assert "dead" in result.stdout


def test_cancel_already_terminal_is_noop(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    save_manifest(runs_root, RunManifest(run_id="run_done", status="DONE", pipeline_state="DONE"))

    result = runner.invoke(app, ["cancel", "run_done", "--runs-root", str(runs_root)])
    assert result.exit_code == 0
    assert "已处于终态" in result.stdout


def test_cancel_kills_alive_training_pid_and_marks_cancelled(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    threading.Thread(target=proc.wait, daemon=True).start()

    save_manifest(
        runs_root,
        RunManifest(
            run_id="run_running", status="RUNNING", pipeline_state="MONITORING", training_pid=proc.pid
        ),
    )

    result = runner.invoke(app, ["cancel", "run_running", "--runs-root", str(runs_root)])
    assert result.exit_code == 0
    assert "已标记为CANCELLED" in result.stdout

    from orchestrator.run_registry import load_manifest

    reloaded = load_manifest(runs_root, "run_running")
    assert reloaded.status == "CANCELLED"


def test_logs_missing_file_reports_absence(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    save_manifest(runs_root, RunManifest(run_id="run_nolog", status="RUNNING", config={"logger_type": "local"}))

    result = runner.invoke(
        app, ["logs", "run_nolog", "--runs-root", str(runs_root), "--logs-root", str(tmp_path / "logs")]
    )
    assert result.exit_code == 0
    assert "暂无日志文件" in result.stdout


def test_logs_prints_existing_file_content(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    save_manifest(runs_root, RunManifest(run_id="run_haslog", status="RUNNING", config={"logger_type": "local"}))

    log_dir = logs_root / "run_haslog" / "local"
    log_dir.mkdir(parents=True)
    (log_dir / "train.log").write_text("epoch=1 loss=0.5\n", encoding="utf-8")

    result = runner.invoke(
        app, ["logs", "run_haslog", "--runs-root", str(runs_root), "--logs-root", str(logs_root)]
    )
    assert result.exit_code == 0
    assert "loss=0.5" in result.stdout


# ----------------------------------------------------------------------
# run / resume：用fake StateMachine验证CLI层参数透传/输出/回调接线，
# 不触发任何真实LLM调用。
# ----------------------------------------------------------------------


class _FakeStateMachine:
    """记录构造参数与回调接线，`run()`/`resume()`直接返回DONE，不做任何真实工作。"""

    last_instance: "_FakeStateMachine | None" = None

    def __init__(self, context: Any, config: Any, human_gate: Any = None) -> None:
        self.ctx = context
        self.config = config
        self.human_gate = human_gate
        self.state = PipelineState.PLANNING
        self.knowledge_card_id: str | None = None
        self.on_event = None
        self.on_transition = None
        self.stopped = False
        _FakeStateMachine.last_instance = self

    @classmethod
    def resume_from_manifest(cls, manifest: RunManifest, human_gate: Any = None) -> "_FakeStateMachine":
        from dataclasses import fields

        from orchestrator.state_machine import RunContext

        context_fields = {f.name for f in fields(RunContext)}
        context = RunContext(**{k: v for k, v in manifest.context.items() if k in context_fields})
        instance = cls(context=context, config=cli_module.StateMachineConfig(), human_gate=human_gate)
        instance.state = PipelineState(manifest.pipeline_state)
        return instance

    def set_callbacks(self, *, on_event: Any = None, on_transition: Any = None) -> None:
        self.on_event = on_event
        self.on_transition = on_transition

    def run(self) -> PipelineState:
        self.state = PipelineState.DONE
        self.knowledge_card_id = "card_fake"
        return self.state

    def resume(self) -> PipelineState:
        self.state = PipelineState.DONE
        self.knowledge_card_id = "card_fake_resumed"
        return self.state

    def stop_all_agents(self) -> None:
        self.stopped = True

    def mark_failed(self, error: BaseException) -> None:
        self.marked_failed_with = error


class _CrashingStateMachine(_FakeStateMachine):
    """`run()`抛出未被状态机自身捕获的异常（模拟AgentRunError/TimeoutError），
    验证`cli.py`会在重新抛出前调用`mark_failed()`。
    """

    def run(self) -> PipelineState:
        raise RuntimeError("engine timed out")


def test_run_command_happy_path_with_fake_state_machine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "StateMachine", _FakeStateMachine)

    result = runner.invoke(
        app,
        [
            "run",
            "--task-description",
            "测试任务",
            "--run-id",
            "run_fake",
            "--runs-root",
            str(tmp_path / "runs"),
            "--logs-root",
            str(tmp_path / "logs"),
            "--knowledge-base-root",
            str(tmp_path / "kb"),
            "--no-tui",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "run_id=run_fake" in result.stdout
    assert "final_state=DONE" in result.stdout
    assert "knowledge_card_id=card_fake" in result.stdout

    instance = _FakeStateMachine.last_instance
    assert instance is not None
    assert instance.stopped is True
    # --no-tui时回调应保持None（NullTUI），不接入任何展示层。
    assert instance.on_event is None
    assert instance.on_transition is None


def test_resume_command_happy_path_with_fake_state_machine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "StateMachine", _FakeStateMachine)

    runs_root = tmp_path / "runs"
    save_manifest(
        runs_root,
        RunManifest(
            run_id="run_fake_resume",
            status="RUNNING",
            pipeline_state="TRAINING",
            context={
                "run_id": "run_fake_resume",
                "task_description": "测试任务",
            },
        ),
    )

    result = runner.invoke(
        app, ["resume", "run_fake_resume", "--runs-root", str(runs_root), "--no-tui"]
    )

    assert result.exit_code == 0, result.stdout
    assert "resume run_id=run_fake_resume" in result.stdout
    assert "final_state=DONE" in result.stdout
    assert "knowledge_card_id=card_fake_resumed" in result.stdout


def test_resume_command_reports_error_for_unknown_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["resume", "no-such-run", "--runs-root", str(tmp_path / "runs")])
    assert result.exit_code == 1
    assert "错误" in result.stdout


def test_run_command_marks_manifest_failed_on_unhandled_exception(
    tmp_path: Path, monkeypatch
) -> None:
    """回归测试：真实测试中发现，`AgentRunError`/`TimeoutError`这类未被状态机
    自身捕获的异常会让CLI进程崩溃退出，但如果`mark_failed()`没有被调用，
    `runs/<run_id>/manifest.json`会永远停留在`status=RUNNING`——即使进程早已
    不存在，`status`/`list`也无法区分"真的还在跑"和"已经崩溃废弃"。
    """
    monkeypatch.setattr(cli_module, "StateMachine", _CrashingStateMachine)

    result = runner.invoke(
        app,
        [
            "run",
            "--task-description",
            "测试任务",
            "--run-id",
            "run_crash",
            "--runs-root",
            str(tmp_path / "runs"),
            "--no-tui",
        ],
    )

    assert result.exit_code != 0
    instance = _FakeStateMachine.last_instance
    assert instance is not None
    assert isinstance(instance.marked_failed_with, RuntimeError)
    assert instance.stopped is True
