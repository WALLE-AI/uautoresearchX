"""运行状态持久化：`RunManifest`落盘为`runs/<run_id>/manifest.json`，是
`resume`/`status`/`list`/`cancel`子命令共同依赖的地基。

`StateMachine`在每次状态切换（`_transition()`）与关键数据产出后写一份
manifest快照；CLI进程崩溃/被杀后，新进程可以通过`load_manifest()`读回最后一次
落盘的状态，用`StateMachine.resume_from_manifest()`重建对象继续跑，而不需要
任何跨进程的内存/IPC机制。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

RunStatus = Literal["RUNNING", "DONE", "FAILED", "CANCELLED"]

_MANIFEST_FILENAME = "manifest.json"

# 终态：一旦到达即不可再resume，只能重新发起一次新的run。
TERMINAL_STATUSES: frozenset[str] = frozenset({"DONE", "FAILED", "CANCELLED"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunManifest(BaseModel):
    """一次pipeline运行的可恢复状态快照。"""

    run_id: str
    status: RunStatus = "RUNNING"
    pipeline_state: str = "PLANNING"
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    owner_pid: int | None = None
    task_description_preview: str = ""

    # RunContext/StateMachineConfig的可序列化快照，供resume时重建。
    context: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)

    # Planning阶段中间产出（现状代码只落盘training_plan，这三项之前完全不持久化）。
    scenario_output: dict[str, Any] | None = None
    dataset_output: dict[str, Any] | None = None
    model_output: dict[str, Any] | None = None

    # Execution阶段进度。
    stage_index: int | None = None
    stage_iteration: int | None = None
    replanning_attempts: int = 0
    training_pid: int | None = None
    training_exit_code_path: str | None = None

    knowledge_card_id: str | None = None
    last_error: str | None = None


class ManifestNotFoundError(Exception):
    """指定run_id在runs_root下没有manifest.json（run_id不存在，或是旧版CLI跑的run）。"""


def manifest_path(runs_root: Path, run_id: str) -> Path:
    return runs_root / run_id / _MANIFEST_FILENAME


def save_manifest(runs_root: Path, manifest: RunManifest) -> None:
    manifest.updated_at = _now_iso()
    path = manifest_path(runs_root, manifest.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(runs_root: Path, run_id: str) -> RunManifest:
    path = manifest_path(runs_root, run_id)
    if not path.exists():
        raise ManifestNotFoundError(f"未找到run_id={run_id!r}的manifest（路径: {path}）")
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def list_manifests(runs_root: Path) -> list[RunManifest]:
    """扫描`runs_root`下全部子目录，返回按更新时间倒序的manifest列表。

    损坏/不完整的manifest.json直接跳过（不中断整个列表），因为这是一个只读的
    展示型接口，不应因为某一个run的文件被意外截断而让`list`命令整体失败。
    """
    if not runs_root.exists():
        return []
    manifests: list[RunManifest] = []
    for entry in sorted(runs_root.iterdir()):
        candidate = entry / _MANIFEST_FILENAME
        if not candidate.exists():
            continue
        try:
            manifests.append(RunManifest.model_validate_json(candidate.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    return sorted(manifests, key=lambda m: m.updated_at, reverse=True)


def is_pid_alive(pid: int) -> bool:
    """POSIX判活：向`pid`发0号信号（不会实际发送信号，只探测进程是否存在/可操作）。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但不属于当前用户，仍视为存活。
        return True
    return True


def terminate_pid(pid: int, *, grace_period_seconds: float = 10.0) -> bool:
    """尽力终止一个（可能不是本进程子进程的）pid：先SIGTERM优雅关闭，超时SIGKILL。

    返回True表示确认进程已不再存活；用于`cancel`子命令终止一次仍在跑的训练。
    """
    import signal
    import time

    if not is_pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.monotonic() + grace_period_seconds
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    time.sleep(0.2)
    return not is_pid_alive(pid)
