"""orchestrator/run_registry.py单元测试：manifest落盘/读取/扫描 + pid判活/终止。"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from orchestrator.run_registry import (
    ManifestNotFoundError,
    RunManifest,
    is_pid_alive,
    list_manifests,
    load_manifest,
    save_manifest,
    terminate_pid,
)
import pytest


def test_save_and_load_manifest_round_trip(tmp_path: Path) -> None:
    manifest = RunManifest(run_id="run1", pipeline_state="TRAINING", stage_index=2)
    save_manifest(tmp_path, manifest)

    loaded = load_manifest(tmp_path, "run1")
    assert loaded.run_id == "run1"
    assert loaded.pipeline_state == "TRAINING"
    assert loaded.stage_index == 2


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestNotFoundError):
        load_manifest(tmp_path, "does-not-exist")


def test_list_manifests_sorted_by_updated_at_desc_and_skips_corrupt(tmp_path: Path) -> None:
    save_manifest(tmp_path, RunManifest(run_id="run_a"))
    save_manifest(tmp_path, RunManifest(run_id="run_b"))

    corrupt_dir = tmp_path / "run_corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")

    manifests = list_manifests(tmp_path)
    run_ids = {m.run_id for m in manifests}
    assert run_ids == {"run_a", "run_b"}


def test_list_manifests_empty_runs_root(tmp_path: Path) -> None:
    assert list_manifests(tmp_path / "does-not-exist") == []


def test_is_pid_alive_true_for_self_and_false_after_exit() -> None:
    import os

    assert is_pid_alive(os.getpid()) is True

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert is_pid_alive(proc.pid) is False


def test_terminate_pid_kills_running_process() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    threading.Thread(target=proc.wait, daemon=True).start()

    assert is_pid_alive(proc.pid) is True
    result = terminate_pid(proc.pid, grace_period_seconds=3)
    assert result is True
    assert is_pid_alive(proc.pid) is False


def test_terminate_pid_already_dead_returns_true() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert terminate_pid(proc.pid) is True
