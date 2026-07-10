"""log_adapters单元测试：为local/wandb/swanlab三种日志形态构造样例文件，
验证解析出的六字段符合预期。"""

from __future__ import annotations

import json
from pathlib import Path

from agents.log_adapters.local_log_adapter import LocalLogAdapter
from agents.log_adapters.swanlab_log_adapter import SwanlabLogAdapter
from agents.log_adapters.wandb_log_adapter import WandbLogAdapter


def test_local_adapter_returns_empty_when_no_log_file(tmp_path: Path) -> None:
    metrics = LocalLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.is_empty()


def test_local_adapter_parses_hf_trainer_dict_lines(tmp_path: Path) -> None:
    log_file = tmp_path / "train.log"
    log_file.write_text(
        "some banner text\n"
        "{'loss': 1.5, 'learning_rate': 2e-05, 'epoch': 0.1}\n"
        "{'loss': 0.8, 'learning_rate': 1e-05, 'epoch': 0.5, "
        "'train_samples_per_second': 12.3}\n"
        "Saving model checkpoint to runs/r1/checkpoints/checkpoint-100\n",
        encoding="utf-8",
    )
    metrics = LocalLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.loss == 0.8
    assert metrics.epoch == 0.5
    assert metrics.speed == 12.3
    assert metrics.checkpoint_path is not None


def test_local_adapter_parses_ultralytics_rows(tmp_path: Path) -> None:
    log_file = tmp_path / "train.log"
    log_file.write_text(
        "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
        "        1/10      2.1G      1.234      0.567      0.891         16        640\n"
        "        2/10      2.1G      0.987      0.456      0.789         16        640\n",
        encoding="utf-8",
    )
    metrics = LocalLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.loss == 0.987
    assert metrics.memory == 2.1


def test_wandb_adapter_returns_empty_when_no_wandb_dir(tmp_path: Path) -> None:
    metrics = WandbLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.is_empty()


def test_wandb_adapter_parses_summary_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "wandb" / "offline-run-20260710_120000-abc123" / "files"
    run_dir.mkdir(parents=True)
    summary = {
        "loss": 0.42,
        "epoch": 1.5,
        "eval/accuracy": 0.91,
        "train/train_samples_per_second": 33.3,
    }
    (run_dir / "wandb-summary.json").write_text(json.dumps(summary), encoding="utf-8")

    metrics = WandbLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.loss == 0.42
    assert metrics.epoch == 1.5
    assert metrics.metric == 0.91
    assert metrics.speed == 33.3


def test_wandb_adapter_picks_most_recently_modified_summary(tmp_path: Path) -> None:
    old_dir = tmp_path / "wandb" / "offline-run-1" / "files"
    new_dir = tmp_path / "wandb" / "offline-run-2" / "files"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "wandb-summary.json").write_text(json.dumps({"loss": 9.9}), encoding="utf-8")
    new_summary = new_dir / "wandb-summary.json"
    new_summary.write_text(json.dumps({"loss": 0.1}), encoding="utf-8")

    import os
    import time

    old_time = time.time() - 100
    os.utime(old_dir / "wandb-summary.json", (old_time, old_time))

    metrics = WandbLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.loss == 0.1


def test_swanlab_adapter_returns_empty_when_no_swanlab_dir(tmp_path: Path) -> None:
    metrics = SwanlabLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.is_empty()


def test_swanlab_adapter_parses_tag_logs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "swanlab" / "proj" / "exp1" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "loss.log").write_text(
        json.dumps({"index": 0, "data": 1.0}) + "\n" + json.dumps({"index": 1, "data": 0.5}) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "epoch.log").write_text(json.dumps({"index": 0, "data": 2.0}) + "\n", encoding="utf-8")

    metrics = SwanlabLogAdapter().read_latest_metrics(tmp_path)
    assert metrics.loss == 0.5
    assert metrics.epoch == 2.0
