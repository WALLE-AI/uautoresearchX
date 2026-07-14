"""Execution阶段三个Agent（Trainer/Monitor/Evaluator）的单元测试。

Trainer的LLM相关方法与Monitor/Evaluator一样用`ScriptedEngine`注入预设结构化
输出；`launch_stage()`不涉及LLM，用`tests/fakes/fake_train_script.py`代替真实
`scripts/<engine>_run.sh`验证子进程能正常启动并按约定写入日志文件。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agents.engines.base_engine import AgentResult
from agents.execution.evaluator_agent import EvaluatorAgent
from agents.execution.monitor_agent import MonitorAgent
from agents.execution.schemas import EvaluatorOutput, MonitorReportOutput, StageConfigOutput
from agents.execution.trainer_agent import (
    TrainerAgent,
    TrainerAgentError,
    read_exit_code,
    validate_engine_registered,
)
from agents.planning.schemas import DataFormatSpec, PipelineStage
from tests.fakes.scripted_engine import ScriptedEngine

_FAKE_TRAIN_SCRIPT = Path(__file__).parent / "fakes" / "fake_train_script.py"


def _result_for(model_instance: Any) -> AgentResult:
    dumped = model_instance.model_dump()
    return AgentResult(text=json.dumps(dumped, ensure_ascii=False), structured_output=dumped)


def _fake_resolver(engine: str) -> list[str]:
    return [sys.executable, str(_FAKE_TRAIN_SCRIPT)]


# ----------------------------------------------------------------------
# TrainerAgent
# ----------------------------------------------------------------------

_SHAREGPT_DATA_FORMAT = DataFormatSpec(
    target_format="ShareGPT", rationale="多轮对话SFT", field_mapping=[]
)
_SFT_STAGE = PipelineStage(
    name="SFT",
    start_from="基础模型",
    goal="指令遵循能力对齐",
    engine="llamafactory",
    key_hyperparams="lr=2e-5, epoch=3",
    estimated_duration="4h",
)


def test_validate_engine_registered_accepts_known_engine() -> None:
    validate_engine_registered("llamafactory")


def test_validate_engine_registered_rejects_unknown_engine() -> None:
    with pytest.raises(TrainerAgentError):
        validate_engine_registered("not-a-real-engine")


def test_prepare_data_converts_to_target_format(tmp_path: Path) -> None:
    agent = TrainerAgent(engine=ScriptedEngine([]))
    records = [{"instruction": "翻译成英文", "input": "你好", "output": "Hello"}]
    result_path = agent.prepare_data(
        _SHAREGPT_DATA_FORMAT, records, run_id="run1", runs_root=tmp_path
    )
    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data[0]["conversations"][0]["from"] == "human"
    agent.stop()


def test_build_stage_config_generates_yaml_from_llm(tmp_path: Path) -> None:
    config_output = StageConfigOutput(
        yaml_content="model_name_or_path: Qwen2.5-7B-Instruct\nlearning_rate: 2e-5\nepoch: 3\n"
    )
    agent = TrainerAgent(engine=ScriptedEngine([_result_for(config_output)]))
    config_path = agent.build_stage_config(
        stage=_SFT_STAGE,
        resource_plan={"GPU": "4x A100-40GB"},
        start_from_path="Qwen2.5-7B-Instruct",
        run_id="run1",
        stage_index=1,
        runs_root=tmp_path,
    )
    assert config_path.exists()
    assert "learning_rate" in config_path.read_text(encoding="utf-8")
    agent.stop()


def test_build_stage_config_rejects_unregistered_engine_without_calling_llm(tmp_path: Path) -> None:
    bad_stage = _SFT_STAGE.model_copy(update={"engine": "not-a-real-engine"})
    agent = TrainerAgent(engine=ScriptedEngine([]))
    with pytest.raises(TrainerAgentError):
        agent.build_stage_config(
            stage=bad_stage,
            resource_plan={},
            start_from_path="base",
            run_id="run1",
            stage_index=1,
            runs_root=tmp_path,
        )
    agent.stop()


def test_launch_stage_runs_fake_script_and_produces_log(tmp_path: Path) -> None:
    agent = TrainerAgent(engine=ScriptedEngine([]))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("learning_rate: 2e-5\n", encoding="utf-8")

    proc, exit_code_path = agent.launch_stage(
        stage=_SFT_STAGE,
        config_path=config_path,
        run_id="run1",
        stage_index=1,
        logger_type="local",
        runs_root=tmp_path / "runs",
        logs_root=tmp_path / "logs",
        run_script_resolver=_fake_resolver,
    )
    exit_code = proc.wait(timeout=30)
    assert exit_code == 0

    log_file = tmp_path / "logs" / "run1" / "local" / "train.log"
    assert log_file.exists()
    assert "loss" in log_file.read_text(encoding="utf-8")

    # exit_code.txt由bash包装脚本写入，可能比proc.wait()的返回略晚一点点落盘。
    for _ in range(50):
        if exit_code_path.exists():
            break
        time.sleep(0.1)
    assert read_exit_code(exit_code_path) == 0
    agent.stop()


# ----------------------------------------------------------------------
# MonitorAgent
# ----------------------------------------------------------------------


def test_poll_once_appends_metrics_and_writes_report(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs" / "run1" / "local"
    log_dir.mkdir(parents=True)
    (log_dir / "train.log").write_text("{'loss': 0.5, 'epoch': 1.0}\n", encoding="utf-8")

    report_fixture = MonitorReportOutput(
        risk_level="Normal",
        gpu_observation="利用率正常",
        loss_trend="平稳下降",
        overfitting_signal="无",
        validation_accuracy="符合预期",
        recommendation="继续训练",
    )
    agent = MonitorAgent(engine=ScriptedEngine([_result_for(report_fixture)]))
    report = agent.poll_once(
        run_id="run1", log_dir=log_dir, logger_type="local", runs_root=tmp_path / "runs"
    )
    assert report.risk_level == "Normal"

    metrics_csv = tmp_path / "runs" / "run1" / "metrics.csv"
    assert metrics_csv.exists()
    assert "0.5" in metrics_csv.read_text(encoding="utf-8")

    report_file = tmp_path / "runs" / "run1" / "monitor_reports" / "report_1.md"
    assert report_file.exists()
    assert "Normal" in report_file.read_text(encoding="utf-8")
    agent.stop()


def test_poll_once_sequential_calls_increment_report_seq(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs" / "run1" / "local"
    log_dir.mkdir(parents=True)
    (log_dir / "train.log").write_text("{'loss': 0.5, 'epoch': 1.0}\n", encoding="utf-8")

    report_fixture = MonitorReportOutput(
        risk_level="Normal",
        gpu_observation="ok",
        loss_trend="ok",
        overfitting_signal="ok",
        validation_accuracy="ok",
        recommendation="ok",
    )
    agent = MonitorAgent(
        engine=ScriptedEngine([_result_for(report_fixture), _result_for(report_fixture)])
    )
    agent.poll_once(run_id="run1", log_dir=log_dir, logger_type="local", runs_root=tmp_path / "runs")
    agent.poll_once(run_id="run1", log_dir=log_dir, logger_type="local", runs_root=tmp_path / "runs")

    reports_dir = tmp_path / "runs" / "run1" / "monitor_reports"
    assert (reports_dir / "report_1.md").exists()
    assert (reports_dir / "report_2.md").exists()
    agent.stop()


# ----------------------------------------------------------------------
# EvaluatorAgent
# ----------------------------------------------------------------------


def test_evaluate_pass_path_appends_improve_guide(tmp_path: Path) -> None:
    pass_fixture = EvaluatorOutput(
        verdict="PASS", gap_analysis="已达标", improvement_suggestions=[], needs_replanning=False
    )
    agent = EvaluatorAgent(engine=ScriptedEngine([_result_for(pass_fixture)]))
    result = agent.evaluate(
        run_id="run1",
        iteration=1,
        indicators="满意度>=85%",
        hyperparams_snapshot={"lr": "2e-5"},
        runs_root=tmp_path,
    )
    assert result.verdict == "PASS"
    guide_path = tmp_path / "run1" / "improve_guide.md"
    assert "PASS" in guide_path.read_text(encoding="utf-8")
    agent.stop()


def test_evaluate_fail_hyperparam_level_path(tmp_path: Path) -> None:
    fail_fixture = EvaluatorOutput(
        verdict="FAIL",
        gap_analysis="loss未收敛",
        improvement_suggestions=["降低学习率"],
        needs_replanning=False,
    )
    agent = EvaluatorAgent(engine=ScriptedEngine([_result_for(fail_fixture)]))
    result = agent.evaluate(
        run_id="run1",
        iteration=1,
        indicators="满意度>=85%",
        hyperparams_snapshot={"lr": "1e-4"},
        runs_root=tmp_path,
    )
    assert result.verdict == "FAIL"
    assert result.needs_replanning is False
    agent.stop()


def test_evaluate_crash_short_circuits_without_calling_llm(tmp_path: Path) -> None:
    agent = EvaluatorAgent(engine=ScriptedEngine([]))  # 空脚本：若调用LLM会因pop(0)报错
    result = agent.evaluate(
        run_id="run1",
        iteration=1,
        indicators="满意度>=85%",
        hyperparams_snapshot={"lr": "2e-5"},
        process_exit_code=137,
        runs_root=tmp_path,
    )
    assert result.verdict == "FAIL"
    guide_path = tmp_path / "run1" / "improve_guide.md"
    assert "崩溃" in guide_path.read_text(encoding="utf-8") or "exit_code" in guide_path.read_text(
        encoding="utf-8"
    )
    agent.stop()
