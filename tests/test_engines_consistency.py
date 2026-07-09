"""三个engine(codex/claude/opencode)一致性集成测试。

用 tests/fakes/ 下的最小协议模拟脚本代替真实CLI二进制，验证三者在
BaseAgentEngine接口下表现一致：start() -> run() -> AgentResult.text正确
聚合增量文本 -> stop()正常退出。不依赖网络/真实模型，可在CI中稳定运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agents.engines.base_engine import AgentResult, BaseAgentEngine
from agents.engines.claude_engine import ClaudeEngine
from agents.engines.codex_engine import CodexEngine
from agents.engines.opencode_engine import OpencodeEngine

FAKES_DIR = Path(__file__).parent / "fakes"


def _make_codex_engine() -> CodexEngine:
    return CodexEngine(
        command=[sys.executable, str(FAKES_DIR / "fake_codex_app_server.py")]
    )


def _make_opencode_engine() -> OpencodeEngine:
    return OpencodeEngine(
        command=[sys.executable, str(FAKES_DIR / "fake_opencode_acp.py")]
    )


def _make_claude_engine() -> ClaudeEngine:
    return ClaudeEngine(
        command=[sys.executable, str(FAKES_DIR / "fake_claude_stream_json.py")]
    )


ENGINE_FACTORIES = {
    "codex": _make_codex_engine,
    "opencode": _make_opencode_engine,
    "claude": _make_claude_engine,
}


@pytest.mark.parametrize("engine_name", list(ENGINE_FACTORIES.keys()))
def test_engine_is_base_agent_engine(engine_name: str) -> None:
    engine = ENGINE_FACTORIES[engine_name]()
    assert isinstance(engine, BaseAgentEngine)


@pytest.mark.parametrize("engine_name", list(ENGINE_FACTORIES.keys()))
def test_engine_run_returns_aggregated_text(engine_name: str) -> None:
    engine = ENGINE_FACTORIES[engine_name]()
    try:
        engine.start()
        result = engine.run(
            system_prompt="你是一个简洁的助手。",
            user_prompt="用一句话回答:1+1等于几",
            timeout=10,
        )
        assert isinstance(result, AgentResult)
        assert "1+1" in result.text
        assert "2" in result.text
    finally:
        engine.stop()


@pytest.mark.parametrize("engine_name", list(ENGINE_FACTORIES.keys()))
def test_engine_start_stop_lifecycle_is_idempotent(engine_name: str) -> None:
    engine = ENGINE_FACTORIES[engine_name]()
    engine.start()
    engine.start()  # 二次start()不应报错（幂等）
    engine.stop()
    engine.stop()  # 二次stop()不应报错（幂等）


@pytest.mark.parametrize("engine_name", list(ENGINE_FACTORIES.keys()))
def test_engine_run_emits_events(engine_name: str) -> None:
    engine = ENGINE_FACTORIES[engine_name]()
    try:
        engine.start()
        result = engine.run(
            system_prompt="",
            user_prompt="用一句话回答:1+1等于几",
            timeout=10,
        )
        assert len(result.events) > 0
        assert any(e.type in ("text_delta", "done") for e in result.events)
    finally:
        engine.stop()
