"""base_agent.py 单元测试。

用一个纯Python的`_ScriptedEngine`测试替身（实现BaseAgentEngine接口，行为
可编排：连续N次抛异常后再成功/一直失败），验证：
- 正常单次成功路径返回的AgentResult
- structured_output校验失败时按build_retry_feedback重试，重试成功后返回
- 重试耗尽后抛出AgentRunError
- configs/agents.yaml中缺少对应agent_id时抛出AgentConfigError
- start()/stop()生命周期只被调用一次（幂等语义由engine自身保证，这里验证
  BaseAgent只在首次run()前调用一次start()）
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agents.base_agent import AgentConfigError, AgentRunError, BaseAgent
from agents.engines.base_engine import AgentResult
from agents.engines.codex_engine import CodexEngineError
from tests.fakes.scripted_engine import ScriptedEngine as _ScriptedEngine


class _Answer(BaseModel):
    value: int


class _EchoAgent(BaseAgent):
    agent_id = "scenario_analysis"

    def build_system_prompt(self, **kwargs: Any) -> str:
        return "system prompt"

    def build_user_prompt(self, **kwargs: Any) -> str:
        return f"user prompt: {kwargs.get('topic', '')}"


class _StructuredAgent(_EchoAgent):
    output_schema = _Answer


def test_missing_agent_id_config_raises() -> None:
    class _BadAgent(BaseAgent):
        agent_id = "not_in_configs_yaml"

        def build_system_prompt(self, **kwargs: Any) -> str:
            return ""

        def build_user_prompt(self, **kwargs: Any) -> str:
            return ""

    with pytest.raises(AgentConfigError):
        _BadAgent(engine=_ScriptedEngine([]))


def test_run_success_first_try_calls_start_once() -> None:
    engine = _ScriptedEngine([AgentResult(text="hello")])
    agent = _EchoAgent(engine=engine)

    result = agent.run(topic="paris")

    assert result.text == "hello"
    assert engine.start_calls == 1
    assert engine.run_calls == [("system prompt", "user prompt: paris")]

    # 再次run()不应重复start()
    engine.script.append(AgentResult(text="again"))
    agent.run(topic="paris")
    assert engine.start_calls == 1


def test_run_retries_on_output_error_then_succeeds() -> None:
    engine = _ScriptedEngine(
        [
            CodexEngineError("输出未通过output_schema校验: 缺少字段value"),
            AgentResult(text='{"value": 42}', structured_output={"value": 42}),
        ]
    )
    agent = _StructuredAgent(engine=engine)

    result = agent.run()

    assert result.structured_output == {"value": 42}
    assert len(engine.run_calls) == 2
    # 第二次调用的user_prompt应包含重试纠错提示
    assert "系统提示" in engine.run_calls[1][1]


def test_run_exhausts_retries_raises_agent_run_error() -> None:
    # max_retries是类属性，用子类覆盖来测试固定重试次数=2 -> 共尝试3次
    class _LimitedAgent(_StructuredAgent):
        max_retries = 2

    engine = _ScriptedEngine([CodexEngineError("err1"), CodexEngineError("err2"), CodexEngineError("err3")])
    limited_agent = _LimitedAgent(engine=engine)

    with pytest.raises(AgentRunError):
        limited_agent.run()

    assert len(engine.run_calls) == 3


def test_stop_calls_engine_stop_once_and_is_idempotent() -> None:
    engine = _ScriptedEngine([AgentResult(text="hi")])
    agent = _EchoAgent(engine=engine)
    agent.run()

    agent.stop()
    agent.stop()

    assert engine.stop_calls == 1


def test_context_manager_calls_stop_on_exit() -> None:
    engine = _ScriptedEngine([AgentResult(text="hi")])
    with _EchoAgent(engine=engine) as agent:
        agent.run()
    assert engine.stop_calls == 1
