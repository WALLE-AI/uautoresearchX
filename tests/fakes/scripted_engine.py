"""纯Python的BaseAgentEngine测试替身，按脚本顺序返回结果/抛出异常。

供 test_base_agent.py / test_planning_agents.py 等复用，避免为每个测试文件
重复定义同样的fake实现。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agents.engines.base_engine import AgentEvent, AgentResult, BaseAgentEngine


class ScriptedEngine(BaseAgentEngine):
    """测试替身：按脚本顺序返回AgentResult或抛出异常，记录调用历史。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.start_calls = 0
        self.stop_calls = 0
        self.run_calls: list[tuple[str, str]] = []

    def start(self) -> None:
        self.start_calls += 1

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel] | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        timeout: float = 120.0,
    ) -> AgentResult:
        self.run_calls.append((system_prompt, user_prompt))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def cancel(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_calls += 1
