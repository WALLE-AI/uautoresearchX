"""BaseAgentEngine 抽象接口 + AgentEvent / AgentResult 数据类。

各具体engine（codex_engine/claude_engine/opencode_engine）需实现本接口，
对上层 BaseAgent 屏蔽codex的JSON-RPC、claude的stream-json NDJSON、
opencode的ACP等协议差异。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    """统一事件模型，各engine内部协议差异在此层被归一化。

    type取值约定：
        "text_delta"     - 增量文本片段
        "tool_use_start" - 工具调用开始
        "tool_use_end"   - 工具调用结束
        "exec_output"    - 子命令/工具执行输出流
        "error"          - 错误事件（归一化的错误语义）
        "done"           - 本轮结束
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """engine.run() 的最终返回结果。"""

    text: str
    structured_output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    events: list[AgentEvent] = field(default_factory=list)


class BaseAgentEngine(ABC):
    """所有CLI引擎适配器的统一抽象接口。"""

    @abstractmethod
    def start(self) -> None:
        """拉起长驻子进程 + 协议握手（若协议需要）。"""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        timeout: float = 120.0,
    ) -> AgentResult:
        """发起一轮对话/任务并阻塞等待结果。

        对上层始终表现为阻塞调用：不传 on_event 等价于旧版
        subprocess.run 语义；传入 on_event 时会在拿到中间事件时
        实时回调（Monitor/Trainer可用于日志/实时展示）。

        若指定 output_schema，会在拿到完整文本后用其(pydantic模型)
        校验/解析 structured_output，失败则由调用方决定是否重试。
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> None:
        """中途取消当前turn（仅JSON-RPC类协议支持，其余可空实现）。"""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """优雅关闭：关闭stdin/发终止请求，超时后SIGTERM。"""
        raise NotImplementedError
