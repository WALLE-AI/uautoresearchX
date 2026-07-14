"""opencode CLI 的 BaseAgentEngine 适配层：`opencode acp` (Agent Client Protocol) 客户端。

实测结论（当前环境 opencode 1.14.40，本地vLLM Qwen3.5-35B-A3B作为默认模型）：
- `opencode acp` 是规范的 JSON-RPC 2.0 over stdio（带id，可直接复用
  `JsonRpcTransport`），流程为：
  `initialize` -> `session/new` (返回 `sessionId`) -> `session/prompt`
  （阻塞至本轮结束，返回 `{"stopReason":..., "usage":{...}}`）。
- 流式内容通过 `session/update` 通知推送，`params.update.sessionUpdate`
  取值实测包含：
    "agent_thought_chunk"  - 推理过程增量文本（params.update.content.text）
    "agent_message_chunk"  - 最终回答增量文本（同上）
    "usage_update"         - token用量增量
- 取消当前turn的方法名遵循ACP规范为 `session/cancel`（未在本环境实测到失败
  路径，按spec实现，失败时尽力而为不阻断上层）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.engines.base_engine import AgentEvent, AgentResult, BaseAgentEngine
from agents.engines.json_extraction import extract_json_payload
from agents.engines.jsonrpc_transport import JsonRpcTransport
from agents.engines.process_manager import ProcessManager


class OpencodeEngineError(Exception):
    """opencode_engine运行期错误。"""


class OpencodeEngine(BaseAgentEngine):
    def __init__(
        self,
        *,
        model: str = "default",
        cwd: str | None = None,
        command: list[str] | None = None,
    ) -> None:
        self._model = model
        self._cwd = cwd or "."

        self._process = ProcessManager(command or ["opencode", "acp"])
        self._transport: JsonRpcTransport | None = None
        self._started = False
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        proc = self._process.start()
        assert proc.stdin is not None and proc.stdout is not None
        self._transport = JsonRpcTransport(proc.stdin, proc.stdout)
        self._transport.start()

        self._transport.request(
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
            timeout=30,
        )
        self._started = True

    def _ensure_session(self) -> str:
        assert self._transport is not None
        if self._session_id is not None:
            return self._session_id

        params: dict[str, Any] = {"cwd": self._cwd, "mcpServers": []}
        response = self._transport.request("session/new", params, timeout=30)
        self._session_id = response["sessionId"]
        return self._session_id

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel] | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        timeout: float = 120.0,
    ) -> AgentResult:
        if not self._started:
            self.start()
        assert self._transport is not None

        session_id = self._ensure_session()

        text_parts: list[str] = []
        events: list[AgentEvent] = []

        def _emit(event: AgentEvent) -> None:
            events.append(event)
            if on_event is not None:
                on_event(event)

        def _on_session_update(params: dict) -> None:
            if params.get("sessionId") != session_id:
                return
            update = params.get("update", {})
            kind = update.get("sessionUpdate")

            if kind == "agent_message_chunk":
                text = update.get("content", {}).get("text", "")
                if text:
                    text_parts.append(text)
                _emit(AgentEvent(type="text_delta", payload={"text": text}, raw=params))
            elif kind == "agent_thought_chunk":
                text = update.get("content", {}).get("text", "")
                _emit(AgentEvent(type="tool_use_start", payload={"text": text}, raw=params))
            elif kind == "usage_update":
                _emit(AgentEvent(type="tool_use_end", payload=update, raw=params))

        self._transport.on_notification("session/update", _on_session_update)

        combined_prompt = user_prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        prompt_params: dict[str, Any] = {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": combined_prompt}],
        }

        try:
            result = self._transport.request("session/prompt", prompt_params, timeout=timeout)
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - 归一化为engine专属异常
            raise OpencodeEngineError(f"opencode session/prompt 执行出错: {exc}") from exc

        _emit(AgentEvent(type="done", payload=result, raw=result))

        text = "".join(text_parts)
        structured_output: dict[str, Any] | None = None
        if output_schema is not None:
            try:
                structured_output = output_schema.model_validate_json(
                    extract_json_payload(text)
                ).model_dump()
            except ValidationError as exc:
                raise OpencodeEngineError(f"输出未通过output_schema校验: {exc}") from exc

        return AgentResult(
            text=text,
            structured_output=structured_output,
            usage=result.get("usage") if isinstance(result, dict) else None,
            events=events,
        )

    def cancel(self) -> None:
        if self._transport is None or self._session_id is None:
            return
        try:
            self._transport.request(
                "session/cancel", {"sessionId": self._session_id}, timeout=10
            )
        except Exception:  # noqa: BLE001 - cancel应尽力而为
            pass

    def stop(self) -> None:
        if self._transport is not None:
            self._transport.stop()
        self._process.stop()
        self._started = False
        self._session_id = None
