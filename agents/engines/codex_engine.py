"""codex CLI 的 BaseAgentEngine 适配层：`codex app-server` JSON-RPC 2.0 客户端。

实测结论（当前环境 codex-cli 0.128.0）：
- `codex app-server` 默认监听 `stdio://`（无需额外的 `--stdio` flag）。
- 协议真实方法名（通过 `codex app-server generate-json-schema` 核实）：
  `initialize` -> `thread/start` -> `turn/start`，流式通知
  `item/agentMessage/delta` / `item/started` / `item/completed` /
  `turn/started` / `turn/completed`，取消用 `turn/interrupt`。
- `approvalPolicy` 为字符串枚举（如 `"never"`），非嵌套对象。
- `sandboxPolicy` 为 `{"type": "workspaceWrite"|"readOnly"|"dangerFullAccess"|"externalSandbox", ...}`。
- `turn/start` 原生支持 `outputSchema` 字段，可直接传入JSON Schema由服务端约束最终assistant消息。
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

_VALID_SANDBOX_MODES = {"read-only", "workspace-write", "danger-full-access"}


class CodexEngineError(Exception):
    """codex_engine运行期错误（区别于底层JsonRpcError，用于携带更明确的上下文）。"""


class CodexEngine(BaseAgentEngine):
    def __init__(
        self,
        *,
        model: str = "default",
        sandbox: str | None = None,
        cwd: str | None = None,
        client_name: str = "uautoresearchx",
        client_version: str = "0.1.0",
        command: list[str] | None = None,
    ) -> None:
        self._model = model
        self._sandbox = sandbox
        self._cwd = cwd

        self._client_name = client_name
        self._client_version = client_version

        self._process = ProcessManager(command or ["codex", "app-server"])
        self._transport: JsonRpcTransport | None = None
        self._started = False

        self._thread_id: str | None = None

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
            {
                "clientInfo": {
                    "name": self._client_name,
                    "version": self._client_version,
                },
                "capabilities": {"experimentalApi": True},
            },
            timeout=30,
        )
        self._started = True

    def _ensure_thread(self) -> str:
        assert self._transport is not None
        if self._thread_id is not None:
            return self._thread_id

        params: dict[str, Any] = {"approvalPolicy": "never"}
        if self._sandbox and self._sandbox in _VALID_SANDBOX_MODES:
            params["sandbox"] = self._sandbox
        if self._cwd:
            params["cwd"] = self._cwd

        response = self._transport.request("thread/start", params, timeout=30)
        self._thread_id = response["thread"]["id"]
        return self._thread_id

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

        thread_id = self._ensure_thread()

        text_parts: list[str] = []
        events: list[AgentEvent] = []
        turn_done = threading.Event()
        turn_error: dict[str, Any] = {}
        current_turn_id: dict[str, str] = {}

        def _emit(event: AgentEvent) -> None:
            events.append(event)
            if on_event is not None:
                on_event(event)

        def _on_delta(params: dict) -> None:
            delta = params.get("delta", "")
            text_parts.append(delta)
            _emit(AgentEvent(type="text_delta", payload={"text": delta}, raw=params))

        def _on_item_started(params: dict) -> None:
            _emit(AgentEvent(type="tool_use_start", payload=params, raw=params))

        def _on_item_completed(params: dict) -> None:
            _emit(AgentEvent(type="tool_use_end", payload=params, raw=params))

        def _on_turn_started(params: dict) -> None:
            turn = params.get("turn", {})
            if isinstance(turn, dict) and turn.get("id"):
                current_turn_id["id"] = turn["id"]

        def _on_turn_completed(params: dict) -> None:
            _emit(AgentEvent(type="done", payload=params, raw=params))
            turn_done.set()

        def _on_error(params: dict) -> None:
            # `willRetry: true` 表示底层连接正在自动重连（如网络抖动），
            # 并非致命错误，turn仍可能最终完成，不应提前终止等待。
            _emit(AgentEvent(type="error", payload=params, raw=params))
            if not params.get("willRetry", False):
                turn_error["error"] = params
                turn_done.set()

        self._transport.on_notification("item/agentMessage/delta", _on_delta)
        self._transport.on_notification("item/started", _on_item_started)
        self._transport.on_notification("item/completed", _on_item_completed)
        self._transport.on_notification("turn/started", _on_turn_started)
        self._transport.on_notification("turn/completed", _on_turn_completed)
        self._transport.on_notification("error", _on_error)

        combined_prompt = user_prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        turn_params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": combined_prompt}],
        }
        if self._model and self._model != "default":
            turn_params["model"] = self._model
        if output_schema is not None:
            turn_params["outputSchema"] = output_schema.model_json_schema()

        self._transport.request("turn/start", turn_params, timeout=timeout)

        if not turn_done.wait(timeout=timeout):
            raise TimeoutError(f"codex turn/start 在 {timeout}s 内未完成")

        if turn_error:
            raise CodexEngineError(f"codex turn 执行出错: {turn_error['error']}")

        text = "".join(text_parts)
        structured_output: dict[str, Any] | None = None
        if output_schema is not None:
            try:
                structured_output = output_schema.model_validate_json(
                    extract_json_payload(text)
                ).model_dump()
            except ValidationError as exc:
                raise CodexEngineError(f"输出未通过output_schema校验: {exc}") from exc

        return AgentResult(
            text=text,
            structured_output=structured_output,
            usage=None,
            events=events,
        )

    def cancel(self) -> None:
        if self._transport is None or self._thread_id is None:
            return
        try:
            self._transport.request(
                "turn/interrupt", {"threadId": self._thread_id}, timeout=10
            )
        except Exception:  # noqa: BLE001 - cancel应尽力而为，不阻断上层清理
            pass

    def stop(self) -> None:
        if self._transport is not None:
            self._transport.stop()
        self._process.stop()
        self._started = False
        self._thread_id = None
