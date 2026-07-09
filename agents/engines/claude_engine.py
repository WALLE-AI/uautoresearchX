"""claude CLI 的 BaseAgentEngine 适配层：`claude -p` NDJSON stream-json 协议客户端。

实测结论（当前环境 claude-code 2.1.148）：
- 通过 `claude -p --input-format stream-json --output-format stream-json
  --include-partial-messages --verbose` 启动长驻进程，stdin/stdout 为逐行
  NDJSON，非JSON-RPC（消息无id字段，靠 `type` 字段区分，不需要请求/响应配对）。
- 实测抓到的启动事件：
  `{"type":"system","subtype":"init","session_id":...,"model":...,"tools":[...]}`
  以及网络重试事件 `{"type":"system","subtype":"api_retry",...}`（非致命）。
- 公开的Claude Code stream-json协议（官方SDK文档）里，一轮对话由：
  `system(init)` -> `assistant`(完整消息) / `stream_event`(增量,需
  --include-partial-messages) -> `result`(终态，含最终文本)组成。
- 结构化输出通过 `--json-schema` CLI参数直传JSON Schema。
- 该CLI未见文档化的“取消当前turn”控制消息，`cancel()`为最佳努力实现
  （SIGINT），不保证可用；确定生效的取消方式是`stop()`整体终止进程。
"""

from __future__ import annotations

import json
import signal
import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from agents.engines.base_engine import AgentEvent, AgentResult, BaseAgentEngine
from agents.engines.process_manager import ProcessManager


class ClaudeEngineError(Exception):
    """claude_engine运行期错误。"""


class ClaudeEngine(BaseAgentEngine):
    def __init__(
        self,
        *,
        model: str = "default",
        permission_mode: str | None = None,
        cwd: str | None = None,
        command: list[str] | None = None,
    ) -> None:
        self._model = model
        self._permission_mode = permission_mode
        self._cwd = cwd
        self._command_override = command

        self._process: ProcessManager | None = None
        self._started = False

        self._reader_thread: threading.Thread | None = None
        self._read_lock = threading.Lock()

        # 当前turn的状态（run()调用期间被读线程写入）
        self._current_text_parts: list[str] | None = None
        self._current_events: list[AgentEvent] | None = None
        self._current_on_event: Callable[[AgentEvent], None] | None = None
        self._current_done: threading.Event | None = None
        self._current_error: dict[str, Any] | None = None
        self._current_result_text: str | None = None

    # ------------------------------------------------------------------
    def _build_command(self) -> list[str]:
        if self._command_override is not None:
            return self._command_override
        cmd = [
            "claude",
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
        ]
        if self._model and self._model != "default":
            cmd += ["--model", self._model]
        if self._permission_mode:
            cmd += ["--permission-mode", self._permission_mode]
        if self._cwd:
            cmd += ["--add-dir", self._cwd]
        return cmd

    def start(self) -> None:
        if self._started:
            return
        self._process = ProcessManager(self._build_command())
        proc = self._process.start()
        assert proc.stdout is not None
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="claude-reader", daemon=True
        )
        self._reader_thread.start()
        self._started = True

    def _read_loop(self) -> None:
        assert self._process is not None
        proc = self._process.process
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._dispatch(message)

    def _emit(self, event: AgentEvent) -> None:
        if self._current_events is not None:
            self._current_events.append(event)
        if self._current_on_event is not None:
            self._current_on_event(event)

    def _dispatch(self, message: dict) -> None:
        msg_type = message.get("type")

        if msg_type == "stream_event":
            event = message.get("event", {})
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                text = delta.get("text", "")
                if text and self._current_text_parts is not None:
                    self._current_text_parts.append(text)
                self._emit(AgentEvent(type="text_delta", payload={"text": text}, raw=message))
            elif event.get("type") in ("content_block_start", "content_block_stop"):
                self._emit(AgentEvent(type="tool_use_start", payload=event, raw=message))
            return

        if msg_type == "assistant":
            # 未启用partial时，assistant消息携带完整content blocks
            content = message.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "text" and self._current_text_parts is not None:
                    text = block.get("text", "")
                    if text and not self._current_text_parts:
                        self._current_text_parts.append(text)
            self._emit(AgentEvent(type="tool_use_end", payload=message, raw=message))
            return

        if msg_type == "result":
            self._current_result_text = message.get("result")
            if message.get("subtype") != "success":
                self._current_error = message
            self._emit(AgentEvent(type="done", payload=message, raw=message))
            if self._current_done is not None:
                self._current_done.set()
            return

        if msg_type == "system" and message.get("subtype") not in (None, "init", "api_retry"):
            self._emit(AgentEvent(type="error", payload=message, raw=message))
            return

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
        assert self._process is not None

        # 注意：claude CLI的 `--json-schema` 是启动期参数，无法按每次run()调用
        # 动态切换（该engine实例复用同一长驻进程）。因此output_schema仅在此处
        # 做“事后pydantic校验”，不像codex_engine那样有服务端强约束。若某次
        # run()确需严格结构化输出，建议为该Agent单独配置一个ClaudeEngine实例。
        with self._read_lock:
            self._current_text_parts = []
            self._current_events = []
            self._current_on_event = on_event
            self._current_done = threading.Event()
            self._current_error = None
            self._current_result_text = None

            combined_prompt = user_prompt
            if system_prompt:
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"

            input_message: dict[str, Any] = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": combined_prompt}],
                },
            }

            proc = self._process.process
            assert proc.stdin is not None
            proc.stdin.write((json.dumps(input_message) + "\n").encode("utf-8"))
            proc.stdin.flush()

            done = self._current_done

        if not done.wait(timeout=timeout):
            raise TimeoutError(f"claude 在 {timeout}s 内未返回result")

        if self._current_error is not None:
            raise ClaudeEngineError(f"claude 执行出错: {self._current_error}")

        text = self._current_result_text or "".join(self._current_text_parts or [])
        events = self._current_events or []

        structured_output: dict[str, Any] | None = None
        if output_schema is not None:
            try:
                structured_output = output_schema.model_validate_json(text).model_dump()
            except ValidationError as exc:
                raise ClaudeEngineError(f"输出未通过output_schema校验: {exc}") from exc

        return AgentResult(
            text=text,
            structured_output=structured_output,
            usage=None,
            events=events,
        )

    def cancel(self) -> None:
        if self._process is None:
            return
        try:
            proc = self._process.process
            proc.send_signal(signal.SIGINT)
        except Exception:  # noqa: BLE001 - cancel应尽力而为
            pass

    def stop(self) -> None:
        if self._process is not None:
            self._process.stop()
        self._started = False
