"""通用 NDJSON stdio JSON-RPC 2.0 读写工具。

被 codex_engine / opencode_engine 共用：负责在一个已启动的长驻子进程的
stdin/stdout 之上，实现请求/响应按 id 匹配、通知（无 id 的 method 消息）
按 method 路由到回调、以及独立读线程持续拉取 stdout 避免阻塞写入。

不负责拉起/管理子进程本身（见 process_manager.py），只接收已打开的
可写/可读文件对象（通常是 Popen.stdin / Popen.stdout）。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any


class JsonRpcError(Exception):
    """远端返回的 JSON-RPC error 对象。"""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class TransportClosedError(Exception):
    """读线程检测到管道关闭（EOF）或读取异常时抛出。"""


@dataclass
class _PendingRequest:
    future: Future
    method: str


class JsonRpcTransport:
    """单个 stdin/stdout 管道对上的 JSON-RPC 2.0 收发器。

    使用方式：
        transport = JsonRpcTransport(proc.stdin, proc.stdout)
        transport.on_notification("item/agentMessage/delta", handler)
        transport.start()
        result = transport.request("initialize", {...}, timeout=30)
        transport.stop()
    """

    def __init__(
        self,
        stdin: Any,
        stdout: Any,
        *,
        include_jsonrpc_field: bool = True,
    ) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._include_jsonrpc_field = include_jsonrpc_field

        self._id_counter = 0
        self._id_lock = threading.Lock()
        self._pending: dict[int, _PendingRequest] = {}
        self._pending_lock = threading.Lock()

        self._notification_handlers: dict[str, list[Callable[[dict], None]]] = {}
        self._handlers_lock = threading.Lock()

        self._reader_thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._close_error: Exception | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动独立读线程，持续拉取stdout并分发消息。"""
        if self._reader_thread is not None:
            return
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="jsonrpc-reader", daemon=True
        )
        self._reader_thread.start()

    def stop(self) -> None:
        """标记关闭状态并让所有pending请求以异常结束。"""
        self._closed.set()
        self._fail_all_pending(TransportClosedError("transport stopped"))

    # ------------------------------------------------------------------
    # 通知路由注册
    # ------------------------------------------------------------------
    def on_notification(self, method: str, handler: Callable[[dict], None]) -> None:
        with self._handlers_lock:
            self._notification_handlers.setdefault(method, []).append(handler)

    # ------------------------------------------------------------------
    # 请求发送
    # ------------------------------------------------------------------
    def request(self, method: str, params: dict | None = None, *, timeout: float = 60.0) -> Any:
        """发送一个请求并阻塞等待匹配id的响应，超时抛TimeoutError。"""
        req_id = self._next_id()
        future: Future = Future()
        with self._pending_lock:
            self._pending[req_id] = _PendingRequest(future=future, method=method)

        message: dict[str, Any] = {"id": req_id, "method": method, "params": params or {}}
        if self._include_jsonrpc_field:
            message["jsonrpc"] = "2.0"
        self._write(message)

        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise

    def notify(self, method: str, params: dict | None = None) -> None:
        """发送无需响应的通知消息（无id）。"""
        message: dict[str, Any] = {"method": method, "params": params or {}}
        if self._include_jsonrpc_field:
            message["jsonrpc"] = "2.0"
        self._write(message)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _next_id(self) -> int:
        with self._id_lock:
            self._id_counter += 1
            return self._id_counter

    def _write(self, message: dict) -> None:
        line = json.dumps(message, ensure_ascii=False)
        data = line + "\n"
        if "b" in getattr(self._stdin, "mode", "b"):
            self._stdin.write(data.encode("utf-8"))
        else:
            self._stdin.write(data)
        self._stdin.flush()

    def _read_loop(self) -> None:
        try:
            for raw_line in iter(self._stdout.readline, b"" if self._is_binary_stdout() else ""):
                if not raw_line:
                    break
                self._dispatch_line(raw_line)
        except Exception as exc:  # noqa: BLE001 - 需要捕获所有异常以清理pending
            self._close_error = exc
        finally:
            self._closed.set()
            self._fail_all_pending(
                self._close_error or TransportClosedError("stdout closed (EOF)")
            )

    def _is_binary_stdout(self) -> bool:
        mode = getattr(self._stdout, "mode", "b")
        return "b" in mode

    def _dispatch_line(self, raw_line: bytes | str) -> None:
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        raw_line = raw_line.strip()
        if not raw_line:
            return
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            return

        if "id" in message and ("result" in message or "error" in message):
            self._handle_response(message)
        elif "method" in message:
            self._handle_notification(message)

    def _handle_response(self, message: dict) -> None:
        req_id = message["id"]
        with self._pending_lock:
            pending = self._pending.pop(req_id, None)
        if pending is None:
            return
        if "error" in message and message["error"] is not None:
            err = message["error"]
            pending.future.set_exception(
                JsonRpcError(
                    code=err.get("code", -1),
                    message=err.get("message", "unknown error"),
                    data=err.get("data"),
                )
            )
        else:
            pending.future.set_result(message.get("result"))

    def _handle_notification(self, message: dict) -> None:
        method = message["method"]
        params = message.get("params", {})
        with self._handlers_lock:
            handlers = list(self._notification_handlers.get(method, ()))
        for handler in handlers:
            try:
                handler(params)
            except Exception:  # noqa: BLE001 - 单个handler异常不应打断读循环
                continue

    def _fail_all_pending(self, exc: Exception) -> None:
        with self._pending_lock:
            pending_items = list(self._pending.items())
            self._pending.clear()
        for _req_id, pending in pending_items:
            if not pending.future.done():
                pending.future.set_exception(exc)
