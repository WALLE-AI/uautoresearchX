"""长驻子进程生命周期管理。

负责启动、存活检测、优雅关闭（关stdin/发终止请求 -> 超时SIGTERM -> SIGKILL）、
异常退出后的可选自动重启策略。不感知具体JSON-RPC/NDJSON协议内容，只管理
subprocess.Popen本身的生命周期，供 codex_engine/claude_engine/opencode_engine
组合使用。
"""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass


class ProcessNotRunningError(Exception):
    """在进程未启动或已退出时尝试操作。"""


@dataclass
class RestartPolicy:
    """异常退出后的自动重启策略。

    enabled=False（默认）：不自动重启，交由上层Agent决定是否重新start()。
    对Planning类一次性调用场景通常无需重启；Monitor等长期运行场景可开启，
    但需注意重启会丢失协议握手状态（如codex的thread上下文），需谨慎使用。
    """

    enabled: bool = False
    max_attempts: int = 3
    backoff_seconds: float = 1.0


class ProcessManager:
    """管理单个长驻子进程的启动/存活检测/优雅关闭/重启。"""

    def __init__(
        self,
        command: Sequence[str],
        *,
        restart_policy: RestartPolicy | None = None,
        graceful_timeout: float = 5.0,
        on_exit: Callable[[int | None], None] | None = None,
    ) -> None:
        self._command = list(command)
        self._restart_policy = restart_policy or RestartPolicy()
        self._graceful_timeout = graceful_timeout
        self._on_exit = on_exit

        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._restart_attempts = 0
        self._watchdog_thread: threading.Thread | None = None
        self._stopping = threading.Event()

    # ------------------------------------------------------------------
    # 启动 / 存活检测
    # ------------------------------------------------------------------
    def start(self) -> subprocess.Popen:
        with self._lock:
            if self._proc is not None and self.is_alive():
                return self._proc
            self._stopping.clear()
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._restart_attempts = 0
            self._start_watchdog()
            return self._proc

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    @property
    def process(self) -> subprocess.Popen:
        if self._proc is None:
            raise ProcessNotRunningError("process has not been started")
        return self._proc

    # ------------------------------------------------------------------
    # 优雅关闭
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """优雅关闭：关闭stdin -> 等待graceful_timeout -> SIGTERM -> 短暂等待 -> SIGKILL。"""
        self._stopping.set()
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return

        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass

        try:
            proc.wait(timeout=self._graceful_timeout)
            return
        except subprocess.TimeoutExpired:
            pass

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=self._graceful_timeout)
            return
        except subprocess.TimeoutExpired:
            pass

        proc.kill()
        proc.wait(timeout=self._graceful_timeout)

    # ------------------------------------------------------------------
    # 异常退出监控 + 自动重启
    # ------------------------------------------------------------------
    def _start_watchdog(self) -> None:
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="process-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        exit_code = proc.wait()
        if self._stopping.is_set():
            if self._on_exit:
                self._on_exit(exit_code)
            return

        # 非主动stop()触发的退出视为异常退出
        if self._on_exit:
            self._on_exit(exit_code)

        if self._restart_policy.enabled and self._restart_attempts < self._restart_policy.max_attempts:
            self._restart_attempts += 1
            time.sleep(self._restart_policy.backoff_seconds)
            try:
                self.start()
            except OSError:
                pass
