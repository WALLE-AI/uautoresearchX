"""实时进度TUI：用`rich.live.Live`把`StateMachine`的`on_event`/`on_transition`
回调渲染成一个持续刷新的终端面板（当前阶段/当前Agent流式输出/最近事件/训练
日志尾部）。

三个engine（`agents/engines/claude_engine.py`/`codex_engine.py`/
`opencode_engine.py`）已经完整实现`AgentEvent`流式回调，本模块只是给这些早已
存在的事件流接一个消费者，不涉及engine层改动。

训练阶段（MONITORING状态）的loss/epoch展示走独立的后台tail线程，直接读
`logs/<run_id>/<logger_type>/train.log`，刷新节奏比LLM驱动的Monitor轮询
（默认5分钟一次）快得多；这条信息流只影响展示，不参与任何PASS/FAIL判定。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from agents.engines.base_engine import AgentEvent

_TEXT_PREVIEW_CHARS = 2000
_MAX_RECENT_EVENTS = 6
_LOG_TAIL_LINES = 12
_LOG_TAIL_POLL_SECONDS = 2.0


class PipelineTUI:
    """`with PipelineTUI(...) as tui: state_machine.set_callbacks(on_event=tui.on_event,
    on_transition=tui.on_transition)`。
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        refresh_per_second: float = 4.0,
        log_tail_path_provider: Callable[[], Path | None] | None = None,
    ) -> None:
        self.console = console or Console()
        self._live = Live(console=self.console, refresh_per_second=refresh_per_second, transient=False)
        self._pipeline_state = "PLANNING"
        self._current_text = ""
        self._recent_events: list[str] = []
        self._log_tail_path_provider = log_tail_path_provider
        self._log_tail_lines: list[str] = []
        self._stop_tail = threading.Event()
        self._tail_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # StateMachine回调
    # ------------------------------------------------------------------
    def on_transition(self, old_state: str, new_state: str) -> None:
        self._pipeline_state = new_state
        self._current_text = ""
        self._recent_events.append(f"{old_state} -> {new_state}")
        self._recent_events = self._recent_events[-_MAX_RECENT_EVENTS:]
        self._refresh()

    def on_event(self, event: AgentEvent) -> None:
        if event.type == "text_delta":
            self._current_text += event.payload.get("text", "")
        elif event.type == "tool_use_start":
            self._recent_events.append(f"[tool开始] {event.payload.get('type', event.payload)}")
            self._recent_events = self._recent_events[-_MAX_RECENT_EVENTS:]
        elif event.type == "error":
            self._recent_events.append(f"[错误] {event.payload}")
            self._recent_events = self._recent_events[-_MAX_RECENT_EVENTS:]
        self._refresh()

    # ------------------------------------------------------------------
    def _render(self) -> Group:
        state_panel = Panel(Text(self._pipeline_state, style="bold cyan"), title="当前阶段")
        text_preview = self._current_text[-_TEXT_PREVIEW_CHARS:] or "(等待Agent输出...)"
        agent_panel = Panel(text_preview, title="当前Agent输出", height=12)
        events_text = "\n".join(self._recent_events) or "(暂无事件)"
        events_panel = Panel(events_text, title="最近事件")
        log_text = "\n".join(self._log_tail_lines) or "(暂无训练日志)"
        log_panel = Panel(log_text, title="训练日志尾部（独立于Monitor分析节奏）")
        return Group(state_panel, agent_panel, events_panel, log_panel)

    def _refresh(self) -> None:
        self._live.update(self._render())

    # ------------------------------------------------------------------
    def _tail_loop(self) -> None:
        last_len = -1
        while not self._stop_tail.is_set():
            path = self._log_tail_path_provider() if self._log_tail_path_provider else None
            if path is not None and path.exists():
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    content = ""
                if len(content) != last_len:
                    last_len = len(content)
                    self._log_tail_lines = content.splitlines()[-_LOG_TAIL_LINES:]
                    self._refresh()
            self._stop_tail.wait(_LOG_TAIL_POLL_SECONDS)

    def __enter__(self) -> PipelineTUI:
        self._live.start()
        self._refresh()
        if self._log_tail_path_provider is not None:
            self._tail_thread = threading.Thread(target=self._tail_loop, daemon=True)
            self._tail_thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stop_tail.set()
        if self._tail_thread is not None:
            self._tail_thread.join(timeout=_LOG_TAIL_POLL_SECONDS + 1)
        self._live.stop()


class NullTUI:
    """`--no-tui`或非tty环境下的降级实现：`on_event`/`on_transition`均为`None`，
    `StateMachine`按现状代码路径直接退化为不回调，行为与升级前完全一致。
    """

    on_event: Callable[[AgentEvent], None] | None = None
    on_transition: Callable[[str, str], None] | None = None

    def __enter__(self) -> NullTUI:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


def build_tui(
    *,
    enabled: bool,
    is_tty: bool,
    console: Console | None = None,
    log_tail_path_provider: Callable[[], Path | None] | None = None,
) -> PipelineTUI | NullTUI:
    """`enabled`来自`--tui/--no-tui`，`is_tty`来自调用方的`sys.stdout.isatty()`
    探测——非tty环境下`rich.Live`会产生乱码/无意义输出，因此这里强制降级，而不是
    交给用户每次手动记得加`--no-tui`。
    """
    if enabled and is_tty:
        return PipelineTUI(console=console, log_tail_path_provider=log_tail_path_provider)
    return NullTUI()
