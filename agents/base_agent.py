"""BaseAgent 抽象基类：读取 `configs/agents.yaml` 配置、构造对应的
`BaseAgentEngine`、拼装system/user prompt、调用engine.run()，并对
`output_schema`（pydantic模型）做结构化输出校验+失败重试。

各具体Agent（Planning六个 + Execution三个 + Knowledge-Update）继承本类，
仅需实现 `build_system_prompt`/`build_user_prompt`，并可选声明
`output_schema`。engine的选择/超时/权限模式等均由 `configs/agents.yaml`
中与 `agent_id` 同名的条目决定，Agent子类本身不感知底层CLI差异。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from agents.engines.base_engine import AgentEvent, AgentResult, BaseAgentEngine
from agents.engines.claude_engine import ClaudeEngine, ClaudeEngineError
from agents.engines.codex_engine import CodexEngine, CodexEngineError
from agents.engines.opencode_engine import OpencodeEngine, OpencodeEngineError
from configs.validate_agents import AgentConfig, load_agents_config

_ENGINE_CLASSES: dict[str, type[BaseAgentEngine]] = {
    "codex": CodexEngine,
    "claude": ClaudeEngine,
    "opencode": OpencodeEngine,
}

# 各engine在结构化输出校验失败时抛出的专属异常类型，BaseAgent统一捕获用于重试判定。
_ENGINE_OUTPUT_ERRORS: tuple[type[Exception], ...] = (
    CodexEngineError,
    ClaudeEngineError,
    OpencodeEngineError,
    ValidationError,
)


class AgentConfigError(Exception):
    """`configs/agents.yaml` 中缺少某Agent条目，或engine字段非法。"""


class AgentRunError(Exception):
    """Agent在耗尽重试次数后仍未能产出合法输出。"""


def build_engine(config: AgentConfig, *, cwd: str | None = None) -> BaseAgentEngine:
    """依据单个Agent的 `AgentConfig` 实例化对应的 `BaseAgentEngine`。"""
    engine_cls = _ENGINE_CLASSES.get(config.engine)
    if engine_cls is None:
        raise AgentConfigError(f"未知engine类型: {config.engine!r}")

    kwargs: dict[str, Any] = {"model": config.model, "cwd": cwd}
    if config.engine == "codex" and config.sandbox:
        kwargs["sandbox"] = config.sandbox
    if config.engine == "claude" and config.permission_mode:
        kwargs["permission_mode"] = config.permission_mode

    return engine_cls(**kwargs)


class BaseAgent(ABC):
    """所有Agent的抽象基类。

    子类必须设置类属性 `agent_id`（须与 `configs/agents.yaml` 中的键一致），
    并实现 `build_system_prompt`/`build_user_prompt`。若需结构化输出校验，
    设置类属性 `output_schema` 为对应的pydantic模型类。
    """

    agent_id: ClassVar[str]
    output_schema: ClassVar[type[BaseModel] | None] = None
    max_retries: ClassVar[int] = 2

    def __init__(
        self,
        *,
        cwd: str | None = None,
        engine: BaseAgentEngine | None = None,
        log_dir: Path | None = None,
    ) -> None:
        if not getattr(self, "agent_id", None):
            raise AgentConfigError(f"{type(self).__name__} 未设置 agent_id 类属性")

        agents_config = load_agents_config()
        if self.agent_id not in agents_config:
            raise AgentConfigError(
                f"configs/agents.yaml 中缺少 agent_id={self.agent_id!r} 的配置条目"
            )
        self.config: AgentConfig = agents_config[self.agent_id]
        # `engine` 仅用于测试注入fake engine，生产代码留空即可按配置自动构建。
        self.engine: BaseAgentEngine = engine if engine is not None else build_engine(
            self.config, cwd=cwd
        )
        self._started = False
        # `log_dir` 非None时，每次run()调用（含每次重试）都会在此目录下落盘一份
        # 记录system/user prompt、返回文本、结构化输出、耗时、错误信息的JSON文件，
        # 供事后追溯每个Agent的真实LLM交互过程（此前完全没有持久化，只存在于
        # 内存中的AgentResult.events，进程一退出就不可追溯）。
        self.log_dir = log_dir
        self._call_counter = 0

    # ------------------------------------------------------------------
    @abstractmethod
    def build_system_prompt(self, **kwargs: Any) -> str:
        """构造本次调用的system prompt。"""
        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(self, **kwargs: Any) -> str:
        """构造本次调用的user prompt（通常包含任务上下文与输出格式要求）。"""
        raise NotImplementedError

    def build_retry_feedback(self, error: Exception, attempt: int) -> str:
        """结构化输出校验失败后，追加到下一次重试user_prompt末尾的纠错提示。

        子类可覆盖以提供更贴合自身output_schema的错误提示措辞。
        """
        return (
            f"\n\n[系统提示：第{attempt}次尝试的输出未通过校验，错误信息如下，"
            f"请修正后重新输出，严格符合要求的JSON结构]\n{error}"
        )

    # ------------------------------------------------------------------
    def run(
        self,
        on_event: Callable[[AgentEvent], None] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """执行一次Agent调用：拼装prompt -> engine.run() -> 校验 -> 按需重试。

        返回的 `AgentResult.structured_output` 在指定了 `output_schema` 且
        校验通过时保证非None；`kwargs` 原样转发给
        `build_system_prompt`/`build_user_prompt`，供子类构造prompt时使用。
        """
        if not self._started:
            self.engine.start()
            self._started = True

        system_prompt = self.build_system_prompt(**kwargs)
        base_user_prompt = self.build_user_prompt(**kwargs)

        last_error: Exception | None = None
        user_prompt = base_user_prompt
        for attempt in range(1, self.max_retries + 2):
            self._call_counter += 1
            started_at = time.monotonic()
            try:
                result = self.engine.run(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=self.output_schema,
                    on_event=on_event,
                    timeout=self.config.timeout,
                )
            except _ENGINE_OUTPUT_ERRORS as exc:
                self._write_call_log(
                    attempt=attempt,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    duration_seconds=time.monotonic() - started_at,
                    result=None,
                    error=exc,
                )
                last_error = exc
                user_prompt = base_user_prompt + self.build_retry_feedback(exc, attempt)
                continue
            except Exception as exc:  # noqa: BLE001 - 超时等非结构化输出类错误也要落盘再抛出
                self._write_call_log(
                    attempt=attempt,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    duration_seconds=time.monotonic() - started_at,
                    result=None,
                    error=exc,
                )
                raise

            self._write_call_log(
                attempt=attempt,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                duration_seconds=time.monotonic() - started_at,
                result=result,
                error=None,
            )
            return result

        raise AgentRunError(
            f"{self.agent_id} 在重试{self.max_retries}次后仍未产出合法输出: {last_error}"
        ) from last_error

    def _write_call_log(
        self,
        *,
        attempt: int,
        system_prompt: str,
        user_prompt: str,
        duration_seconds: float,
        result: AgentResult | None,
        error: Exception | None,
    ) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{self.agent_id}_{self._call_counter:03d}_attempt{attempt}.json"
        record: dict[str, Any] = {
            "agent_id": self.agent_id,
            "attempt": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 3),
            "engine": self.config.engine,
            "timeout": self.config.timeout,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        if result is not None:
            record["result_text"] = result.text
            record["structured_output"] = result.structured_output
            record["event_count"] = len(result.events)
        log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def stop(self) -> None:
        """释放底层engine长驻进程。"""
        if self._started:
            self.engine.stop()
            self._started = False

    def __enter__(self) -> BaseAgent:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()
