"""BaseAgent 抽象基类：读取 `configs/agents.yaml` 配置、构造对应的
`BaseAgentEngine`、拼装system/user prompt、调用engine.run()，并对
`output_schema`（pydantic模型）做结构化输出校验+失败重试。

各具体Agent（Planning六个 + Execution三个 + Knowledge-Update）继承本类，
仅需实现 `build_system_prompt`/`build_user_prompt`，并可选声明
`output_schema`。engine的选择/超时/权限模式等均由 `configs/agents.yaml`
中与 `agent_id` 同名的条目决定，Agent子类本身不感知底层CLI差异。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
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
            try:
                return self.engine.run(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=self.output_schema,
                    on_event=on_event,
                    timeout=self.config.timeout,
                )
            except _ENGINE_OUTPUT_ERRORS as exc:
                last_error = exc
                user_prompt = base_user_prompt + self.build_retry_feedback(exc, attempt)
                continue

        raise AgentRunError(
            f"{self.agent_id} 在重试{self.max_retries}次后仍未产出合法输出: {last_error}"
        ) from last_error

    def stop(self) -> None:
        """释放底层engine长驻进程。"""
        if self._started:
            self.engine.stop()
            self._started = False

    def __enter__(self) -> BaseAgent:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()
