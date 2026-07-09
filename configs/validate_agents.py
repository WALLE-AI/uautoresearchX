"""校验 configs/agents.yaml 是否符合Agent配置Schema。

用法：
    uv run python configs/validate_agents.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

CONFIG_PATH = Path(__file__).with_name("agents.yaml")

REQUIRED_AGENT_IDS = {
    "scenario_analysis",
    "dataset_analysis",
    "model_selection",
    "training_plan",
    "plan_reviewer",
    "report_writer",
    "trainer",
    "monitor",
    "evaluator",
    "knowledge_update",
}


class AgentConfig(BaseModel):
    engine: Literal["codex", "claude", "opencode"]
    model: str
    timeout: int
    permission_mode: str | None = None
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] | None = None
    interval_minutes: int | None = None


def load_agents_config(path: Path = CONFIG_PATH) -> dict[str, AgentConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {agent_id: AgentConfig(**entry) for agent_id, entry in raw.items()}


def main() -> int:
    try:
        agents = load_agents_config()
    except ValidationError as exc:
        print(f"[FAIL] Schema校验失败：\n{exc}")
        return 1
    except FileNotFoundError:
        print(f"[FAIL] 未找到配置文件：{CONFIG_PATH}")
        return 1

    missing = REQUIRED_AGENT_IDS - agents.keys()
    if missing:
        print(f"[FAIL] 缺少必需Agent条目：{sorted(missing)}")
        return 1

    for agent_id, cfg in agents.items():
        print(f"[OK] {agent_id}: {cfg.model_dump(exclude_none=True)}")

    print(f"\n[PASS] 共{len(agents)}个Agent条目，Schema校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
