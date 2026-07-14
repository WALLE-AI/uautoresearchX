"""Knowledge-Update Agent：训练闭环结束（PASS或多轮FAIL终止）后触发，汇总
`training_plan.md`/`analysis_report.md`/`improve_guide.md`/关键`monitor_reports`
全文，生成Knowledge Card写入`knowledge_base/cards/<card_id>.json`并更新
`knowledge_base/index.json`。

`index.json`结构严格对齐`agents/planning/training_plan_generator.py`里
`_load_similar_cases_summary()`已经假定的形状（`{"entries": [{"card_id",
"summary","task_types"}, ...]}`），这样T2阶段已写好的检索逻辑无需任何改动
即可在T6完成后生效。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.engines.base_engine import AgentEvent
from agents.knowledge.schemas import KnowledgeCardOutput
from agents.planning.prompt_utils import format_kv_block, schema_instruction

_INDEX_FILENAME = "index.json"


def _load_index(knowledge_base_root: Path) -> dict[str, Any]:
    index_path = knowledge_base_root / _INDEX_FILENAME
    if not index_path.exists():
        return {"entries": []}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"entries": []}


def _save_index(knowledge_base_root: Path, index: dict[str, Any]) -> None:
    index_path = knowledge_base_root / _INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


class KnowledgeUpdateAgent(BaseAgent):
    agent_id = "knowledge_update"
    output_schema = KnowledgeCardOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名训练经验沉淀专家。汇总本次训练闭环的完整过程材料，提炼出结构化"
            "Knowledge Card：任务描述与数据集特征摘要、数据统计摘要、模型选型理由与"
            "最佳超参摘要、最终评估结果摘要、遇到的问题与解决方案（长尾问题处理/"
            "数据增强策略等经验总结）、最终采用的pipeline_stages流程（供未来相似"
            "任务直接复用）、本次任务归属的task_types标签列表（用于未来按任务类型"
            "检索匹配，应与Scenario-Analysis阶段使用的任务类型命名风格一致，如"
            "'llm-sft'/'cv-detect'等）。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        training_plan_markdown: str = kwargs.get("training_plan_markdown", "")
        analysis_report_markdown: str = kwargs.get("analysis_report_markdown", "")
        improve_guide_markdown: str = kwargs.get("improve_guide_markdown", "")
        monitor_reports_text: str = kwargs.get("monitor_reports_text", "")

        context = format_kv_block(
            "本次训练闭环完整材料",
            {
                "training_plan.md": training_plan_markdown,
                "analysis_report.md": analysis_report_markdown,
                "improve_guide.md": improve_guide_markdown,
                "关键monitor_reports摘录": monitor_reports_text,
            },
        )
        return f"{context}\n\n请生成Knowledge Card。\n\n{schema_instruction(self.output_schema)}"

    def write_card(
        self,
        card: KnowledgeCardOutput,
        run_id: str,
        knowledge_base_root: Path = Path("knowledge_base"),
    ) -> str:
        """把card落盘为`cards/<card_id>.json`并更新`index.json`，返回card_id。"""
        card_id = f"card_{run_id}"
        cards_dir = knowledge_base_root / "cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        (cards_dir / f"{card_id}.json").write_text(
            card.model_dump_json(indent=2), encoding="utf-8"
        )

        index = _load_index(knowledge_base_root)
        index.setdefault("entries", [])
        index["entries"] = [e for e in index["entries"] if e.get("card_id") != card_id]
        index["entries"].append(
            {
                "card_id": card_id,
                "summary": card.task_summary,
                "task_types": card.task_types,
            }
        )
        _save_index(knowledge_base_root, index)
        return card_id

    def run_and_save(
        self,
        run_id: str,
        training_plan_markdown: str,
        analysis_report_markdown: str,
        improve_guide_markdown: str,
        monitor_reports_text: str,
        knowledge_base_root: Path = Path("knowledge_base"),
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> str:
        """一次性完成LLM生成card + 落盘 + 更新index的完整流程，返回card_id。"""
        result = self.run(
            training_plan_markdown=training_plan_markdown,
            analysis_report_markdown=analysis_report_markdown,
            improve_guide_markdown=improve_guide_markdown,
            monitor_reports_text=monitor_reports_text,
            on_event=on_event,
        )
        assert result.structured_output is not None
        card = KnowledgeCardOutput(**result.structured_output)
        return self.write_card(card, run_id, knowledge_base_root)
