"""KnowledgeUpdateAgent单元测试：验证card落盘与index.json更新，且index.json
结构与`training_plan_generator.py:_load_similar_cases_summary()`已假定的形状
兼容（后者在T2阶段就已实现，本测试反向验证T6的写入契约与之一致）。"""

from __future__ import annotations

import json
from pathlib import Path

from agents.engines.base_engine import AgentResult
from agents.knowledge.knowledge_update_agent import KnowledgeUpdateAgent
from agents.knowledge.schemas import KnowledgeCardOutput
from agents.planning.training_plan_generator import _load_similar_cases_summary
from tests.fakes.scripted_engine import ScriptedEngine

_CARD_FIXTURE = KnowledgeCardOutput(
    task_summary="客服问答机器人SFT任务",
    dataset_stats_summary="5000条对话样本",
    model_and_hyperparams_summary="Qwen2.5-7B-Instruct, lr=2e-5",
    final_metrics_summary="满意度87%",
    lessons_learned=["长尾问题需数据增强"],
    reused_pipeline_stages=[],
    task_types=["llm-sft"],
)


def _result_for(model_instance) -> AgentResult:
    dumped = model_instance.model_dump()
    return AgentResult(text=json.dumps(dumped, ensure_ascii=False), structured_output=dumped)


def test_run_and_save_writes_card_and_updates_index(tmp_path: Path) -> None:
    kb_root = tmp_path / "knowledge_base"
    agent = KnowledgeUpdateAgent(engine=ScriptedEngine([_result_for(_CARD_FIXTURE)]))

    card_id = agent.run_and_save(
        run_id="run1",
        training_plan_markdown="# plan",
        analysis_report_markdown="# report",
        improve_guide_markdown="# guide",
        monitor_reports_text="",
        knowledge_base_root=kb_root,
    )

    card_path = kb_root / "cards" / f"{card_id}.json"
    assert card_path.exists()
    saved_card = json.loads(card_path.read_text(encoding="utf-8"))
    assert saved_card["task_summary"] == _CARD_FIXTURE.task_summary

    index = json.loads((kb_root / "index.json").read_text(encoding="utf-8"))
    assert index["entries"][0]["card_id"] == card_id
    assert index["entries"][0]["task_types"] == ["llm-sft"]
    agent.stop()


def test_write_card_is_idempotent_for_same_run_id(tmp_path: Path) -> None:
    kb_root = tmp_path / "knowledge_base"
    agent = KnowledgeUpdateAgent(engine=ScriptedEngine([]))
    agent.write_card(_CARD_FIXTURE, run_id="run1", knowledge_base_root=kb_root)
    agent.write_card(_CARD_FIXTURE, run_id="run1", knowledge_base_root=kb_root)

    index = json.loads((kb_root / "index.json").read_text(encoding="utf-8"))
    assert len(index["entries"]) == 1
    agent.stop()


def test_index_json_shape_is_consumable_by_training_plan_generator(
    tmp_path: Path, monkeypatch
) -> None:
    kb_root = tmp_path / "knowledge_base"
    agent = KnowledgeUpdateAgent(engine=ScriptedEngine([]))
    agent.write_card(_CARD_FIXTURE, run_id="run1", knowledge_base_root=kb_root)

    monkeypatch.setattr(
        "agents.planning.training_plan_generator._KNOWLEDGE_INDEX_PATH",
        kb_root / "index.json",
    )
    summary = _load_similar_cases_summary("llm-sft")
    assert "card_run1" in summary
    agent.stop()
