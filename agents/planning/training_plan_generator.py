"""Training-Plan-Generator Agent：汇总前三个Planning Agent输出，产出完整
`training_plan.md`（含pipeline_stages多阶段流程与数据格式定案）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agents.base_agent import BaseAgent
from agents.planning.prompt_utils import format_kv_block, schema_instruction
from agents.planning.schemas import TrainingPlanOutput

_PIPELINE_PATTERNS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "training_pipeline_patterns.yaml"
)
_KNOWLEDGE_INDEX_PATH = (
    Path(__file__).resolve().parents[2] / "knowledge_base" / "index.json"
)


def _load_pipeline_patterns_summary() -> str:
    patterns = yaml.safe_load(_PIPELINE_PATTERNS_PATH.read_text(encoding="utf-8")) or {}
    lines = ["训练流程模式库（configs/training_pipeline_patterns.yaml，仅作参考基线，可自主裁剪/扩展）："]
    for name, spec in patterns.items():
        task_types = spec.get("task_types", [])
        description = spec.get("description", "")
        lines.append(f"- {name} (适用: {', '.join(task_types)}): {description}")
    return "\n".join(lines)


def _load_similar_cases_summary(task_type: str) -> str:
    """从knowledge_base/index.json检索相似任务类型的历史案例（若存在）。"""
    if not _KNOWLEDGE_INDEX_PATH.exists():
        return "knowledge_base/index.json 暂无历史案例记录。"
    try:
        index = json.loads(_KNOWLEDGE_INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "knowledge_base/index.json 解析失败，视为暂无历史案例。"

    matches = [
        entry
        for entry in index.get("entries", [])
        if task_type and task_type in entry.get("task_types", [])
    ]
    if not matches:
        return f"knowledge_base 中未找到与任务类型'{task_type}'匹配的历史案例。"
    lines = [f"knowledge_base 中找到{len(matches)}条与'{task_type}'匹配的历史案例："]
    for entry in matches:
        lines.append(f"- card_id={entry.get('card_id')}: {entry.get('summary', '')}")
    return "\n".join(lines)


class TrainingPlanGeneratorAgent(BaseAgent):
    agent_id = "training_plan"
    output_schema = TrainingPlanOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        task_type: str = kwargs.get("task_type", "")
        return (
            "你是一名训练计划编排专家。汇总Scenario-Analysis/Dataset-Analysis/"
            "Model-Selection三者的输出，产出完整的training_plan.md训练计划，包含："
            "训练引擎选择、GPU/CPU/存储资源规划、Batch Size、Epoch、学习率策略、优化器、"
            "精度、分阶段训练日历。\n\n"
            "训练流程策略：你需要明确回答'起点（基础模型 vs 中间checkpoint）+阶段序列'。"
            "决策依据优先级：(1)优先复用knowledge_base中相似任务的历史成功案例；"
            "(2)若无合适案例，调用WebSearch/WebFetch检索行业实践/论文补充依据；"
            "(3)综合任务类型、下方模式库、检索结果、历史案例，自主选择/裁剪/扩展出最终"
            "阶段序列，不局限于模式库中已有条目。pipeline_stages每阶段需标注：起点权重"
            "来源、训练目标、所用训练引擎、关键超参、预计耗时，并在decision_references"
            "中附上历史案例ID或检索URL/论文标题。\n\n"
            "数据格式定案：结合Dataset-Analysis的候选格式列表与Model-Selection的模型"
            "硬性格式要求，确定唯一最终目标格式（若冲突，以Model-Selection的硬性要求为"
            "准），产出完整字段映射规则(原始字段→目标格式字段)。\n\n"
            f"{_load_pipeline_patterns_summary()}\n\n"
            f"{_load_similar_cases_summary(task_type)}\n\n"
            "markdown字段必须是完整的training_plan.md文档，遵循以下结构：\n"
            "# <任务名> - 训练计划\n## TL;DR（人类速览）\n## 资源规划\n"
            "## Pipeline Stages（训练流程）\n## 数据格式\n## 训练日历（分阶段）\n"
            "## 验证与达标标准\n## 决策依据与引用来源"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        task_description: str = kwargs.get("task_description", "")
        scenario_output: str = kwargs.get("scenario_output", "")
        dataset_output: str = kwargs.get("dataset_output", "")
        model_selection_output: str = kwargs.get("model_selection_output", "")
        indicators: str = kwargs.get("indicators", "无特殊要求")
        resource_constraints: str = kwargs.get("resource_constraints", "无特殊约束")

        context = format_kv_block(
            "任务输入",
            {
                "训练任务描述": task_description,
                "指标要求": indicators,
                "资源/时间约束": resource_constraints,
                "Scenario-Analysis输出": scenario_output,
                "Dataset-Analysis输出": dataset_output,
                "Model-Selection输出": model_selection_output,
            },
        )
        return f"{context}\n\n请产出完整训练计划。\n\n{schema_instruction(self.output_schema)}"
