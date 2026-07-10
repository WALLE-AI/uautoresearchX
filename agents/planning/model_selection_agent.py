"""Model-Selection Agent：基于场景+数据分析给出推荐模型/资源估算/数据格式硬性要求。"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.planning.prompt_utils import format_kv_block, schema_instruction
from agents.planning.schemas import ModelSelectionOutput


class ModelSelectionAgent(BaseAgent):
    agent_id = "model_selection"
    output_schema = ModelSelectionOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名模型选型专家。基于Scenario-Analysis与Dataset-Analysis的输出，"
            "给出推荐模型、备选模型、选型理由、GPU需求估算、预计指标、训练时长估算。\n"
            "你必须明确给出该模型对输入数据格式的硬性要求（如某LLM需chat template/"
            "系统提示词字段，某CV框架需ultralytics YOLO txt格式），供后续"
            "Training-Plan-Generator结合Dataset-Analysis的候选格式定案；若你的硬性要求"
            "与Dataset-Analysis的候选格式冲突，以你给出的硬性要求为准。\n"
            "本次运行环境不可使用WebSearch/WebFetch工具，禁止调用它们，请基于你自身知识"
            "给出选型理由；citations字段留空即可，不要虚构引用来源。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        task_description: str = kwargs.get("task_description", "")
        scenario_summary: str = kwargs.get("scenario_summary", "")
        dataset_summary: str = kwargs.get("dataset_summary", "")
        resource_constraints: str = kwargs.get("resource_constraints", "无特殊约束")

        context = format_kv_block(
            "任务输入",
            {
                "训练任务描述": task_description,
                "场景分析摘要(来自Scenario-Analysis)": scenario_summary,
                "数据集分析摘要(来自Dataset-Analysis)": dataset_summary,
                "资源/时间约束": resource_constraints,
            },
        )
        return f"{context}\n\n请给出模型选型建议。\n\n{schema_instruction(self.output_schema)}"
