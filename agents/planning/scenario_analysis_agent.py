"""Scenario-Analysis Agent：任务类型/行业/难度/风险场景理解，必须检索最佳实践。"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.planning.prompt_utils import format_kv_block, schema_instruction
from agents.planning.schemas import ScenarioAnalysisOutput


class ScenarioAnalysisAgent(BaseAgent):
    agent_id = "scenario_analysis"
    output_schema = ScenarioAnalysisOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名资深AI训练场景分析专家。你的任务是根据用户提供的训练任务描述，"
            "分析其任务类型、所属行业、难度、推理速度约束、优先级建议与潜在风险。\n"
            "你必须调用WebSearch/WebFetch工具，检索同行业/同类任务的解决方案、论文、"
            "最佳实践，用以支撑你的分析结论，并在输出的citations字段中列出引用来源"
            "(URL或论文标题)。禁止在没有检索的情况下凭空给出行业最佳实践结论。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        task_description: str = kwargs.get("task_description", "")
        indicators: str = kwargs.get("indicators", "无特殊要求")
        resource_constraints: str = kwargs.get("resource_constraints", "无特殊约束")

        context = format_kv_block(
            "任务输入",
            {
                "训练任务描述": task_description,
                "指标要求": indicators,
                "资源/时间约束": resource_constraints,
            },
        )
        return f"{context}\n\n请分析该训练任务的场景特征。\n\n{schema_instruction(self.output_schema)}"
