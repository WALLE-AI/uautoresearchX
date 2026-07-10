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
            "本次运行环境不可使用WebSearch/WebFetch工具，禁止调用它们。请完全基于你自身的"
            "知识给出同行业/同类任务的解决方案、最佳实践分析；citations字段留空即可，"
            "不要虚构引用来源。"
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
