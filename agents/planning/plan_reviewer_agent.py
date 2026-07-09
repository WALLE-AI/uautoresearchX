"""Plan Reviewer Agent：评审training_plan.md，拒绝时列出问题清单并按根源分层回退。"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.planning.prompt_utils import format_kv_block, schema_instruction
from agents.planning.schemas import PlanReviewOutput

# 问题类别 -> 应回退的Planning Agent，按根源深度排序（数据理解最根本，计划参数最表层）。
_ROLLBACK_TARGET_BY_CATEGORY: dict[str, str] = {
    "数据理解": "dataset_analysis",
    "选型判断": "model_selection",
    "计划参数": "training_plan",
}
_CATEGORY_PRIORITY: list[str] = ["数据理解", "选型判断", "计划参数"]


def determine_rollback_target(issues: list[dict[str, Any]]) -> str | None:
    """给定PlanReviewOutput.issues（未通过时），按根源深度返回应回退到的agent_id。

    若issues中出现多个类别，优先回退到根源最深的一个（数据理解 > 选型判断 > 计划参数），
    因为修正上游问题后下游通常需要重新生成。issues为空(approved=True场景)返回None。
    """
    categories = {issue["category"] for issue in issues}
    for category in _CATEGORY_PRIORITY:
        if category in categories:
            return _ROLLBACK_TARGET_BY_CATEGORY[category]
    return None


class PlanReviewerAgent(BaseAgent):
    agent_id = "plan_reviewer"
    output_schema = PlanReviewOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名训练计划评审专家。审查training_plan.md是否满足以下要求：\n"
            "1. 资源可行性：GPU/显存是否足够支撑该计划的Batch Size/模型规模；\n"
            "2. 超参合理性：学习率/优化器/精度等设置是否合理；\n"
            "3. pipeline_stages设计是否匹配任务类型（阶段顺序/起点权重来源是否合理）；\n"
            "4. 数据格式是否满足模型硬性要求且字段映射完整；\n"
            "5. 是否有依据支撑（历史案例引用或检索来源），而非凭空给出；\n"
            "6. 能否达成用户目标指标的可行性。\n\n"
            "若存在任何问题，approved=false，并在issues中逐条列出，每条标注category："
            "'计划参数'(资源/超参/流程编排不当)、'选型判断'(模型选型错误导致的问题)、"
            "'数据理解'(数据集理解错误导致的问题)。若多个问题源于同一根本原因，只需按"
            "最根本的类别归类一次，避免重复。若全部通过，approved=true，issues为空列表。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        training_plan_markdown: str = kwargs.get("training_plan_markdown", "")
        available_resources: str = kwargs.get("available_resources", "8x NVIDIA A100-SXM4-40GB")
        indicators: str = kwargs.get("indicators", "无特殊要求")

        context = format_kv_block(
            "评审输入",
            {
                "当前可用资源": available_resources,
                "用户指标要求": indicators,
            },
        )
        return (
            f"{context}\n\n待评审的training_plan.md全文：\n---\n{training_plan_markdown}\n---\n\n"
            f"请给出评审结论。\n\n{schema_instruction(self.output_schema)}"
        )
