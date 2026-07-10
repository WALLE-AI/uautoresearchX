"""Knowledge-Update Agent的pydantic输出schema定义。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.planning.schemas import PipelineStage


class KnowledgeCardOutput(BaseModel):
    """Knowledge-Update Agent产出：训练闭环结束后沉淀的经验卡片。"""

    task_summary: str = Field(description="任务描述与数据集特征摘要")
    dataset_stats_summary: str = Field(description="数据统计摘要")
    model_and_hyperparams_summary: str = Field(description="模型选型理由与最佳超参摘要")
    final_metrics_summary: str = Field(description="最终评估结果摘要")
    lessons_learned: list[str] = Field(
        default_factory=list, description="遇到的问题与解决方案（长尾处理/增强策略等经验总结）"
    )
    reused_pipeline_stages: list[PipelineStage] = Field(
        default_factory=list, description="最终采用的pipeline_stages流程，供未来相似任务直接复用"
    )
    task_types: list[str] = Field(
        default_factory=list,
        description="本次任务归属的任务类型标签列表（供knowledge_base/index.json按task_type检索）",
    )
