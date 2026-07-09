"""Planning阶段六个Agent的pydantic输出schema定义。

每个schema对应 `多智能体自动化训练框架架构设计v1.0.md` 中"阶段一：Planning
Agent"章节对各Agent的产出要求，作为各Agent `output_schema` 类属性，由
`BaseAgent.run()` 统一校验+失败重试。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScenarioAnalysisOutput(BaseModel):
    """Scenario-Analysis Agent产出：任务类型/行业/难度/风险等场景理解。"""

    task_type: str = Field(description="任务类型，如 llm-sft / cv-detect / cv-segment 等")
    industry: str = Field(description="应用行业/领域")
    difficulty: Literal["low", "medium", "high"] = Field(description="任务难度评估")
    latency_constraint: str = Field(description="推理速度约束描述，无特殊要求可填'无'")
    priority_suggestion: str = Field(description="优先级建议")
    risks: list[str] = Field(default_factory=list, description="潜在风险提示列表")
    citations: list[str] = Field(
        default_factory=list, description="WebSearch/WebFetch检索到的引用来源(URL/标题)"
    )


class DatasetAnalysisOutput(BaseModel):
    """Dataset-Analysis Agent产出：EDA统计 + 候选数据格式推荐。"""

    num_samples: int | None = Field(default=None, description="样本数，无法访问数据集时可为None")
    spec: str = Field(description="图像/文本规格描述，如分辨率/序列长度分布")
    num_classes: int | None = Field(default=None, description="类别数")
    class_distribution: dict[str, float] | None = Field(
        default=None, description="类别分布，key为类别名，value为占比或样本数"
    )
    quality_score: float = Field(ge=0, le=10, description="数据质量评分，0-10")
    augmentation_suggestions: list[str] = Field(default_factory=list, description="数据增强建议")
    candidate_formats: list[str] = Field(
        default_factory=list,
        description="候选目标数据格式列表，如 ['ShareGPT','Alpaca']，参考configs/data_format_patterns.yaml",
    )
    field_mapping_hints: list[str] = Field(
        default_factory=list, description="当前数据集字段与候选格式的初步字段映射建议"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="以上统计的置信度（若基于样例/描述估算而非直接访问数据集，应为medium/low）"
    )
    citations: list[str] = Field(default_factory=list, description="检索到的同类数据集/benchmark引用来源")


class ModelSelectionOutput(BaseModel):
    """Model-Selection Agent产出：推荐模型/选型理由/资源估算/数据格式硬性要求。"""

    recommended_model: str = Field(description="推荐模型名称")
    alternative_models: list[str] = Field(default_factory=list, description="备选模型列表")
    rationale: str = Field(description="选型理由")
    gpu_requirement: str = Field(description="GPU需求估算，如'8x A100-40GB'")
    estimated_metric: str = Field(description="预计达到的指标，如'mAP≈0.87'")
    estimated_training_duration: str = Field(description="预计训练时长估算")
    data_format_requirements: list[str] = Field(
        default_factory=list,
        description="模型对输入数据格式的硬性要求，如'需chat template系统提示词字段'",
    )
    citations: list[str] = Field(default_factory=list, description="检索到的相关论文/benchmark引用来源")


class PipelineStage(BaseModel):
    """训练流程中的单个阶段。"""

    name: str = Field(description="阶段名，如 SFT/DPO/GRPO/Warmup")
    start_from: str = Field(description="起点权重来源，如'基础模型'或'Stage1 checkpoint'")
    goal: str = Field(description="本阶段训练目标")
    engine: str = Field(description="所用训练引擎，如 llamafactory/trl/transformers/ultralytics/verl")
    key_hyperparams: str = Field(description="关键超参摘要，如'lr=2e-5, epoch=3'")
    estimated_duration: str = Field(description="预计耗时")


class FieldMapping(BaseModel):
    """数据格式字段映射规则中的单条映射。"""

    source_field: str = Field(description="原始字段名")
    target_field: str = Field(description="目标格式字段名")
    rule: str = Field(description="转换规则描述")


class DataFormatSpec(BaseModel):
    """训练计划中确定的最终数据格式与字段映射。"""

    target_format: str = Field(description="最终确定的目标格式，如 ShareGPT/COCO-seg/YOLO txt")
    rationale: str = Field(description="选择理由，来自Model-Selection硬性要求或Dataset-Analysis候选推荐依据")
    field_mapping: list[FieldMapping] = Field(default_factory=list)


class TrainingPlanOutput(BaseModel):
    """Training-Plan-Generator Agent产出：完整training_plan.md + 机器可解析的结构化字段。"""

    markdown: str = Field(description="完整的training_plan.md文档内容（含TL;DR/资源规划/Pipeline Stages等全部章节）")
    resource_plan: dict[str, str] = Field(
        default_factory=dict, description="资源规划键值对，如 {'GPU':'8x A100-40GB','Batch Size':'32'}"
    )
    pipeline_stages: list[PipelineStage] = Field(default_factory=list)
    data_format: DataFormatSpec
    decision_references: list[str] = Field(
        default_factory=list, description="决策依据引用（历史案例ID/检索URL/论文标题）"
    )


class ReviewIssue(BaseModel):
    """Plan Reviewer拒绝时列出的单条问题。"""

    category: Literal["计划参数", "选型判断", "数据理解"] = Field(description="问题类别，决定分层回退目标")
    description: str = Field(description="具体问题描述")


class PlanReviewOutput(BaseModel):
    """Plan Reviewer Agent产出：评审结论。"""

    approved: bool = Field(description="是否通过评审")
    issues: list[ReviewIssue] = Field(default_factory=list, description="未通过时的问题清单，逐条编号")
    summary: str = Field(description="评审结论摘要")


class AnalysisReportOutput(BaseModel):
    """ReportWriterAgent产出：汇总四份Planning Agent结构化输出的analysis_report.md。"""

    markdown: str = Field(description="完整的analysis_report.md文档内容（含四章节）")
