"""ReportWriterAgent：汇总四个Planning Agent结构化输出，生成analysis_report.md。"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.planning.prompt_utils import format_kv_block, schema_instruction
from agents.planning.schemas import AnalysisReportOutput


class ReportWriterAgent(BaseAgent):
    agent_id = "report_writer"
    output_schema = AnalysisReportOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名技术报告撰写专家。将Scenario-Analysis/Dataset-Analysis/"
            "Model-Selection/Training-Plan-Generator四个Agent的结构化输出，汇总为一份"
            "分章节的analysis_report.md技术报告，包含：\n"
            "第一章 场景需求深度分析（含检索到的行业/同类任务最佳实践引用）\n"
            "第二章 数据集EDA分析（类别分布、质量问题、增强策略依据）\n"
            "第三章 模型选型依据（含检索到的相关论文/benchmark对比、选型理由引用来源）\n"
            "第四章 训练计划总览（资源/日历/超参一览）\n"
            "各章节需保留原Agent输出中的citations引用来源(URL/论文标题)，不要遗漏。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        scenario_output: str = kwargs.get("scenario_output", "")
        dataset_output: str = kwargs.get("dataset_output", "")
        model_selection_output: str = kwargs.get("model_selection_output", "")
        training_plan_output: str = kwargs.get("training_plan_output", "")

        context = format_kv_block(
            "四份Planning Agent结构化输出",
            {
                "Scenario-Analysis": scenario_output,
                "Dataset-Analysis": dataset_output,
                "Model-Selection": model_selection_output,
                "Training-Plan-Generator": training_plan_output,
            },
        )
        return (
            f"{context}\n\n请汇总生成analysis_report.md。\n\n{schema_instruction(self.output_schema)}"
        )
