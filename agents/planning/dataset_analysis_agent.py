"""Dataset-Analysis Agent：EDA统计 + 候选数据格式推荐，必须检索同类数据集/benchmark。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agents.base_agent import BaseAgent
from agents.planning.prompt_utils import format_kv_block, schema_instruction
from agents.planning.schemas import DatasetAnalysisOutput

_DATA_FORMAT_PATTERNS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "data_format_patterns.yaml"
)


def _load_data_format_summary() -> str:
    """加载configs/data_format_patterns.yaml，渲染为简要的格式清单文本供prompt引用。"""
    patterns = yaml.safe_load(_DATA_FORMAT_PATTERNS_PATH.read_text(encoding="utf-8")) or {}
    lines = ["可选数据格式规范清单（configs/data_format_patterns.yaml）："]
    for name, spec in patterns.items():
        category = spec.get("category", "")
        typical_use = spec.get("typical_use", "")
        lines.append(f"- {name} (category={category}): {typical_use}")
    return "\n".join(lines)


class DatasetAnalysisAgent(BaseAgent):
    agent_id = "dataset_analysis"
    output_schema = DatasetAnalysisOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名数据集分析专家。你的任务是对用户提供的数据集路径/样例进行EDA分析："
            "样本数、图像/文本规格、类别数、类别分布、质量评分、数据增强建议。若数据集不可"
            "直接访问，基于任务描述与样例估算并将confidence标注为medium或low。\n"
            "本次运行环境不可使用WebSearch/WebFetch工具，禁止调用它们。请完全基于你自身的"
            "知识，对比同类开源数据集/已知benchmark指标给出分析，并提出该任务类型适用的数据"
            "处理/标注规范/数据增强最佳实践建议；citations字段留空即可，不要虚构引用来源。\n\n"
            "此外，你需要仅基于任务类型（尚不知模型选型）提出候选目标数据格式："
            "LLM任务→ShareGPT/Alpaca/OpenAI messages；CV检测→COCO/YOLO txt/VOC XML；"
            "CV分割→COCO-seg多边形/PNG mask。最终格式将由后续Training-Plan-Generator结合"
            "Model-Selection的硬性要求定案，你只需给出候选列表与初步字段映射建议。\n\n"
            f"{_load_data_format_summary()}"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        task_description: str = kwargs.get("task_description", "")
        dataset_path: str = kwargs.get("dataset_path", "未提供，请基于描述估算")
        dataset_sample: str = kwargs.get("dataset_sample", "")
        scenario_summary: str = kwargs.get("scenario_summary", "")

        context = format_kv_block(
            "任务输入",
            {
                "训练任务描述": task_description,
                "数据集路径": dataset_path,
                "数据集样例": dataset_sample,
                "场景分析摘要(来自Scenario-Analysis)": scenario_summary,
            },
        )
        return f"{context}\n\n请分析该数据集。\n\n{schema_instruction(self.output_schema)}"
