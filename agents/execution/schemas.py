"""Execution阶段三个Agent（Trainer/Monitor/Evaluator）的pydantic输出schema定义。"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class StageConfigOutput(BaseModel):
    """Trainer Agent产出：单个pipeline stage的训练配置YAML文本。"""

    yaml_content: str = Field(description="生成的训练配置YAML文本内容，必须是合法的YAML mapping")

    @field_validator("yaml_content")
    @classmethod
    def _must_be_valid_yaml_mapping(cls, value: str) -> str:
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f"yaml_content不是合法YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("yaml_content解析后必须是一个字典（YAML mapping）")
        return value


class MonitorReportOutput(BaseModel):
    """Monitor Agent产出：单轮LLM全面分析结论。"""

    risk_level: Literal["Normal", "Warning", "Critical"] = Field(description="本轮风险等级")
    gpu_observation: str = Field(description="GPU利用率/显存/异常波动观察结论")
    loss_trend: str = Field(description="loss下降趋势观察结论")
    overfitting_signal: str = Field(description="早熟/过拟合迹象观察结论")
    validation_accuracy: str = Field(description="验证精度观察结论")
    other_anomalies: str = Field(default="", description="其他可观测异常")
    recommendation: str = Field(description="建议措施")
    crash_detected: bool = Field(
        default=False, description="是否从日志内容判断训练已实质性崩溃（区别于进程退出码硬信号）"
    )


class EvaluatorOutput(BaseModel):
    """Evaluator Agent产出：单轮评测判定结论。"""

    verdict: Literal["PASS", "FAIL"] = Field(description="本轮评测结论")
    gap_analysis: str = Field(description="与目标指标的差距分析")
    improvement_suggestions: list[str] = Field(default_factory=list, description="改进建议")
    needs_replanning: bool = Field(
        description="FAIL时是否判定为规划层问题（模型选型/资源不足），需回退到Planning重新规划；"
        "PASS时应为False"
    )
