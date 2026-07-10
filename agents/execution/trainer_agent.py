"""Trainer Agent：唯一可改config/超参的Execution Agent。

职责：
    1. `prepare_data()` —— 依据`training_plan.md`定案的`data_format`，调用
       对应`data_format_converters`把原始数据集转换到`runs/<run_id>/data/`。
    2. `build_stage_config()` —— 调用LLM把`pipeline_stages`中某一阶段的自由
       文本超参摘要（`key_hyperparams`）转成具体YAML配置内容，写入
       `runs/<run_id>/stage_<n>_<name>/config.yaml`。
    3. `launch_stage()` —— 后台启动`scripts/<engine>_run.sh`子进程（不阻塞），
       唯一负责调用训练脚本的入口。

`start_from_path`（某阶段的起点权重路径，基础模型或上一阶段checkpoint目录）
由调用方（`orchestrator/state_machine.py`）解析后传入，本Agent不解析
`PipelineStage.start_from`这段自由文本本身——那需要感知"上一阶段checkpoint
产出目录在哪"这类跨阶段状态，属于编排层职责，不属于单个Agent。
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from agents.base_agent import BaseAgent
from agents.data_format_converters.alpaca_converter import AlpacaConverter
from agents.data_format_converters.base_converter import BaseDataFormatConverter
from agents.data_format_converters.coco_converter import COCOConverter
from agents.data_format_converters.mask_converter import MaskConverter
from agents.data_format_converters.sharegpt_converter import ShareGPTConverter
from agents.data_format_converters.yolo_converter import YOLOConverter
from agents.execution.schemas import StageConfigOutput
from agents.planning.prompt_utils import format_kv_block, schema_instruction
from agents.planning.schemas import DataFormatSpec, PipelineStage

_TRAINING_ENGINES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "training_engines.yaml"
)

# 顺序很重要：判定要按更具体的模式优先（"coco-seg"须先于"coco"判断）。
_CONVERTER_BY_FORMAT_HINT: list[tuple[str, type[BaseDataFormatConverter], str]] = [
    ("sharegpt", ShareGPTConverter, "sharegpt.json"),
    ("alpaca", AlpacaConverter, "alpaca.json"),
    ("coco-seg", COCOConverter, "coco_seg.json"),
    ("coco", COCOConverter, "coco.json"),
    ("yolo", YOLOConverter, "yolo"),
    ("mask", MaskConverter, "masks"),
]


class TrainerAgentError(Exception):
    """TrainerAgent运行期错误（未注册引擎、未支持数据格式等）。"""


def _load_registered_engines() -> set[str]:
    raw = yaml.safe_load(_TRAINING_ENGINES_PATH.read_text(encoding="utf-8")) or {}
    return set(raw.keys())


def slugify_stage_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "stage"


def stage_dir(runs_root: Path, run_id: str, stage_index: int, stage_name: str) -> Path:
    return runs_root / run_id / f"stage_{stage_index}_{slugify_stage_name(stage_name)}"


def resolve_converter(target_format: str) -> tuple[BaseDataFormatConverter, str] | None:
    """按`target_format`自由文本模糊匹配已注册转换器，返回(转换器实例, 目标文件/目录名)。

    未命中任何已知格式时返回None，由调用方决定降级策略（如写原始JSON并标注
    需人工补充转换器），不在此处抛异常，因为"未知格式"是设计文档明确预期会
    出现的场景，不是错误。
    """
    lowered = target_format.lower()
    for hint, converter_cls, filename in _CONVERTER_BY_FORMAT_HINT:
        if hint in lowered:
            return converter_cls(), filename
    return None


def validate_engine_registered(engine_name: str) -> None:
    registered = _load_registered_engines()
    if engine_name not in registered:
        raise TrainerAgentError(
            f"pipeline stage指定的引擎'{engine_name}'未在configs/training_engines.yaml中注册"
            f"（已注册: {sorted(registered)}）"
        )


class TrainerAgent(BaseAgent):
    agent_id = "trainer"
    output_schema = StageConfigOutput

    def build_system_prompt(self, **kwargs: Any) -> str:
        return (
            "你是一名训练配置工程师。你的任务是把某一训练阶段的自由文本超参摘要"
            "转换成该训练引擎可直接使用的YAML配置文件内容。你必须只输出合法YAML"
            "文本本身（作为JSON字段的字符串值），不要遗漏关键字段：模型/起点权重"
            "路径、输出目录会由调用方另外注入，你只需给出学习率/batch"
            "size/epoch/优化器/精度等训练超参字段，以及该引擎所需的必要字段"
            "（如llamafactory需要`stage`/`finetuning_type`等，trl需要"
            "`trl_subcommand`，ultralytics需要`task`/`model`/`data`等）。"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        stage: PipelineStage = kwargs["stage"]
        resource_plan: dict[str, str] = kwargs.get("resource_plan", {})
        start_from_path: str = kwargs["start_from_path"]
        dataset_path: str = kwargs.get("dataset_path", "")

        context = format_kv_block(
            "阶段配置输入",
            {
                "阶段名": stage.name,
                "训练目标": stage.goal,
                "训练引擎": stage.engine,
                "关键超参摘要": stage.key_hyperparams,
                "起点权重路径": start_from_path,
                "数据集路径": dataset_path,
                "资源规划": resource_plan,
            },
        )
        return f"{context}\n\n请生成该阶段的config.yaml内容。\n\n{schema_instruction(self.output_schema)}"

    # ------------------------------------------------------------------
    # 编排相关方法（非LLM调用）
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        data_format: DataFormatSpec,
        records: list[dict[str, Any]],
        run_id: str,
        runs_root: Path = Path("runs"),
    ) -> Path:
        """按`data_format`定案结果转换原始数据集，输出到`runs/<run_id>/data/`。"""
        data_dir = runs_root / run_id / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        resolved = resolve_converter(data_format.target_format)
        if resolved is None:
            fallback_path = data_dir / "raw_unconverted.json"
            fallback_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return fallback_path

        converter, filename = resolved
        return converter.convert(records, data_format.field_mapping, data_dir / filename)

    def build_stage_config(
        self,
        stage: PipelineStage,
        resource_plan: dict[str, str],
        start_from_path: str,
        run_id: str,
        stage_index: int,
        dataset_path: str = "",
        runs_root: Path = Path("runs"),
    ) -> Path:
        """调用LLM生成本阶段config.yaml并落盘，返回config.yaml路径。"""
        validate_engine_registered(stage.engine)

        result = self.run(
            stage=stage,
            resource_plan=resource_plan,
            start_from_path=start_from_path,
            dataset_path=dataset_path,
        )
        assert result.structured_output is not None

        config_dir = stage_dir(runs_root, run_id, stage_index, stage.name)
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(result.structured_output["yaml_content"], encoding="utf-8")
        return config_path

    def launch_stage(
        self,
        stage: PipelineStage,
        config_path: Path,
        run_id: str,
        stage_index: int,
        logger_type: str,
        runs_root: Path = Path("runs"),
        logs_root: Path = Path("logs"),
        run_script_resolver: Callable[[str], Sequence[str]] | None = None,
    ) -> subprocess.Popen:
        """后台启动`scripts/<engine>_run.sh`（或测试注入的替身脚本），不阻塞等待。

        训练脚本自身负责把stdout/日志写入`log_dir`（见`scripts/*_run.sh`的
        `tee`逻辑），因此这里不对子进程stdout/stderr设置PIPE——避免重蹈
        Engine桥接层里"stderr未被drain导致管道死锁"的覆辙，直接重定向到
        DEVNULL即可，训练进度统一通过log_adapters读取落盘的日志文件。
        """
        resolver = run_script_resolver or (lambda engine: ["bash", f"scripts/{engine}_run.sh"])

        run_dir = stage_dir(runs_root, run_id, stage_index, stage.name)
        log_dir = logs_root / run_id / logger_type
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        command = [*resolver(stage.engine), str(run_dir), str(config_path), str(log_dir), logger_type]
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
