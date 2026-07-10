"""数据格式转换器统一接口：`convert(records, mapping_rules, dst_path) -> Path`。

`records`由调用方（Trainer Agent）已经从原始数据集文件中读出（JSON/JSONL/CSV
等源格式解析不属于转换器职责），转换器只负责按`training_plan.md`定案的字段
映射规则做结构重塑，并落盘为目标格式要求的文件。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agents.planning.schemas import FieldMapping


def apply_field_mapping(record: dict[str, Any], mapping_rules: list[FieldMapping]) -> dict[str, Any]:
    """按`mapping_rules`对`record`做浅层顶级字段重命名(source_field->target_field)。

    未在mapping_rules中出现的字段原样保留，转换器可继续按目标格式的规范字段名
    访问；重命名只作用于顶层key，不处理嵌套路径（嵌套结构的归一化由各转换器
    自行实现，因为不同目标格式的嵌套结构差异较大，无法用统一规则表达）。
    """
    if not mapping_rules:
        return dict(record)
    result = dict(record)
    for mapping in mapping_rules:
        if mapping.source_field in result and mapping.source_field != mapping.target_field:
            result[mapping.target_field] = result.pop(mapping.source_field)
    return result


class UnsupportedRecordShapeError(ValueError):
    """输入记录的结构不属于当前转换器能识别的已知形状时抛出。"""


class BaseDataFormatConverter(ABC):
    """所有数据格式转换器的统一接口。"""

    @abstractmethod
    def convert(
        self,
        records: list[dict[str, Any]],
        mapping_rules: list[FieldMapping],
        dst_path: Path,
    ) -> Path:
        """将`records`转换为本转换器对应的目标格式，写入`dst_path`并返回该路径。

        `dst_path`对单文件格式（ShareGPT/Alpaca/COCO/COCO-seg JSON）是目标文件
        路径；对多文件格式（YOLO txt/Mask PNG）是目标目录路径，返回值仍是
        `dst_path`本身，调用方通过遍历该目录获取生成的文件列表。
        """
        raise NotImplementedError
