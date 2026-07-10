"""转换目标格式：COCO 检测/分割 JSON。

输入`records`的期望形状（转换器职责边界内的通用简化标注结构，具体源格式
解析由调用方负责）：
    {
        "image": {"file_name": str, "width": int, "height": int},
        "objects": [
            {"category": str, "bbox": [x, y, w, h], "segmentation": [[x1,y1,...]]?},
            ...
        ],
    }
`bbox`为绝对像素坐标的`[x, y, w, h]`；`segmentation`可选，若提供则写入COCO-seg
的多边形字段，否则只产出检测标注（`segmentation`留空列表）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.data_format_converters.base_converter import (
    BaseDataFormatConverter,
    UnsupportedRecordShapeError,
    apply_field_mapping,
)
from agents.planning.schemas import FieldMapping


class COCOConverter(BaseDataFormatConverter):
    def convert(
        self,
        records: list[dict[str, Any]],
        mapping_rules: list[FieldMapping],
        dst_path: Path,
    ) -> Path:
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        category_name_to_id: dict[str, int] = {}

        next_image_id = 1
        next_annotation_id = 1

        for record in records:
            mapped = apply_field_mapping(record, mapping_rules)
            image_info = mapped.get("image")
            if not isinstance(image_info, dict) or "file_name" not in image_info:
                raise UnsupportedRecordShapeError(
                    f"记录缺少合法的image字段: {mapped!r}"
                )

            image_id = next_image_id
            next_image_id += 1
            images.append(
                {
                    "id": image_id,
                    "file_name": image_info["file_name"],
                    "width": image_info.get("width", 0),
                    "height": image_info.get("height", 0),
                }
            )

            for obj in mapped.get("objects", []):
                category = obj.get("category")
                if category is None:
                    raise UnsupportedRecordShapeError(f"标注对象缺少category字段: {obj!r}")
                if category not in category_name_to_id:
                    category_name_to_id[category] = len(category_name_to_id) + 1

                bbox = obj.get("bbox", [0, 0, 0, 0])
                area = bbox[2] * bbox[3] if len(bbox) == 4 else 0
                annotations.append(
                    {
                        "id": next_annotation_id,
                        "image_id": image_id,
                        "category_id": category_name_to_id[category],
                        "bbox": bbox,
                        "segmentation": obj.get("segmentation", []),
                        "area": area,
                        "iscrowd": obj.get("iscrowd", 0),
                    }
                )
                next_annotation_id += 1

        categories = [
            {"id": cat_id, "name": name} for name, cat_id in category_name_to_id.items()
        ]
        coco = {"images": images, "annotations": annotations, "categories": categories}

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        return dst_path
