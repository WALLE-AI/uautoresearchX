"""转换目标格式：Ultralytics YOLO txt（每张图一个`.txt`，每行
`<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>`）。

输入`records`形状与`coco_converter.py`一致：
    {
        "image": {"file_name": str, "width": int, "height": int},
        "objects": [{"category": str, "bbox": [x, y, w, h]}, ...],
    }
`bbox`为绝对像素坐标`[x, y, w, h]`（左上角+宽高），本转换器负责归一化为
`[0,1]`区间的中心点坐标+宽高。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.data_format_converters.base_converter import (
    BaseDataFormatConverter,
    UnsupportedRecordShapeError,
    apply_field_mapping,
)
from agents.planning.schemas import FieldMapping


class YOLOConverter(BaseDataFormatConverter):
    def convert(
        self,
        records: list[dict[str, Any]],
        mapping_rules: list[FieldMapping],
        dst_path: Path,
    ) -> Path:
        dst_path.mkdir(parents=True, exist_ok=True)
        category_name_to_id: dict[str, int] = {}

        for record in records:
            mapped = apply_field_mapping(record, mapping_rules)
            image_info = mapped.get("image")
            if not isinstance(image_info, dict) or "file_name" not in image_info:
                raise UnsupportedRecordShapeError(f"记录缺少合法的image字段: {mapped!r}")

            width = image_info.get("width")
            height = image_info.get("height")
            if not width or not height:
                raise UnsupportedRecordShapeError(
                    f"YOLO格式转换需要image.width/height，记录: {mapped!r}"
                )

            lines: list[str] = []
            for obj in mapped.get("objects", []):
                category = obj.get("category")
                if category is None:
                    raise UnsupportedRecordShapeError(f"标注对象缺少category字段: {obj!r}")
                if category not in category_name_to_id:
                    category_name_to_id[category] = len(category_name_to_id)
                class_id = category_name_to_id[category]

                x, y, w, h = obj.get("bbox", [0, 0, 0, 0])
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

            stem = Path(image_info["file_name"]).stem
            (dst_path / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        classes_file = dst_path / "classes.txt"
        classes_file.write_text(
            "\n".join(name for name, _ in sorted(category_name_to_id.items(), key=lambda kv: kv[1])),
            encoding="utf-8",
        )
        return dst_path
