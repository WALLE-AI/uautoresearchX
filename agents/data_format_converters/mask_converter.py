"""转换目标格式：语义分割PNG mask（单通道，像素值=class_id，0为背景）。

输入`records`形状：
    {
        "image": {"file_name": str, "width": int, "height": int},
        "objects": [{"category": str, "polygon": [x1, y1, x2, y2, ...]}, ...],
    }
`polygon`为绝对像素坐标的扁平点列表（COCO-seg风格的单个多边形，若一个物体有
多个多边形需拆成多条`objects`记录）。多个物体的多边形在同一张mask上按
`objects`列表顺序依次栅格化（后绘制的覆盖先绘制的重叠区域）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from agents.data_format_converters.base_converter import (
    BaseDataFormatConverter,
    UnsupportedRecordShapeError,
    apply_field_mapping,
)
from agents.planning.schemas import FieldMapping


class MaskConverter(BaseDataFormatConverter):
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
                    f"mask格式转换需要image.width/height，记录: {mapped!r}"
                )

            mask = Image.new("L", (width, height), color=0)
            draw = ImageDraw.Draw(mask)

            for obj in mapped.get("objects", []):
                category = obj.get("category")
                if category is None:
                    raise UnsupportedRecordShapeError(f"标注对象缺少category字段: {obj!r}")
                if category not in category_name_to_id:
                    category_name_to_id[category] = len(category_name_to_id) + 1
                class_id = category_name_to_id[category]

                polygon = obj.get("polygon", [])
                if len(polygon) < 6:
                    continue
                points = list(zip(polygon[0::2], polygon[1::2]))
                draw.polygon(points, fill=class_id)

            stem = Path(image_info["file_name"]).stem
            mask.save(dst_path / f"{stem}_mask.png")

        classes_file = dst_path / "classes.txt"
        classes_file.write_text(
            "\n".join(f"{cid}:{name}" for name, cid in category_name_to_id.items()),
            encoding="utf-8",
        )
        return dst_path
