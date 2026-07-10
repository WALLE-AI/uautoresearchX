"""data_format_converters单元测试：为LLM对话样例与CV标注样例构造最小输入，
验证各转换器输出文件内容符合`configs/data_format_patterns.yaml`描述的结构。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from agents.data_format_converters.alpaca_converter import AlpacaConverter
from agents.data_format_converters.coco_converter import COCOConverter
from agents.data_format_converters.mask_converter import MaskConverter
from agents.data_format_converters.sharegpt_converter import ShareGPTConverter
from agents.data_format_converters.yolo_converter import YOLOConverter
from agents.planning.schemas import FieldMapping

_ALPACA_RECORDS = [
    {"instruction": "翻译成英文", "input": "你好世界", "output": "Hello world"},
]
_MESSAGES_RECORDS = [
    {
        "messages": [
            {"role": "system", "content": "你是一个helpful assistant"},
            {"role": "user", "content": "介绍一下巴黎"},
            {"role": "assistant", "content": "巴黎是法国首都"},
        ]
    }
]

_DETECTION_RECORDS = [
    {
        "image": {"file_name": "0001.jpg", "width": 640, "height": 480},
        "objects": [
            {"category": "person", "bbox": [10, 20, 100, 150]},
            {"category": "car", "bbox": [200, 100, 80, 60]},
        ],
    }
]

_SEGMENT_RECORDS = [
    {
        "image": {"file_name": "0001.jpg", "width": 64, "height": 64},
        "objects": [
            {"category": "person", "polygon": [10, 10, 50, 10, 50, 50, 10, 50]},
        ],
    }
]


def test_sharegpt_converter_from_alpaca_shape(tmp_path: Path) -> None:
    dst = tmp_path / "sharegpt.json"
    result = ShareGPTConverter().convert(_ALPACA_RECORDS, [], dst)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data[0]["conversations"][0]["from"] == "human"
    assert "你好世界" in data[0]["conversations"][0]["value"]
    assert data[0]["conversations"][1] == {"from": "gpt", "value": "Hello world"}


def test_sharegpt_converter_from_messages_shape(tmp_path: Path) -> None:
    dst = tmp_path / "sharegpt.json"
    result = ShareGPTConverter().convert(_MESSAGES_RECORDS, [], dst)
    data = json.loads(result.read_text(encoding="utf-8"))
    roles = [turn["from"] for turn in data[0]["conversations"]]
    assert roles == ["system", "human", "gpt"]


def test_alpaca_converter_from_messages_shape(tmp_path: Path) -> None:
    dst = tmp_path / "alpaca.json"
    result = AlpacaConverter().convert(_MESSAGES_RECORDS, [], dst)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data[0]["instruction"] == "介绍一下巴黎"
    assert data[0]["output"] == "巴黎是法国首都"
    assert data[0]["system"] == "你是一个helpful assistant"


def test_field_mapping_renames_source_field_before_conversion(tmp_path: Path) -> None:
    records = [{"task": "翻译成英文", "answer": "Hello world"}]
    mapping = [
        FieldMapping(source_field="task", target_field="instruction", rule="rename"),
        FieldMapping(source_field="answer", target_field="output", rule="rename"),
    ]
    dst = tmp_path / "sharegpt.json"
    result = ShareGPTConverter().convert(records, mapping, dst)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data[0]["conversations"][-1]["value"] == "Hello world"


def test_coco_converter_produces_valid_structure(tmp_path: Path) -> None:
    dst = tmp_path / "coco.json"
    result = COCOConverter().convert(_DETECTION_RECORDS, [], dst)
    coco = json.loads(result.read_text(encoding="utf-8"))
    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "0001.jpg"
    assert len(coco["annotations"]) == 2
    assert {c["name"] for c in coco["categories"]} == {"person", "car"}
    for ann in coco["annotations"]:
        assert ann["image_id"] == 1
        assert ann["area"] > 0


def test_yolo_converter_normalizes_bbox(tmp_path: Path) -> None:
    dst_dir = tmp_path / "yolo_out"
    result = YOLOConverter().convert(_DETECTION_RECORDS, [], dst_dir)
    txt_file = result / "0001.txt"
    assert txt_file.exists()
    lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parts = lines[0].split()
    assert len(parts) == 5
    for value in parts[1:]:
        assert 0.0 <= float(value) <= 1.0
    classes = (result / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes == ["person", "car"]


def test_mask_converter_rasterizes_polygon(tmp_path: Path) -> None:
    dst_dir = tmp_path / "mask_out"
    result = MaskConverter().convert(_SEGMENT_RECORDS, [], dst_dir)
    mask_file = result / "0001_mask.png"
    assert mask_file.exists()
    img = Image.open(mask_file)
    assert img.mode == "L"
    assert img.size == (64, 64)
    pixel_inside = img.getpixel((30, 30))
    pixel_outside = img.getpixel((2, 2))
    assert pixel_inside == 1
    assert pixel_outside == 0
