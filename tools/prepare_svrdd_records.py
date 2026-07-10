"""一次性测试用脚本：把 SVRDD_YOLO 原始数据集转成框架 TrainerAgent.prepare_data()
期望的通用中间记录格式（dataset_records），供 --dataset-records-file 使用。

用法：
    uv run python tools/prepare_svrdd_records.py

产出：
    tools/svrdd_dataset_records.json  —— list[{"image": {...}, "objects": [...]}]
    tools/svrdd_dataset_sample.txt    —— 供 --dataset-sample 的文本摘要

本次测试用户已确认使用完整数据集（train+val，不抽样）。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

DATASET_ROOT = Path("/home/dataset1/gaojing/xibeiyuan/datasets/SVRDD/SVRDD_YOLO")
OUT_DIR = Path(__file__).resolve().parent

# 类别 ID -> 名称映射，来自 readme.md（类别编号列，英文名用于record.objects.category）
CLASSES = [
    "Longitudinal Crack",  # 0 纵向裂缝
    "Transverse Crack",  # 1 横向裂缝
    "Alligator Crack",  # 2 龟裂
    "Pothole",  # 3 坑槽
    "Longitudinal Patch",  # 4 纵向修补
    "Transverse Patch",  # 5 横向修补
    "Manhole Cover",  # 6 井盖
]

# readme.md 里的实例数量统计（用于 dataset-sample 摘要，不参与转换逻辑）
README_INSTANCE_COUNTS = {
    "Longitudinal Crack": 4665,
    "Longitudinal Patch": 4128,
    "Transverse Crack": 3404,
    "Manhole Cover": 3339,
    "Transverse Patch": 2622,
    "Alligator Crack": 1728,
    "Pothole": 918,
}


def load_split(split_file: str) -> list[str]:
    return (DATASET_ROOT / split_file).read_text(encoding="utf-8").splitlines()


def label_path_for(image_path: Path) -> Path:
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def build_records(image_paths: list[str], split_name: str) -> list[dict]:
    records = []
    for i, raw_path in enumerate(image_paths):
        image_path = Path(raw_path)
        label_path = label_path_for(image_path)

        with Image.open(image_path) as im:
            width, height = im.size

        objects = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                class_id, xc, yc, w, h = line.split()
                class_id = int(class_id)
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                abs_w = w * width
                abs_h = h * height
                abs_x = xc * width - abs_w / 2
                abs_y = yc * height - abs_h / 2
                objects.append(
                    {
                        "category": CLASSES[class_id],
                        "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                    }
                )

        records.append(
            {
                "image": {
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                },
                "objects": objects,
                "split": split_name,
            }
        )

        if (i + 1) % 1000 == 0:
            print(f"[{split_name}] {i + 1}/{len(image_paths)} 完成")

    return records


def main() -> None:
    train_paths = load_split("train_abs.txt")
    val_paths = load_split("val_abs.txt")

    print(f"train: {len(train_paths)} 张, val: {len(val_paths)} 张")

    records = build_records(train_paths, "train") + build_records(val_paths, "val")

    out_json = OUT_DIR / "svrdd_dataset_records.json"
    out_json.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"写入 {out_json}（{len(records)} 条记录）")

    total_objects = sum(len(r["objects"]) for r in records)
    class_list_str = ", ".join(f"{c}({README_INSTANCE_COUNTS.get(c, '?')})" for c in CLASSES)

    sample_lines = (DATASET_ROOT / "Dongcheng" / "labels").glob("*.txt")
    example_label_path = next(iter(sample_lines))
    example_lines = example_label_path.read_text(encoding="utf-8").splitlines()[:2]

    summary = f"""SVRDD_YOLO 道路病害检测数据集（北京五区: Fengtai/Chaoyang/Xicheng/Dongcheng/Haidian），
原始格式为 Ultralytics YOLO txt（每张图一个txt，每行 `class_id x_center y_center w h` 归一化坐标），图片均为 1024x1024 RGB。
本次使用完整数据集，不抽样：train {len(train_paths)} 张 / val {len(val_paths)} 张，共 {len(records)} 张，标注对象总数 {total_objects} 个。
共 7 个类别（名称及 readme.md 中的实例数量统计，供参考，不是本次转换出的精确统计）: {class_list_str}。
原始标注行示例（来自 {example_label_path.name}）:
{chr(10).join(example_lines)}
"""
    out_txt = OUT_DIR / "svrdd_dataset_sample.txt"
    out_txt.write_text(summary, encoding="utf-8")
    print(f"写入 {out_txt}")
    print("\n=== dataset-sample 内容预览 ===")
    print(summary)


if __name__ == "__main__":
    main()
