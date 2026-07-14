"""从已生成的完整 `tools/svrdd_dataset_records.json`（7000条，见
`tools/prepare_svrdd_records.py`）里抽样一个小子集，供CLI命令功能验收测试
使用——目的是快速跑完一次真实（非fake）训练闭环来验证`uautoresearchx`各
子命令是否可用，不是为了产出高精度模型，因此不需要全量数据。

用法：
    uv run python tools/sample_svrdd_records.py [--train N] [--val M]

产出：
    tools/svrdd_dataset_records_sample.json
    tools/svrdd_dataset_sample_summary.txt   —— 供 --dataset-sample 使用
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
FULL_RECORDS_PATH = OUT_DIR / "svrdd_dataset_records.json"

CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
    "Longitudinal Patch",
    "Transverse Patch",
    "Manhole Cover",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="从SVRDD完整records里抽样一个小子集")
    parser.add_argument("--train", type=int, default=80, help="抽样train张数")
    parser.add_argument("--val", type=int, default=20, help="抽样val张数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not FULL_RECORDS_PATH.exists():
        raise SystemExit(
            f"{FULL_RECORDS_PATH} 不存在，请先运行 "
            "`uv run python tools/prepare_svrdd_records.py` 生成完整records"
        )

    all_records = json.loads(FULL_RECORDS_PATH.read_text(encoding="utf-8"))
    train_records = [r for r in all_records if r.get("split") == "train"]
    val_records = [r for r in all_records if r.get("split") == "val"]

    rng = random.Random(args.seed)
    sampled_train = rng.sample(train_records, min(args.train, len(train_records)))
    sampled_val = rng.sample(val_records, min(args.val, len(val_records)))
    sample = sampled_train + sampled_val

    out_json = OUT_DIR / "svrdd_dataset_records_sample.json"
    out_json.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    total_objects = sum(len(r["objects"]) for r in sample)
    class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for record in sample:
        for obj in record["objects"]:
            class_counts[obj["category"]] = class_counts.get(obj["category"], 0) + 1
    class_list_str = ", ".join(f"{name}({count})" for name, count in class_counts.items())

    summary = f"""SVRDD_YOLO 道路路面病害检测数据集（北京五区: Fengtai/Chaoyang/Xicheng/Dongcheng/Haidian）的
小规模抽样子集（供CLI命令功能验收测试快速跑通训练闭环，非最终精度评估）：
train {len(sampled_train)} 张 / val {len(sampled_val)} 张，共 {len(sample)} 张，标注对象总数 {total_objects} 个。
原始格式为Ultralytics YOLO txt（每张图一个txt，每行 `class_id x_center y_center w h` 归一化坐标），图片均为 1024x1024 RGB。
共7个类别（本次抽样子集中的实际实例数）: {class_list_str}。
完整数据集（未抽样）共7000张/18232个标注对象，位于 {FULL_RECORDS_PATH.parent.parent}/
的 SVRDD_YOLO 目录，本次仅用于CLI功能验证，后续若要提升精度应换回全量数据集
（重新运行 `tools/prepare_svrdd_records.py` 不加抽样）并增加训练轮次。
"""
    out_txt = OUT_DIR / "svrdd_dataset_sample_summary.txt"
    out_txt.write_text(summary, encoding="utf-8")

    print(f"写入 {out_json}（{len(sample)}条记录）")
    print(f"写入 {out_txt}")
    print("\n=== dataset-sample 内容预览 ===")
    print(summary)


if __name__ == "__main__":
    main()
