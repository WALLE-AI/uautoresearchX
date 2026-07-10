"""通用同步"训练"脚本替身，供TrainerAgent测试/T8端到端demo通过
`run_script_resolver`注入，代替真实的`scripts/<engine>_run.sh`。

调用约定与真实脚本一致：
    python fake_train_script.py <run_dir> <config_path> <log_dir> <logger_type>

通过环境变量`FAKE_TRAIN_MODE`控制行为（默认"normal"）：
    normal      - 写入几行逐步下降的loss，正常退出(exit 0)
    early_stop  - loss很快趋于平稳不再下降（模拟早熟）
    diverge     - loss持续上升（模拟发散）
    crash       - 只写一行日志后以非0退出码终止（模拟崩溃）
写入的日志行采用`local_log_adapter.py`能识别的HuggingFace Trainer风格字典格式，
供log_adapters的解析逻辑与真实场景保持一致。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <run_dir> <config_path> <log_dir> <logger_type>", file=sys.stderr)
        return 1

    run_dir, config_path, log_dir, logger_type = sys.argv[1:5]
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    mode = os.environ.get("FAKE_TRAIN_MODE", "normal")
    log_file = Path(log_dir) / "train.log"

    if mode == "crash":
        with log_file.open("a", encoding="utf-8") as f:
            f.write("{'loss': 5.0, 'epoch': 0.1}\n")
            f.write("RuntimeError: CUDA out of memory\n")
        return 137

    if mode == "diverge":
        losses = [1.0, 1.5, 2.2, 3.1, 4.5]
    elif mode == "early_stop":
        losses = [1.0, 0.6, 0.59, 0.585, 0.583]
    else:
        losses = [2.0, 1.2, 0.7, 0.4, 0.2]

    with log_file.open("a", encoding="utf-8") as f:
        for i, loss in enumerate(losses, start=1):
            epoch = round(i / len(losses), 2)
            f.write(f"{{'loss': {loss}, 'epoch': {epoch}, 'train_samples_per_second': 10.0}}\n")
        checkpoint_dir = Path(run_dir) / "checkpoints" / f"checkpoint-{len(losses)}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        f.write(f"Saving model checkpoint to {checkpoint_dir}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
