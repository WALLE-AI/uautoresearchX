"""Planning Agent产出文件的读写辅助函数（training_plan.md/plan_review_log.md等）。"""

from __future__ import annotations

from pathlib import Path


def write_markdown(path: Path, content: str) -> None:
    """覆盖写入markdown文件，自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_markdown_log(path: Path, section: str) -> None:
    """向markdown日志文件追加一节内容（如plan_review_log.md逐次评审记录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(section)
        if not section.endswith("\n"):
            f.write("\n")
        f.write("\n")
