"""Planning Agent共用的prompt构造辅助函数。"""

from __future__ import annotations

import json

from pydantic import BaseModel


def schema_instruction(schema: type[BaseModel]) -> str:
    """生成要求LLM严格按pydantic JSON Schema输出的指令文本。"""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "请严格按照以下JSON Schema输出，只输出合法JSON本身，不要用```代码块包裹，"
        "不要输出任何多余的解释性文字：\n"
        f"{schema_json}"
    )


def format_kv_block(title: str, data: dict[str, object]) -> str:
    """将dict渲染为"标题:\\n- key: value"形式的文本块，便于拼入user_prompt。"""
    lines = [f"{title}:"]
    for key, value in data.items():
        if value is None:
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
