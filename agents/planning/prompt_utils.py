"""Planning Agent共用的prompt构造辅助函数。"""

from __future__ import annotations

import json

from pydantic import BaseModel


def schema_instruction(schema: type[BaseModel]) -> str:
    """生成要求LLM严格按pydantic JSON Schema输出的指令文本。

    真实调用中观察到部分模型（尤其是偏对话/agentic倾向的模型）即使已经要求
    "只输出JSON"，仍然会：(1)用```json代码块包裹；(2)在JSON前后附加解释或
    反问式结尾（如"还需要我做什么吗？"）；(3)完全不输出JSON，只给一段说明
    接下来要做什么的对话文字。前两种由`agents/engines/json_extraction.py`
    做后处理兜底，第三种无法靠后处理恢复，只能靠更强的指令约束——因此这里
    在指令开头与结尾各强调一次（LLM对prompt末尾的指令通常更敏感），并显式
    禁止提出澄清性问题（本框架是全自动流程，没有人类会回复）。
    """
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "【输出格式要求，必须严格遵守】\n"
        "你的回复会被程序直接解析，不会有任何人类阅读或回复你——因此：\n"
        "1. 只输出一个合法JSON对象/数组本身，不要输出任何其他文字（不要有前言、"
        "解释、总结、markdown标题，也不要用```代码块包裹）。\n"
        "2. 不要提出任何澄清性问题或反问——这是全自动流程，没有人会回答你，必须"
        "基于已给信息直接给出最合理的结论。\n"
        "3. 严格按以下JSON Schema的字段结构输出：\n"
        f"{schema_json}\n"
        "再次强调：你的整个回复必须能被json.loads()直接解析——第一个字符必须是"
        "'{'或'['，最后一个字符必须是对应的'}'或']'，中间不能有任何非JSON内容，"
        "结尾也不能有任何问句或说明文字。"
    )


def format_kv_block(title: str, data: dict[str, object]) -> str:
    """将dict渲染为"标题:\\n- key: value"形式的文本块，便于拼入user_prompt。"""
    lines = [f"{title}:"]
    for key, value in data.items():
        if value is None:
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)
