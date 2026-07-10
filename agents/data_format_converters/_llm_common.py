"""ShareGPT/Alpaca/OpenAI-Messages三种LLM对话格式互转的共享辅助函数。

设计范围：Dataset-Analysis Agent只会在这三种候选格式之间推荐（见
`configs/data_format_patterns.yaml`），因此sharegpt_converter/alpaca_converter
只需要能识别这三种输入形状并互转，不追求支持任意自定义LLM数据集结构。
"""

from __future__ import annotations

from typing import Any

from agents.data_format_converters.base_converter import UnsupportedRecordShapeError

_ROLE_TO_SHAREGPT_FROM = {"system": "system", "user": "human", "assistant": "gpt"}
_SHAREGPT_FROM_TO_ROLE = {"system": "system", "human": "user", "gpt": "assistant"}


def record_to_turns(record: dict[str, Any]) -> list[tuple[str, str]]:
    """把record（ShareGPT/OpenAI-Messages/Alpaca三种已知形状之一）统一转换为
    `(from, value)`元组列表，`from`取值为`system`/`human`/`gpt`。
    """
    if "conversations" in record:
        turns = []
        for turn in record["conversations"]:
            role = turn.get("from")
            if role not in ("system", "human", "gpt"):
                raise UnsupportedRecordShapeError(f"未知的conversations.from取值: {role!r}")
            turns.append((role, turn.get("value", "")))
        return turns

    if "messages" in record:
        turns = []
        for msg in record["messages"]:
            role = msg.get("role")
            mapped = _ROLE_TO_SHAREGPT_FROM.get(role)
            if mapped is None:
                raise UnsupportedRecordShapeError(f"未知的messages.role取值: {role!r}")
            turns.append((mapped, msg.get("content", "")))
        return turns

    if "instruction" in record and "output" in record:
        turns = []
        system = record.get("system")
        if system:
            turns.append(("system", system))
        human_text = record["instruction"]
        extra_input = record.get("input")
        if extra_input:
            human_text = f"{human_text}\n{extra_input}"
        turns.append(("human", human_text))
        turns.append(("gpt", record["output"]))
        return turns

    raise UnsupportedRecordShapeError(
        f"无法识别的LLM记录形状，record keys={sorted(record.keys())}"
    )


def turns_to_sharegpt(turns: list[tuple[str, str]]) -> dict[str, Any]:
    return {"conversations": [{"from": role, "value": text} for role, text in turns]}


def turns_to_messages(turns: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": _SHAREGPT_FROM_TO_ROLE[role], "content": text} for role, text in turns
        ]
    }


def turns_to_alpaca(turns: list[tuple[str, str]]) -> dict[str, Any]:
    system_text = ""
    human_text = ""
    gpt_text = ""
    for role, text in turns:
        if role == "system":
            system_text = text
        elif role == "human":
            human_text = text
        elif role == "gpt":
            gpt_text = text
    result: dict[str, Any] = {"instruction": human_text, "input": "", "output": gpt_text}
    if system_text:
        result["system"] = system_text
    return result
