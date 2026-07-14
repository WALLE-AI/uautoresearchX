"""agents/engines/json_extraction.py单元测试：从真实LLM响应里观察到的两类
"合法JSON但无法直接model_validate_json"场景（markdown代码块包裹 / JSON后面
附带对话式结尾文字）都应能被正确提取。
"""

from __future__ import annotations

import json

from agents.engines.json_extraction import extract_json_payload


def test_plain_json_passthrough() -> None:
    text = '{"a": 1, "b": [1, 2, 3]}'
    assert json.loads(extract_json_payload(text)) == {"a": 1, "b": [1, 2, 3]}


def test_markdown_json_fence_is_stripped() -> None:
    text = '\n\n```json\n{"recommended_model": "yolo11x", "citations": []}\n```\n'
    result = json.loads(extract_json_payload(text))
    assert result == {"recommended_model": "yolo11x", "citations": []}


def test_plain_fence_without_json_language_tag() -> None:
    text = '```\n{"x": 1}\n```'
    assert json.loads(extract_json_payload(text)) == {"x": 1}


def test_trailing_conversational_text_after_valid_json_is_dropped() -> None:
    text = '\n\n{\n  "markdown": "# plan"\n}\n\nWould you like me to do next?'
    assert json.loads(extract_json_payload(text)) == {"markdown": "# plan"}


def test_leading_conversational_text_before_json_is_dropped() -> None:
    text = "I'll analyze the task first.\n\n{\"task_type\": \"cv-detect\"}"
    assert json.loads(extract_json_payload(text)) == {"task_type": "cv-detect"}


def test_unrecoverable_text_returned_unchanged() -> None:
    text = "Sorry, I can't help with that right now."
    assert extract_json_payload(text) == text


def test_array_payload_supported() -> None:
    text = "```json\n[1, 2, 3]\n```"
    assert json.loads(extract_json_payload(text)) == [1, 2, 3]


def test_literal_control_characters_in_string_are_tolerated() -> None:
    # 真实观察到的失败模式：模型在多行markdown字段里直接写字面换行符，而不是
    # 转义成`\n`，标准json.loads(strict=True)/pydantic都会报
    # "control character found while parsing a string"。
    text = '{\n  "markdown": "# Title\nline two\nline three"\n}'
    result = json.loads(extract_json_payload(text))
    assert result == {"markdown": "# Title\nline two\nline three"}
