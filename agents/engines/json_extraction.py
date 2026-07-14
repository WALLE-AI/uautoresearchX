"""从LLM原始文本响应中尽力提取出JSON payload的共享工具，供三个engine
（`codex_engine.py`/`claude_engine.py`/`opencode_engine.py`）在
`output_schema.model_validate_json(text)`之前统一预处理。

`多智能体训练框架-架构审阅与修订计划-v1.1.md`已经点名过这个风险（"真实模型
输出偶尔带```json代码块包裹是已知风险"），本模块是该风险的具体修复：即使
system/user prompt已明确要求"只输出JSON"，真实模型仍然经常：
    1. 用markdown代码块包裹JSON（``` json ... ``` 或 ``` ... ```）；
    2. 在合法JSON前后附加对话式文字（"我先分析一下任务..."/"还需要我做什么
       吗？"），导致pydantic报"trailing characters"或"expected value"；
    3. 在多行字符串字段（如`markdown`长文档字段）里直接写字面换行符而不转义
       为`\\n`，导致pydantic报"control character found while parsing a
       string"——`json.JSONDecoder`默认`strict=True`同样会拒绝这种输入，需要
       显式用`strict=False`容忍字面控制字符后重新序列化，才能产出严格合法的
       JSON供pydantic校验。
"""

from __future__ import annotations

import json
import re

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_LENIENT_DECODER = json.JSONDecoder(strict=False)


def extract_json_payload(text: str) -> str:
    """尽力从`text`中提取出一段可被`json.loads`解析的JSON payload。

    提取失败（找不到任何JSON结构，或找到的片段本身就不是合法JSON）时原样
    返回输入文本，让调用方后续的pydantic校验照常抛出原始错误信息，不掩盖
    真正的格式问题（本函数只处理"JSON本身合法（或经`strict=False`容忍后可
    视为合法），但被包裹/附带了额外文字"这一类可安全恢复的情况）。
    """
    candidate = text.strip()

    fence_match = _FENCE_PATTERN.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    start_candidates = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if not start_candidates:
        return text
    start = min(start_candidates)
    candidate = candidate[start:]

    try:
        obj, _ = _LENIENT_DECODER.raw_decode(candidate)
    except json.JSONDecodeError:
        return text

    return json.dumps(obj, ensure_ascii=False)
