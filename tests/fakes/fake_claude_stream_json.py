"""模拟 `claude -p --input-format stream-json --output-format stream-json` 的
最小NDJSON协议，供claude_engine单元测试使用。

覆盖流程：启动时发system/init -> 每收到一条用户输入行，回复
stream_event(content_block_delta) 增量 + result(success)。
"""

from __future__ import annotations

import json
import sys


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    send({"type": "system", "subtype": "init", "session_id": "fake-session-1", "model": "fake-model"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        json.loads(line)  # 校验输入是合法JSON（当前fake不关心具体字段）

        send(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "1+1"},
                },
            }
        )
        send(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "等于2"},
                },
            }
        )
        send({"type": "result", "subtype": "success", "result": "1+1等于2"})


if __name__ == "__main__":
    main()
