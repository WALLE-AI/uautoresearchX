"""模拟 `opencode acp` 的最小ACP协议，供opencode_engine单元测试使用。

覆盖流程：initialize -> session/new -> session/prompt
(伴随 session/update 通知，含 agent_message_chunk)。
"""

from __future__ import annotations

import json
import sys


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    session_id = "fake-session-1"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method")
        req_id = req.get("id")

        if method == "initialize":
            send({"id": req_id, "result": {"protocolVersion": 1, "agentCapabilities": {}}})
        elif method == "session/new":
            send({"id": req_id, "result": {"sessionId": session_id}})
        elif method == "session/prompt":
            send(
                {
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "1+1"},
                        },
                    },
                }
            )
            send(
                {
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "等于2"},
                        },
                    },
                }
            )
            send(
                {
                    "id": req_id,
                    "result": {"stopReason": "end_turn", "usage": {"totalTokens": 10}},
                }
            )
        elif method == "session/cancel":
            send({"id": req_id, "result": {}})


if __name__ == "__main__":
    main()
