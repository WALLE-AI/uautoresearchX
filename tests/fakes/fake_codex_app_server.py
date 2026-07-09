"""模拟 `codex app-server` 的最小JSON-RPC协议，供codex_engine单元测试使用。

覆盖流程：initialize -> thread/start -> turn/start
(伴随 item/agentMessage/delta 通知 + turn/completed 通知)。
"""

from __future__ import annotations

import json
import sys


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    thread_id = "fake-thread-1"
    turn_id = "fake-turn-1"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method")
        req_id = req.get("id")

        if method == "initialize":
            send({"id": req_id, "result": {"userAgent": "fake/0.0"}})
        elif method == "thread/start":
            send({"id": req_id, "result": {"thread": {"id": thread_id}}})
        elif method == "turn/start":
            send({"id": req_id, "result": {"turn": {"id": turn_id, "status": "inProgress"}}})
            send(
                {
                    "method": "item/agentMessage/delta",
                    "params": {"delta": "1+1", "threadId": thread_id, "turnId": turn_id},
                }
            )
            send(
                {
                    "method": "item/agentMessage/delta",
                    "params": {"delta": "等于2", "threadId": thread_id, "turnId": turn_id},
                }
            )
            send(
                {
                    "method": "turn/completed",
                    "params": {"threadId": thread_id, "turn": {"id": turn_id, "status": "completed"}},
                }
            )
        elif method == "turn/interrupt":
            send({"id": req_id, "result": {}})


if __name__ == "__main__":
    main()
