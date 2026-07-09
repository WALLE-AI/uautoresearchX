"""jsonrpc_transport.py 单元测试：用 os.pipe 模拟 stdio 管道。

覆盖：
- 请求/响应按id匹配
- 请求超时
- 通知按method分发
- stdout关闭(EOF)后pending请求异常结束（优雅关闭路径）
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from agents.engines.jsonrpc_transport import (
    JsonRpcError,
    JsonRpcTransport,
    TransportClosedError,
)


class PipePair:
    """创建一对 os.pipe，返回可读/可写的二进制文件对象。"""

    def __init__(self) -> None:
        read_fd, write_fd = os.pipe()
        self.reader = os.fdopen(read_fd, "rb", buffering=0)
        self.writer = os.fdopen(write_fd, "wb", buffering=0)

    def close(self) -> None:
        for fp in (self.reader, self.writer):
            try:
                fp.close()
            except OSError:
                pass


@pytest.fixture()
def transport_setup():
    """构造一个transport：client_to_server(模拟stdin) + server_to_client(模拟stdout)。"""
    client_to_server = PipePair()  # transport写入 -> 测试代码从reader读到"服务端收到的请求"
    server_to_client = PipePair()  # 测试代码写入writer -> transport作为stdout读取

    transport = JsonRpcTransport(stdin=client_to_server.writer, stdout=server_to_client.reader)
    transport.start()

    yield transport, client_to_server, server_to_client

    transport.stop()
    client_to_server.close()
    server_to_client.close()


def _read_one_json_line(fp) -> dict:
    line = fp.readline()
    return json.loads(line.decode("utf-8"))


def test_request_response_matching(transport_setup):
    transport, client_to_server, server_to_client = transport_setup

    result_holder: dict = {}

    def do_request():
        result_holder["result"] = transport.request(
            "initialize", {"foo": "bar"}, timeout=5
        )

    t = threading.Thread(target=do_request)
    t.start()

    sent = _read_one_json_line(client_to_server.reader)
    assert sent["method"] == "initialize"
    assert sent["params"] == {"foo": "bar"}
    assert sent["jsonrpc"] == "2.0"
    req_id = sent["id"]

    response = {"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}}
    server_to_client.writer.write((json.dumps(response) + "\n").encode("utf-8"))

    t.join(timeout=5)
    assert result_holder["result"] == {"ok": True}


def test_request_error_response_raises(transport_setup):
    transport, client_to_server, server_to_client = transport_setup

    error_holder: dict = {}

    def do_request():
        try:
            transport.request("bad_method", timeout=5)
        except JsonRpcError as exc:
            error_holder["exc"] = exc

    t = threading.Thread(target=do_request)
    t.start()

    sent = _read_one_json_line(client_to_server.reader)
    req_id = sent["id"]

    response = {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"},
    }
    server_to_client.writer.write((json.dumps(response) + "\n").encode("utf-8"))

    t.join(timeout=5)
    assert isinstance(error_holder.get("exc"), JsonRpcError)
    assert error_holder["exc"].code == -32601


def test_request_timeout(transport_setup):
    transport, _client_to_server, _server_to_client = transport_setup
    with pytest.raises(TimeoutError):
        transport.request("never_answered", timeout=0.3)


def test_notification_dispatch(transport_setup):
    transport, _client_to_server, server_to_client = transport_setup

    received: list[dict] = []
    event = threading.Event()

    def handler(params: dict) -> None:
        received.append(params)
        event.set()

    transport.on_notification("item/agentMessage/delta", handler)

    notification = {
        "jsonrpc": "2.0",
        "method": "item/agentMessage/delta",
        "params": {"text": "hello"},
    }
    server_to_client.writer.write((json.dumps(notification) + "\n").encode("utf-8"))

    assert event.wait(timeout=5)
    assert received == [{"text": "hello"}]


def test_eof_fails_pending_requests(transport_setup):
    transport, _client_to_server, server_to_client = transport_setup

    error_holder: dict = {}

    def do_request():
        try:
            transport.request("will_never_reply", timeout=5)
        except TransportClosedError as exc:
            error_holder["exc"] = exc

    t = threading.Thread(target=do_request)
    t.start()

    # 关闭服务端写端，模拟对端进程退出（EOF）
    server_to_client.writer.close()

    t.join(timeout=5)
    assert isinstance(error_holder.get("exc"), TransportClosedError)
