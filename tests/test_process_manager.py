"""process_manager.py 单元测试：用真实的轻量python子进程模拟目标CLI进程。

覆盖：
- 正常退出路径（进程自行退出，watchdog正确捕获退出码）
- 优雅关闭：关闭stdin后进程能在graceful_timeout内自行退出
- 超时SIGTERM路径：进程忽略stdin关闭，需SIGTERM才退出
- 异常退出后自动重启路径
"""

from __future__ import annotations

import sys
import time

from agents.engines.process_manager import ProcessManager, RestartPolicy


def test_normal_exit_is_captured():
    exit_codes: list[int | None] = []
    pm = ProcessManager(
        [sys.executable, "-c", "pass"],
        on_exit=exit_codes.append,
    )
    pm.start()
    # 进程立刻自行退出，等待watchdog捕获
    deadline = time.time() + 3
    while not exit_codes and time.time() < deadline:
        time.sleep(0.02)

    assert exit_codes == [0]
    assert pm.is_alive() is False


def test_graceful_stop_via_stdin_close():
    script = "import sys; sys.stdin.read(); print('bye')"
    pm = ProcessManager([sys.executable, "-c", script], graceful_timeout=3.0)
    pm.start()
    assert pm.is_alive() is True

    start = time.time()
    pm.stop()
    elapsed = time.time() - start

    assert pm.is_alive() is False
    # 应在远小于graceful_timeout的时间内通过stdin EOF正常退出
    assert elapsed < 3.0


def test_timeout_then_sigterm():
    script = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))\n"
        "while True: time.sleep(0.05)\n"
    )
    pm = ProcessManager([sys.executable, "-c", script], graceful_timeout=0.5)
    pm.start()
    assert pm.is_alive() is True

    start = time.time()
    pm.stop()
    elapsed = time.time() - start

    assert pm.is_alive() is False
    # stdin关闭不会让该进程退出，需等待graceful_timeout后SIGTERM才生效
    assert elapsed >= 0.5


def test_auto_restart_on_abnormal_exit():
    exit_codes: list[int | None] = []
    pm = ProcessManager(
        [sys.executable, "-c", "import sys; sys.exit(1)"],
        restart_policy=RestartPolicy(enabled=True, max_attempts=2, backoff_seconds=0.05),
        on_exit=exit_codes.append,
    )
    pm.start()

    deadline = time.time() + 5
    while len(exit_codes) < 3 and time.time() < deadline:
        time.sleep(0.05)

    # 初次退出 + 最多2次重启后再退出 = 3次on_exit回调
    assert len(exit_codes) == 3
    assert all(code == 1 for code in exit_codes)
