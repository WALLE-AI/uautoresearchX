"""向后兼容入口：等价于打包后CLI的`run`子命令（`orchestrator/cli.py`）。

用法（与此前完全一致，不受打包升级影响）：
    uv run python -m orchestrator.run_pipeline \
        --task-description "构建一个客服问答机器人" \
        --dataset-path /path/to/dataset \
        --indicators "人工评估满意度>=85%" \
        --resource-constraints "8卡A100, 24小时内完成"

`uv pip install -e .`后建议改用正式打包的CLI命令：
    uautoresearchx run --task-description ... （功能等价）
    uautoresearchx resume/status/list/cancel/logs  （新增的运行管理子命令，
    本模块不提供，只作为`run`的兼容包装）

生产模式下`--interval-minutes`默认覆盖`configs/agents.yaml`中`monitor.
interval_minutes`（真实sleep等待，用于长时间训练场景）；本入口不提供"模拟/加速
轮询"选项——那属于测试专用配置（见`tests/test_state_machine.py`/
`tests/test_end_to_end_demo.py`里直接构造`StateMachineConfig`并覆盖
`poll_interval_seconds`/`max_polls_per_stage`/`run_script_resolver`的用法）。
"""

from __future__ import annotations

import sys

import typer

from orchestrator.cli import app


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    try:
        app(["run", *args], standalone_mode=False)
    except typer.Exit as exc:
        return exc.exit_code or 0
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
