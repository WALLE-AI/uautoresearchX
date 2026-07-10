"""CLI入口：从用户输入到闭环执行。

用法：
    uv run python -m orchestrator.run_pipeline \
        --task-description "构建一个客服问答机器人" \
        --dataset-path /path/to/dataset \
        --indicators "人工评估满意度>=85%" \
        --resource-constraints "8卡A100, 24小时内完成"

生产模式下`--interval-minutes`默认覆盖`configs/agents.yaml`中`monitor.
interval_minutes`（真实sleep等待，用于长时间训练场景）；本CLI不提供"模拟/加速
轮询"选项——那属于测试专用配置（见`tests/test_state_machine.py`/
`tests/test_end_to_end_demo.py`里直接构造`StateMachineConfig`并覆盖
`poll_interval_seconds`/`max_polls_per_stage`/`run_script_resolver`的用法）。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from orchestrator.state_machine import PipelineState, RunContext, StateMachine, StateMachineConfig


def generate_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:8]}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多智能体自动化训练框架 CLI 入口")
    parser.add_argument("--task-description", required=True, help="训练任务描述")
    parser.add_argument("--dataset-path", default="未提供，请基于描述估算")
    parser.add_argument("--dataset-sample", default="")
    parser.add_argument(
        "--dataset-records-file",
        default=None,
        help="JSON文件路径，内容为list[dict]，供Trainer Agent的data_format_converters转换使用",
    )
    parser.add_argument("--indicators", default="无特殊要求")
    parser.add_argument("--resource-constraints", default="无特殊约束")
    parser.add_argument("--available-resources", default="8x NVIDIA A100-SXM4-40GB")
    parser.add_argument("--run-id", default=None, help="不指定则自动生成")
    parser.add_argument("--logger-type", default="local", choices=["local", "wandb", "swanlab"])
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=5.0,
        help="Monitor轮询间隔（分钟），默认5分钟，对应configs/agents.yaml的monitor.interval_minutes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id or generate_run_id()
    dataset_records = []
    if args.dataset_records_file:
        dataset_records = json.loads(Path(args.dataset_records_file).read_text(encoding="utf-8"))

    context = RunContext(
        run_id=run_id,
        task_description=args.task_description,
        dataset_path=args.dataset_path,
        dataset_sample=args.dataset_sample,
        dataset_records=dataset_records,
        indicators=args.indicators,
        resource_constraints=args.resource_constraints,
        available_resources=args.available_resources,
    )
    config = StateMachineConfig(
        poll_interval_seconds=args.interval_minutes * 60,
        logger_type=args.logger_type,
    )

    state_machine = StateMachine(context=context, config=config)
    try:
        final_state = state_machine.run()
    finally:
        state_machine.stop_all_agents()

    print(f"run_id={run_id}")
    print(f"final_state={final_state.value}")
    print(f"产物目录: {config.runs_root / run_id}")
    if state_machine.knowledge_card_id:
        print(f"knowledge_card_id={state_machine.knowledge_card_id}")

    return 0 if final_state == PipelineState.DONE else 1


if __name__ == "__main__":
    sys.exit(main())
