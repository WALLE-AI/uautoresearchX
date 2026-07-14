"""多智能体训练框架的正式CLI入口（打包后的`uautoresearchx`命令）。

子命令：
    run      发起一次新的训练闭环（等价于原`run_pipeline.py`的全部能力）
    resume   从`manifest.json`恢复一次被中断的运行，继续跑完剩余闭环
    wizard   交互式向导：逐项询问任务信息后发起一次`run`，无需记忆命令行参数
    status   查看某次运行的当前状态（只读，不构造Agent对象）
    list     列出`runs/`下全部运行及其状态
    cancel   终止一次运行（若训练子进程仍在跑则发送SIGTERM/SIGKILL）
    logs     查看/跟随某次运行的训练日志

打包入口见`pyproject.toml`的`[project.scripts]`：`uv pip install -e .`后可直接
执行`uautoresearchx <子命令>`，不再需要手写`python -m
orchestrator.run_pipeline`长命令（该模块仍保留，作为薄兼容包装）。
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from orchestrator.human_gate import InteractiveHumanGate
from orchestrator.run_registry import (
    TERMINAL_STATUSES,
    ManifestNotFoundError,
    is_pid_alive,
    list_manifests,
    load_manifest,
    save_manifest,
    terminate_pid,
)
from orchestrator.state_machine import PipelineState, RunContext, StateMachine, StateMachineConfig
from orchestrator.tui import NullTUI, PipelineTUI, build_tui

app = typer.Typer(add_completion=False, no_args_is_help=True, help="多智能体自动化训练框架 CLI")
console = Console()


def _log_tail_path(config: StateMachineConfig, run_id: str) -> Path:
    return config.logs_root / run_id / config.logger_type / "train.log"


def _run_with_tui(
    state_machine: StateMachine, *, tui_enabled: bool, driver: str
) -> PipelineState:
    """在`tui_enabled`且当前终端是tty时接上`PipelineTUI`，否则退化为现状的纯
    文本输出；`driver`是`"run"`或`"resume"`，决定调用`state_machine.run()`还是
    `state_machine.resume()`。
    """
    tui: PipelineTUI | NullTUI = build_tui(
        enabled=tui_enabled,
        is_tty=sys.stdout.isatty(),
        console=console,
        log_tail_path_provider=lambda: _log_tail_path(state_machine.config, state_machine.ctx.run_id),
    )
    with tui:
        state_machine.set_callbacks(on_event=tui.on_event, on_transition=tui.on_transition)
        try:
            if driver == "run":
                return state_machine.run()
            return state_machine.resume()
        except Exception as exc:
            # 未被状态机自身捕获的异常（如AgentRunError重试耗尽、engine层
            # TimeoutError）会在这里向上冒泡——先把manifest标记为FAILED，
            # 避免进程崩溃退出后manifest永远停留在status=RUNNING，
            # 让status/list/resume之后还能正确反映"这次运行已经废弃"。
            state_machine.mark_failed(exc)
            raise
        finally:
            state_machine.stop_all_agents()


def _generate_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:8]}"


def _load_dataset_records(path: str | None) -> list[dict]:
    if not path:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_run_result(state_machine: StateMachine) -> None:
    console.print(f"final_state={state_machine.state.value}")
    console.print(f"产物目录: {state_machine.config.runs_root / state_machine.ctx.run_id}")
    if state_machine.knowledge_card_id:
        console.print(f"knowledge_card_id={state_machine.knowledge_card_id}")


@app.command()
def run(
    task_description: Annotated[str, typer.Option("--task-description", help="训练任务描述")],
    dataset_path: Annotated[str, typer.Option("--dataset-path")] = "未提供，请基于描述估算",
    dataset_sample: Annotated[str, typer.Option("--dataset-sample")] = "",
    dataset_records_file: Annotated[
        str | None,
        typer.Option(
            "--dataset-records-file",
            help="JSON文件路径，内容为list[dict]，供Trainer Agent的data_format_converters转换使用",
        ),
    ] = None,
    indicators: Annotated[str, typer.Option("--indicators")] = "无特殊要求",
    resource_constraints: Annotated[str, typer.Option("--resource-constraints")] = "无特殊约束",
    available_resources: Annotated[
        str, typer.Option("--available-resources")
    ] = "8x NVIDIA A100-SXM4-40GB",
    run_id: Annotated[str | None, typer.Option("--run-id", help="不指定则自动生成")] = None,
    logger_type: Annotated[str, typer.Option("--logger-type")] = "local",
    interval_minutes: Annotated[
        float,
        typer.Option(
            "--interval-minutes",
            help="Monitor轮询间隔（分钟），默认5分钟，对应configs/agents.yaml的monitor.interval_minutes",
        ),
    ] = 5.0,
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    knowledge_base_root: Annotated[Path, typer.Option("--knowledge-base-root")] = Path("knowledge_base"),
    tui: Annotated[
        bool,
        typer.Option("--tui/--no-tui", help="实时进度展示面板（非tty环境自动降级为纯文本输出）"),
    ] = True,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            help="在关键决策点（Plan Review结论/训练FAIL后的回退方向/达到最大重试次数）暂停等待人工确认，默认全自动",
        ),
    ] = False,
) -> None:
    """发起一次新的训练闭环，从Planning跑到DONE/FAILED。

    训练子进程会脱离本CLI进程组独立运行（见`agents/execution/trainer_agent.py:
    launch_stage`），单纯Ctrl-C关闭本次`run`不会顺带终止已启动的训练——如需
    真正停止，请用`uautoresearchx cancel <run_id>`；被打断的运行可以用
    `uautoresearchx resume <run_id>`继续。
    """
    resolved_run_id = run_id or _generate_run_id()
    dataset_records = _load_dataset_records(dataset_records_file)

    context = RunContext(
        run_id=resolved_run_id,
        task_description=task_description,
        dataset_path=dataset_path,
        dataset_sample=dataset_sample,
        dataset_records=dataset_records,
        indicators=indicators,
        resource_constraints=resource_constraints,
        available_resources=available_resources,
    )
    config = StateMachineConfig(
        poll_interval_seconds=interval_minutes * 60,
        logger_type=logger_type,
        runs_root=runs_root,
        logs_root=logs_root,
        knowledge_base_root=knowledge_base_root,
    )

    console.print(f"[bold]run_id[/bold]={resolved_run_id}")
    human_gate = InteractiveHumanGate(console=console) if interactive else None
    state_machine = StateMachine(context=context, config=config, human_gate=human_gate)
    final_state = _run_with_tui(state_machine, tui_enabled=tui and not interactive, driver="run")

    _print_run_result(state_machine)
    raise typer.Exit(code=0 if final_state == PipelineState.DONE else 1)


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="要恢复的run_id")],
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    tui: Annotated[
        bool,
        typer.Option("--tui/--no-tui", help="实时进度展示面板（非tty环境自动降级为纯文本输出）"),
    ] = True,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            help="在关键决策点（Plan Review结论/训练FAIL后的回退方向/达到最大重试次数）暂停等待人工确认，默认全自动",
        ),
    ] = False,
) -> None:
    """从`manifest.json`恢复一次被中断（CLI进程被杀/机器重启）的运行。

    resume的粒度是"重跑最后一次未完成的单个操作"：被打断的单次Planning
    Agent调用会完整重新调用一次；若被杀时训练子进程仍存活（`launch_stage()`用
    `start_new_session=True`使其独立于CLI进程组），会重新接管该pid继续监控，
    不会重启训练。
    """
    try:
        manifest = load_manifest(runs_root, run_id)
    except ManifestNotFoundError as exc:
        console.print(f"[red]错误[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]resume run_id[/bold]={run_id}（恢复前状态: {manifest.pipeline_state}）")
    human_gate = InteractiveHumanGate(console=console) if interactive else None
    try:
        state_machine = StateMachine.resume_from_manifest(manifest, human_gate=human_gate)
    except ValueError as exc:
        console.print(f"[red]错误[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    final_state = _run_with_tui(state_machine, tui_enabled=tui and not interactive, driver="resume")

    _print_run_result(state_machine)
    raise typer.Exit(code=0 if final_state == PipelineState.DONE else 1)


@app.command()
def wizard(
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    knowledge_base_root: Annotated[Path, typer.Option("--knowledge-base-root")] = Path("knowledge_base"),
) -> None:
    """交互式向导：逐项询问任务描述/数据集/指标等信息后发起一次`run`。

    等价于`run`子命令，只是把一长串必须记忆的`--xxx`参数换成逐项问答，适合
    偶尔使用/不熟悉全部参数的场景；熟悉参数后建议直接用`run`加脚本化。
    """
    console.print("[bold]多智能体训练框架 —— 交互式发起向导[/bold]\n")

    task_description = Prompt.ask("训练任务描述")
    dataset_path = Prompt.ask("数据集路径", default="未提供，请基于描述估算")
    dataset_sample = Prompt.ask("数据集样例摘要（可留空）", default="")
    dataset_records_file = Prompt.ask("dataset_records JSON文件路径（可留空）", default="") or None
    indicators = Prompt.ask("目标指标", default="无特殊要求")
    resource_constraints = Prompt.ask("资源/时间约束", default="无特殊约束")
    available_resources = Prompt.ask("可用资源", default="8x NVIDIA A100-SXM4-40GB")
    logger_type = Prompt.ask("日志形态", choices=["local", "wandb", "swanlab"], default="local")
    interval_minutes = float(Prompt.ask("Monitor轮询间隔（分钟）", default="5.0"))
    interactive = Confirm.ask("是否在关键决策点暂停等待人工确认？", default=False)
    tui_enabled = Confirm.ask("是否启用实时进度面板？", default=True)

    console.print("\n[bold]即将发起一次新的训练闭环[/bold]")
    if not Confirm.ask("确认发起？", default=True):
        console.print("已取消。")
        raise typer.Exit(code=0)

    run(
        task_description=task_description,
        dataset_path=dataset_path,
        dataset_sample=dataset_sample,
        dataset_records_file=dataset_records_file,
        indicators=indicators,
        resource_constraints=resource_constraints,
        available_resources=available_resources,
        run_id=None,
        logger_type=logger_type,
        interval_minutes=interval_minutes,
        runs_root=runs_root,
        logs_root=logs_root,
        knowledge_base_root=knowledge_base_root,
        tui=tui_enabled,
        interactive=interactive,
    )


@app.command("list")
def list_runs(
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
) -> None:
    """列出`runs_root`下全部运行及其状态。"""
    manifests = list_manifests(runs_root)
    if not manifests:
        console.print("（暂无运行记录）")
        return

    table = Table(title="训练闭环运行列表")
    table.add_column("run_id")
    table.add_column("status")
    table.add_column("pipeline_state")
    table.add_column("task_description")
    table.add_column("updated_at")
    for manifest in manifests:
        table.add_row(
            manifest.run_id,
            manifest.status,
            manifest.pipeline_state,
            manifest.task_description_preview[:40],
            manifest.updated_at,
        )
    console.print(table)


@app.command()
def status(
    run_id: Annotated[str, typer.Argument()],
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
) -> None:
    """查看某次运行的当前状态（只读，不构造Agent对象，不产生任何LLM调用）。"""
    try:
        manifest = load_manifest(runs_root, run_id)
    except ManifestNotFoundError as exc:
        console.print(f"[red]错误[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"run_id: {manifest.run_id}")
    console.print(f"status: {manifest.status}")
    console.print(f"pipeline_state: {manifest.pipeline_state}")
    console.print(f"stage_index: {manifest.stage_index}")
    console.print(f"stage_iteration: {manifest.stage_iteration}")
    console.print(f"replanning_attempts: {manifest.replanning_attempts}")
    console.print(f"updated_at: {manifest.updated_at}")
    if manifest.training_pid is not None:
        alive = is_pid_alive(manifest.training_pid)
        console.print(
            f"training_pid: {manifest.training_pid} "
            f"({'[green]alive[/green]' if alive else '[red]dead[/red]'})"
        )
    if manifest.knowledge_card_id:
        console.print(f"knowledge_card_id: {manifest.knowledge_card_id}")

    monitor_reports_dir = runs_root / run_id / "monitor_reports"
    if monitor_reports_dir.exists():
        reports = sorted(monitor_reports_dir.glob("report_*.md"))
        if reports:
            console.print(f"最新monitor报告: {reports[-1].name}")


@app.command()
def cancel(
    run_id: Annotated[str, typer.Argument()],
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
) -> None:
    """终止一次运行：若训练子进程仍存活则先SIGTERM/超时SIGKILL，再标记CANCELLED。

    `CANCELLED`是终态，之后`resume`会拒绝恢复——如需继续，请发起一次新的run。
    """
    try:
        manifest = load_manifest(runs_root, run_id)
    except ManifestNotFoundError as exc:
        console.print(f"[red]错误[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    if manifest.status in TERMINAL_STATUSES:
        console.print(f"run_id={run_id}已处于终态{manifest.status}，无需取消")
        return

    if manifest.training_pid is not None and is_pid_alive(manifest.training_pid):
        console.print(f"正在终止训练进程 pid={manifest.training_pid} ...")
        killed = terminate_pid(manifest.training_pid)
        console.print("已终止" if killed else "[red]终止失败，请手动检查该进程[/red]")

    manifest.status = "CANCELLED"
    save_manifest(runs_root, manifest)
    console.print(f"run_id={run_id}已标记为CANCELLED")


@app.command()
def logs(
    run_id: Annotated[str, typer.Argument()],
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    follow: Annotated[bool, typer.Option("--follow", "-f", help="像tail -f一样持续跟随新增内容")] = False,
) -> None:
    """查看/跟随某次运行的训练日志（`logs/<run_id>/<logger_type>/train.log`）。"""
    try:
        manifest = load_manifest(runs_root, run_id)
    except ManifestNotFoundError as exc:
        console.print(f"[red]错误[/red]: {exc}")
        raise typer.Exit(code=1) from exc

    logger_type = manifest.config.get("logger_type", "local")
    log_path = logs_root / run_id / logger_type / "train.log"
    if not log_path.exists():
        console.print(f"（暂无日志文件: {log_path}）")
        return

    with log_path.open("r", encoding="utf-8") as f:
        console.print(f.read(), end="")
        if not follow:
            return
        while True:
            line = f.readline()
            if line:
                console.print(line, end="")
            else:
                time.sleep(1)


if __name__ == "__main__":
    app()
