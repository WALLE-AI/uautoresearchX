# 多智能体训练框架 CLI 模式设计与实施计划 v1

## Context

`uautoresearchX` 目前已有一个可用的 Planning→Execution→Knowledge 全闭环编排引擎
（`orchestrator/state_machine.py`），入口是 `orchestrator/run_pipeline.py`——一个
基于 `argparse` 的一次性脚本：`uv run python -m orchestrator.run_pipeline
--task-description ... `，`main()` 阻塞式跑完 `StateMachine.run()` 后打印
`final_state` 就退出。这个入口只解决了"能从命令行发起一次训练闭环"，但作为
生产可用的 CLI 工具还有明显差距：

- **不可安装**：`pyproject.toml` 没有 `[project.scripts]`，只能用
  `python -m` 长命令调用，无法 `pip install -e .` 后直接敲命令。
- **无运行管理能力**：一次运行从 PLANNING 跑到 DONE/FAILED 全程只存在于单个
  Python 进程的内存里；`runs/<run_id>/` 只落盘产物文件（`training_plan.md`
  等），没有任何"当前处于哪个状态/第几个stage/进程是否还活着"的记录。CLI 进程
  一旦被杀（Ctrl-C、SSH断连、机器重启），这次运行就彻底丢失上下文，无法查看
  状态、无法恢复、也无法主动取消——只能翻 `runs/`/`logs/` 目录里的文件自己猜。
- **零人工干预点**：Plan Reviewer 通过/拒绝、Evaluator PASS/FAIL 后的回退方向，
  全部由 LLM 自主决定，用户即使想在关键节点介入确认也做不到。
- **零实时反馈**：三个engine（`agents/engines/*`）其实已经实现了完整的流式
  `on_event` 回调（`text_delta`/`tool_use_start`/`tool_use_end`/`done`），但
  `state_machine.py` 里没有任何一处调用时传了 `on_event`，用户在几分钟到几十
  分钟的一次 Agent 调用期间只能对着一个空白终端等待。

本计划的目标：让该多智能体框架真正"支持CLI模式"，覆盖用户确认的四个方面——
**(1) 打包为正式可安装CLI工具，(2) 增加 run/resume/status/list/cancel 子命令，
(3) 交互式/REPL模式，(4) 实时进度TUI展示**。四者共享同一套底层基础设施（运行
状态持久化 + 事件回调），因此设计上先做基础设施，再在其上叠加子命令/交互/TUI。

## 现状代码基线（供后续开发对照）

- `orchestrator/run_pipeline.py`：`build_arg_parser()` + `main()`，单一
  `--task-description`等flag，无子命令。
- `orchestrator/state_machine.py`：`StateMachine.__init__`（第91-129行）是唯一
  的构造入口，`_transition()`（136-138行）是所有状态切换的唯一收口点，
  `run()`（140-153行）是唯一的顶层驱动方法。9处 `*_agent.run(...)`/
  `poll_once`/`build_stage_config` 调用（158-282行、369行、182行）目前都没有
  传 `on_event`。
- `agents/execution/trainer_agent.py:launch_stage`（183-213行）用
  `subprocess.Popen(..., stdout=DEVNULL, stderr=DEVNULL)` 启动训练子进程，**未
  设置 `start_new_session`**，与CLI主进程同一进程组，主进程被杀训练子进程大概率
  一起被杀。
- `agents/engines/claude_engine.py`/`opencode_engine.py` 已经完整实现
  `on_event` 流式回调（文本delta、tool调用等），随时可接。
- `runs/<run_id>/` 目前只有产物文件，无任何状态清单；`pyproject.toml` 依赖里
  没有任何终端UI/CLI框架库（`rich`/`typer`/`click`/`textual`均未引入）。
- `多智能体训练框架-架构审阅与修订计划-v1.1.md` 记录的 P0-4（`codex_engine.py`/
  `opencode_engine.py` 每次 `run()` 都新增notification handler、从不注销，长期
  重复调用会内存泄漏）与本计划关系密切：一旦给 Monitor 循环接上 `on_event` 做
  TUI展示，Monitor 本来就要跑数小时/数天、每 `interval_minutes` 调一次
  `run()`，会直接放大这个已知泄漏。**建议在做TUI集成前先修复P0-4**（工作量小，
  独立可验证），否则长跑场景下TUI功能反而会暴露/加剧现有bug。这不是本计划新增
  范围，只是顺序建议。

## 总体设计

新增/改动文件一览：

```
orchestrator/
├── run_registry.py     # 新增：RunManifest模型 + 落盘/读取/扫描
├── cli.py               # 新增：Typer app，run/resume/status/list/cancel/logs子命令
├── human_gate.py         # 新增：交互式人工确认点的回调接口 + 默认自动实现 + 交互实现
├── tui.py                # 新增：基于rich的实时进度展示（Live + Layout）
├── state_machine.py      # 改动：接入on_event/on_transition回调、manifest持久化、resume入口、human_gate调用点
└── run_pipeline.py       # 改动为cli.py run子命令的薄兼容包装（保留旧用法不breaking）
pyproject.toml            # 改动：新增 typer/rich 依赖 + [project.scripts]入口
```

### 1. 运行状态持久化（Run Registry）—— 其余三项能力的共同地基

新增 `orchestrator/run_registry.py`：

```python
class RunManifest(BaseModel):
    run_id: str
    status: Literal["RUNNING", "PAUSED", "DONE", "FAILED", "CANCELLED"]
    pipeline_state: str                    # PipelineState.value
    created_at: str
    updated_at: str
    owner_pid: int | None                  # 持有本次运行的CLI进程pid，用于判活
    context: dict                          # RunContext的可序列化快照
    config: dict                           # StateMachineConfig的可序列化字段（去掉run_script_resolver）
    scenario_output: dict | None
    dataset_output: dict | None
    model_output: dict | None
    stage_index: int | None                # 当前/最后处理到的pipeline stage序号
    stage_iteration: int | None
    replanning_attempts: int
    training_pid: int | None               # 当前training子进程pid（若在跑），用于跨进程判活/cancel
    knowledge_card_id: str | None
    last_error: str | None

def save_manifest(runs_root: Path, manifest: RunManifest) -> None
def load_manifest(runs_root: Path, run_id: str) -> RunManifest
def list_manifests(runs_root: Path) -> list[RunManifest]   # 扫描 runs/*/manifest.json
```

`StateMachine` 改动：
- `_transition()`（现有唯一状态切换收口点）在切换状态后追加调用
  `self._save_manifest()`，把 `state`/`stage_index`/`replanning_attempts`等当前
  已知信息落盘到 `runs/<run_id>/manifest.json`。
- `_call_scenario`/`_call_dataset`/`_call_model_selection` 拿到
  `structured_output` 后同步写入 manifest（补齐现状代码中"这三个字典从不落盘"
  的缺口，是resume能拿到规划阶段中间结果的前提）。
- `launch_stage` 调用处记录 `training_pid` 到 manifest；`trainer_agent.py:
  launch_stage` 新增 `start_new_session=True`（POSIX，本项目环境是Linux）—— 训练
  子进程与CLI进程脱离进程组，CLI被杀不会连带杀死训练进程，这是"训练还在跑的时候
  CLI崩了，之后resume/status还能感知到它"的必要前提。
- 新增 `StateMachine.resume(manifest: RunManifest) -> StateMachine`
  classmethod：从manifest恢复 `RunContext`/`training_plan`（从
  `training_plan.json`重新load）/`_scenario_output`等，重建全新的Agent对象
  （现状代码本来就是每次全新构造，符合"Agent对象不可跨进程序列化，需重新构造"
  的现实），`state`直接设为manifest里记录的`pipeline_state`，跳过已完成的步骤。

**resume的粒度说明（重要，需要在计划里明确，避免过度设计）**：不做"任意时刻
快照恢复"，只做"恢复到最后一次成功完成的状态转换点"。例如CLI在某个Planning
Agent的LLM调用进行到一半时被杀，resume后会**重新完整调用一次该Agent**，而不是
尝试恢复半截的文本流——这与现有`BaseAgent.max_retries`重试语义一致，成本可接受
（单次Agent调用而非整条闭环）。若被杀时训练子进程仍在跑（`training_pid`存活），
resume直接重新进入 `_monitor_until_stage_done` 轮询该pid，不重启训练。

### 2. CLI打包与子命令（`orchestrator/cli.py`，用 `typer`）

`pyproject.toml` 新增依赖：`typer`（子命令框架，基于click，类型注解自动生成
`--help`）、`rich`（表格/进度展示，typer的可选美化依赖，同时供TUI/交互模式
复用）。新增 `[project.scripts] uautoresearchx = "orchestrator.cli:app"`。

子命令设计（复用 `run_registry.py` + 改造后的 `StateMachine`）：

| 子命令 | 功能 | 关键实现点 |
| --- | --- | --- |
| `uautoresearchx run` | 等价于现有`run_pipeline.py`全部flag，新增 `--tui/--no-tui`、`--interactive`、`--detach` | 复用`build_arg_parser`定义的参数（迁移为typer Option），构造`RunContext`+`StateMachineConfig`+`StateMachine`，默认接上TUI（见第4节） |
| `uautoresearchx resume RUN_ID` | 从`manifest.json`恢复并继续跑 | `run_registry.load_manifest` + `StateMachine.resume()`，若`status`已是`DONE`/`FAILED`则报错提示 |
| `uautoresearchx status RUN_ID` | 打印当前状态、stage进度、最近一次monitor报告风险等级、训练进程是否存活（`os.kill(pid, 0)`探活） | 只读，不构造Agent对象，直接读manifest+`runs/<run_id>/monitor_reports/`最新文件 |
| `uautoresearchx list` | 用`rich.table`列出所有run的run_id/status/task_description摘要/更新时间 | `run_registry.list_manifests()`遍历`runs/*/manifest.json` |
| `uautoresearchx cancel RUN_ID` | 终止：manifest标记`CANCELLED`；若`training_pid`存活则发`SIGTERM`（超时后`SIGKILL`） | 复用`process_manager.py`里已有的"优雅关闭→超时SIGTERM"模式思路，但作用对象是裸PID非`Popen`对象 |
| `uautoresearchx logs RUN_ID [--follow]` | 输出`logs/<run_id>/<logger_type>/train.log`或最近的agent调用日志（`logs/<run_id>/agents/*.json`） | `--follow`用简单的文件tail循环（`stat`轮询mtime+读增量） |

`orchestrator/run_pipeline.py` 改为薄包装：`main()`内部调用
`cli.run_command(...)`，保留原有`python -m orchestrator.run_pipeline`调用方式
不被破坏（现有测试/文档引用它的地方不用改）。

### 3. 交互式模式（`orchestrator/human_gate.py`）

现状：Plan Reviewer通过/拒绝、Evaluator PASS/FAIL后的回退方向完全由LLM自主
决定，无人工介入点。新增一个可选的人工确认层，**默认关闭，不改变现有自动化
行为**，仅在 `--interactive` 时启用：

```python
class HumanGate(Protocol):
    def review_plan(self, plan: TrainingPlanOutput, review: PlanReviewOutput) -> Literal["accept_llm_verdict", "force_approve", "force_reject"]: ...
    def on_stage_fail(self, evaluator_output: EvaluatorOutput) -> Literal["accept_llm_verdict", "retry", "replan", "abort"]: ...
    def on_max_retries_exceeded(self, context: str) -> Literal["abort", "extend_retries"]: ...

class AutoHumanGate:  # 默认实现，行为与现状完全一致（全部返回accept_llm_verdict）
class InteractiveHumanGate:  # rich.prompt.Confirm/Prompt实现，打印plan摘要/评审问题清单/失败诊断，等待用户输入
```

`StateMachine.__init__` 新增 `human_gate: HumanGate = AutoHumanGate()` 参数，在
`_run_planning_loop`（评审通过/拒绝分支）与 `_run_single_stage`（PASS/FAIL分支）
的判定点各插入一次 `human_gate.xxx(...)` 调用，返回值决定是否覆盖LLM原有判定。

另外新增一个独立的 **输入向导子命令** `uautoresearchx wizard`（或
`run --interactive-input`）：用 `rich.prompt` 逐项询问任务描述/数据集路径/
指标要求/资源约束等（对应现有`run`命令的一长串flag），生成等价的
`RunContext`后直接发起，降低用户记忆一长串命令行参数的门槛——这是用户提到
"REPL模式"最直接对应的需求点，实现成本低、独立于交互式人工确认点，可优先做。

### 4. 实时进度TUI（`orchestrator/tui.py`，用 `rich.live.Live`）

- `StateMachine.__init__` 新增 `on_event: Callable[[AgentEvent], None] | None`
  参数，转发给现状代码里9处尚未传参的`*_agent.run(...)`调用；`_transition()`
  额外触发一个 `on_transition(old_state, new_state)` 回调（新增参数），供TUI
  感知阶段切换。
- `tui.py` 提供 `PipelineTUI`：用 `rich.layout.Layout` 分区（当前阶段/当前Agent
  流式文本/最近metrics.csv行/最近monitor报告风险等级），通过上面两个回调驱动
  更新，`with PipelineTUI(...) as tui: state_machine = StateMachine(..., 
  on_event=tui.on_event, on_transition=tui.on_transition); state_machine.run()`。
- 训练阶段（MONITORING状态）的loss/epoch展示不必等5分钟一次的LLM Monitor调用
  ——额外起一个后台线程按更快节奏（如每2秒）tail `logs/<run_id>/<logger_type>/
  train.log`（复用`agents/log_adapters/local_log_adapter.py`的解析逻辑）刷新
  TUI里的"训练进度"面板，与LLM驱动的Monitor分析报告（风险等级/建议）是两条
  独立信息流，互不影响判定逻辑（只影响展示，不改变现有crash/FAIL判定路径）。
- **非TTY环境自动降级**：`run`命令用`sys.stdout.isatty()`判断，非交互终端
  （管道/CI/nohup）自动关闭TUI，退化为现有的纯文本print输出，避免`rich.Live`
  在非tty环境下产生乱码或阻塞。

## 任务拆解（延续仓库既有 `task-breakdown-c877cf.md` 的T-编号风格，新增T9-T13）

- **T9-1** `orchestrator/run_registry.py`：`RunManifest`模型 + 落盘/读取/
  `list_manifests`。【产出】模块+单元测试（round-trip序列化、扫描多个run目录）。
  【验证】`uv run pytest`。【依赖】无。
- **T9-2** 改造`trainer_agent.py:launch_stage`加`start_new_session=True`；
  改造`state_machine.py`在`_transition()`与三个planning调用点后写manifest。
  【产出】改动后的两个模块+回归测试（现有`test_state_machine.py`不应破坏）。
  【验证】`uv run pytest tests/test_state_machine.py tests/test_execution_agents.py`。
  【依赖】T9-1。
- **T9-3** `StateMachine.resume()`：从manifest恢复并跳过已完成步骤的集成测试
  （模拟"跑到TRAINING阶段中途→构造新StateMachine.resume()→验证不重新跑Planning"）。
  【产出】方法+测试。【验证】新增`tests/test_run_registry.py`或扩展
  `test_state_machine.py`。【依赖】T9-2。
- **T10-1** `pyproject.toml`加`typer`/`rich`依赖 + `[project.scripts]`；
  `orchestrator/cli.py`实现`run`（迁移现有argparse定义）+ `run_pipeline.py`
  改为薄包装。【产出】可安装命令，`uv pip install -e .`后`uautoresearchx run
  --help`可用。【验证】`uv run uautoresearchx run --help`；现有
  `python -m orchestrator.run_pipeline`调用方式不报错。【依赖】T9-1。
- **T10-2** `list`/`status`/`cancel`/`logs`子命令。【产出】4个子命令+
  `typer.testing.CliRunner`测试（构造临时`runs/`目录+假manifest验证输出）。
  【验证】`uv run pytest tests/test_cli.py`。【依赖】T9-1, T10-1。
- **T10-3** `resume`子命令接入T9-3。【产出】子命令+集成测试（先`run`一个
  故意在训练中途"崩溃"的场景，再`resume`验证从正确状态继续）。【验证】
  `uv run pytest`。【依赖】T9-3, T10-1。
- **T11-1** `orchestrator/human_gate.py`：`HumanGate`协议 + `AutoHumanGate`
  默认实现 + `InteractiveHumanGate`（rich.prompt）。`StateMachine`接入两处
  调用点。【产出】模块+改动。【验证】用`AutoHumanGate`跑现有全部
  `test_state_machine.py`断言行为不变；新增fake HumanGate测试三种返回值分支
  （accept/force_approve/force_reject等）都正确生效。【依赖】T9-2。
- **T11-2** `run --interactive` flag接入 + `wizard`输入向导子命令。【产出】
  CLI改动。【验证】CliRunner模拟stdin输入跑通向导生成正确的RunContext。
  【依赖】T11-1, T10-1。
- **T12-1** `on_event`/`on_transition`回调接入`StateMachine`全部9处调用点。
  【产出】改动+测试（用假回调断言收到的事件类型/次数符合预期，参考
  `test_state_machine.py`现有`ScriptedEngine`模式）。【验证】`uv run pytest`。
  【依赖】T9-2。
- **T12-2** `orchestrator/tui.py`：`PipelineTUI`（Live+Layout）+ train.log
  后台tail线程。`run`命令默认启用（`sys.stdout.isatty()`降级判断）。【产出】
  模块+`--tui/--no-tui` flag。【验证】非tty环境（如`| cat`）自动降级为文本
  输出的手工验证；`on_event`/`on_transition`回调正确驱动面板更新的单元测试
  （断言Layout内容而非视觉效果）。【依赖】T12-1, T10-1。
- **T13-1** 端到端回归：用`tests/fakes/`已有的`ScriptedEngine`+
  `fake_train_script.py`跑一次"完整run→模拟中途kill→resume→DONE"，同时验证
  `list`/`status`/`cancel`在这个场景下各阶段输出正确。【产出】新增端到端测试
  （参考现有`test_end_to_end_demo.py`模式）。【验证】`uv run pytest`。【依赖】
  T9-3, T10-3, T11-2, T12-2。

**建议实现顺序**：T9（地基）→ T10（打包+基础子命令，最快产生可见价值）→ T12
（TUI，复用T9的回调点，独立于交互模式可并行）→ T11（交互式，依赖T9的human_gate
挂载点）→ T13（收尾回归）。T11/T12可并行开发。

## 已知风险/权衡

- **resume不是任意粒度快照**：见上文"resume的粒度说明"，被杀时正在进行的单次
  LLM调用会整体重跑，这是有意的简化，不做sub-step级checkpoint。
- **`start_new_session=True`让训练子进程脱离CLI进程组**：意味着用户单纯
  `Ctrl-C`CLI不会再顺带杀掉训练——这是resume能力要求的必要代价，但也意味着
  用户必须显式用`cancel`子命令才能真正停止一次训练，需要在`run`命令的输出提示
  里明确告知（避免用户以为Ctrl-C退出了就万事大吉，结果GPU还在被占用）。
- **P0-4通知handler泄漏**（`多智能体训练框架-架构审阅与修订计划-v1.1.md`）：
  T12接入`on_event`后会被Monitor的长期重复`run()`调用放大暴露，建议T12-1开始
  前先确认该问题是否已修复（若未修复，建议作为T12-1的前置小任务一并处理，
  而非扩大本计划范围去重写整个engine层）。
- **非POSIX兼容性**：`start_new_session`/`os.kill(pid, 0)`探活为POSIX语义，当前
  部署环境是Linux，暂不考虑Windows兼容（与现有`claude_engine.py`注释里对
  Windows信号处理的已知限制保持一致的态度）。
- **TUI不覆盖训练脚本自身的实时loss**：面板展示的是`train.log`的tail结果，
  更新粒度取决于训练脚本`tee`写盘的频率，不是逐step推送。

## 验证方式

1. `uv run pytest`：全部新增/改动模块的单元测试 + 现有测试套件（
   `test_state_machine.py`/`test_execution_agents.py`等）保持通过，确认改动
   未破坏现有自动化行为。
2. 打包验证：`uv pip install -e .` 后执行 `uautoresearchx run --help`/
   `uautoresearchx list`/`uautoresearchx --help` 确认命令可发现、子命令齐全。
3. 端到端场景（复用`tests/fakes/`里的`ScriptedEngine`+假训练脚本，不依赖真实
   GPU/CLI二进制）：`run`一次→中途模拟进程被杀（不调用`stop_all_agents`直接
   丢弃对象）→`status`确认能读到`RUNNING`+训练pid存活→`resume`→跑到`DONE`→
   `list`确认状态更新为`DONE`。
4. TUI手工验证：真实终端下跑一次小规模demo（可复用
   `测试计划-SVRDD_YOLO端到端跑通-v1.md`里已经验证过的真实claude引擎+真实小
   数据集路径）观察面板刷新是否符合预期；`uautoresearchx run ... | cat`验证
   非tty自动降级不报错。
5. 交互式验证：`uautoresearchx wizard`手工走一遍向导流程生成正确命令；
   `run --interactive`人为构造一次Plan Reviewer拒绝场景，验证能在终端里看到
   问题清单并等待用户输入。
