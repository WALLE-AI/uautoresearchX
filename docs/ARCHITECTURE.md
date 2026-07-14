# 架构说明与数据契约文档

本文档描述当前仓库已实现的多智能体自动化训练框架的最终目录结构、各阶段职责、
Agent列表与文件契约。设计动机与原始架构讨论见仓库根目录的两份v1.0设计文档；
本文档只描述"当前代码实际实现的样子"。

## 目录结构

```
uautoresearchX/
├── configs/
│   ├── agents.yaml                     # 10个Agent的engine/model/timeout等配置
│   ├── training_engines.yaml           # 训练引擎清单与scripts/<engine>_run.sh映射
│   ├── training_pipeline_patterns.yaml # 训练流程模式库（参考基线）
│   └── data_format_patterns.yaml       # 数据格式规范库
├── agents/
│   ├── base_agent.py                   # BaseAgent抽象：构造prompt->调用engine->校验+重试
│   ├── engines/                        # codex/claude/opencode CLI桥接层
│   ├── planning/                       # 6个Planning Agent + schemas + prompt/io工具
│   ├── log_adapters/                   # local/wandb/swanlab日志形态归一化
│   ├── data_format_converters/         # sharegpt/alpaca/coco/yolo/mask转换器
│   ├── execution/                      # trainer/monitor/evaluator + schemas
│   └── knowledge/                      # knowledge_update_agent + schemas
├── orchestrator/
│   ├── state_machine.py                # 三阶段+核心循环状态机
│   ├── run_registry.py                 # RunManifest运行状态持久化（resume/status/list/cancel的地基）
│   ├── human_gate.py                   # 可选人工确认层（--interactive）
│   ├── tui.py                          # 实时进度展示（rich.Live）
│   ├── cli.py                          # 正式打包CLI（`uautoresearchx`命令），run/resume/wizard/list/status/cancel/logs
│   └── run_pipeline.py                 # 向后兼容入口，等价于`cli.py`的run子命令
├── scripts/                            # <engine>_run.sh 训练启动脚本
├── knowledge_base/
│   ├── index.json                      # {"entries": [{"card_id","summary","task_types"}]}
│   └── cards/<card_id>.json            # Knowledge Card
├── logs/<run_id>/{local,wandb,swanlab}/  # 训练引擎原始日志
├── runs/<run_id>/                      # 见下方"文件契约"
└── tests/                              # 单元测试 + fakes/（协议/训练脚本模拟）
```

## 三阶段职责

### 阶段一：Planning（`agents/planning/`）
`Scenario-Analysis -> Dataset-Analysis -> Model-Selection -> Training-Plan-
Generator -> Plan-Reviewer -> Report-Writer`，由`orchestrator/state_machine.py`
的`_run_planning_loop()`串联。Plan Reviewer拒绝时按`determine_rollback_target()`
返回的问题根源分层回退（数据理解->`dataset_analysis`，选型判断->
`model_selection`，计划参数->`training_plan`），最多重试
`StateMachineConfig.max_plan_review_retries`次（默认3次，即最多4轮评审），
超过则终止为`FAILED`状态并在`plan_review_log.md`中记录需要人工介入。

### 阶段二：Execution（`agents/execution/`）
`Trainer`（唯一可写config/超参、唯一调用`scripts/<engine>_run.sh`）+
`Monitor`（LLM驱动只读分析，每轮追加`metrics.csv`+`monitor_reports/*.md`）+
`Evaluator`（PASS/FAIL判定，逐轮追加`improve_guide.md`）构成核心循环，按
`training_plan.md`的`pipeline_stages`逐阶段执行。

"崩溃"判定采用双信号：训练子进程退出码（硬信号，由`state_machine.py`直接
检测）+ Monitor报告的`crash_detected`字段（软信号）任一为真即直接判FAIL，
不依赖LLM必须"发现"进程已经死亡。

FAIL时：`needs_replanning=False` -> 回退Trainer调整超参重试当前阶段（限
`max_stage_retries`次，默认3次）；`needs_replanning=True` -> 回退整个
Planning阶段重新规划（限`max_replanning_attempts`次，默认2次）。

### 阶段三：Knowledge（`agents/knowledge/`）
训练闭环结束（全部阶段PASS）后触发，汇总`training_plan.md`/
`analysis_report.md`/`improve_guide.md`/`monitor_reports/*.md`生成Knowledge
Card，写入`knowledge_base/cards/<card_id>.json`并更新`knowledge_base/
index.json`，供未来`Training-Plan-Generator`的`_load_similar_cases_summary()`
检索复用。

## Agent列表与`agent_id`

| Agent | `agent_id`（对应`configs/agents.yaml`） | 类 |
| --- | --- | --- |
| Scenario-Analysis | `scenario_analysis` | `agents.planning.scenario_analysis_agent.ScenarioAnalysisAgent` |
| Dataset-Analysis | `dataset_analysis` | `agents.planning.dataset_analysis_agent.DatasetAnalysisAgent` |
| Model-Selection | `model_selection` | `agents.planning.model_selection_agent.ModelSelectionAgent` |
| Training-Plan-Generator | `training_plan` | `agents.planning.training_plan_generator.TrainingPlanGeneratorAgent` |
| Plan-Reviewer | `plan_reviewer` | `agents.planning.plan_reviewer_agent.PlanReviewerAgent` |
| Report-Writer | `report_writer` | `agents.planning.report_writer_agent.ReportWriterAgent` |
| Trainer | `trainer` | `agents.execution.trainer_agent.TrainerAgent` |
| Monitor | `monitor` | `agents.execution.monitor_agent.MonitorAgent` |
| Evaluator | `evaluator` | `agents.execution.evaluator_agent.EvaluatorAgent` |
| Knowledge-Update | `knowledge_update` | `agents.knowledge.knowledge_update_agent.KnowledgeUpdateAgent` |

## 文件契约（`runs/<run_id>/`）

| 路径 | 产出者 | 说明 |
| --- | --- | --- |
| `training_plan.md` | Training-Plan-Generator | 人类可读训练计划 |
| `training_plan.json` | Training-Plan-Generator | `TrainingPlanOutput`结构化快照，Trainer/Monitor/Evaluator不解析markdown，直接消费此结构（见下方"关键设计决策"） |
| `plan_review_log.md` | Plan-Reviewer | 逐次评审记录（拒绝原因/问题清单/通过版本） |
| `analysis_report.md` | Report-Writer | 规划阶段汇总分析报告 |
| `data/` | Trainer（`prepare_data`） | 数据格式转换产物 |
| `stage_<n>_<name>/config.yaml` | Trainer（`build_stage_config`） | 单阶段训练配置 |
| `stage_<n>_<name>/checkpoints/` | 训练脚本自身产出 | 供下一阶段`start_from_path`解析 |
| `metrics.csv` | Monitor（`poll_once`） | 逐轮归一化六字段指标+GPU状态，累积趋势 |
| `monitor_reports/report_<seq>.md` | Monitor | 每轮LLM全面分析报告 |
| `improve_guide.md` | Evaluator（`evaluate`） | 逐轮迭代优化记录（PASS/FAIL均追加） |

`logs/<run_id>/{local,wandb,swanlab}/train.log`（或对应SDK本地同步文件）由
`scripts/<engine>_run.sh`产出，`agents/log_adapters/*`负责归一化解析。

## CLI模式

`uv pip install -e .`后可直接使用打包后的`uautoresearchx`命令（`pyproject.toml`
的`[project.scripts]`）；`orchestrator/run_pipeline.py`仍保留，作为
`uautoresearchx run`的薄兼容包装，`python -m orchestrator.run_pipeline
--task-description ...`旧用法不受影响。

- **`run`/`resume`/`wizard`/`list`/`status`/`cancel`/`logs`**：见
  `orchestrator/cli.py`。`run`发起新的训练闭环；`resume`从
  `runs/<run_id>/manifest.json`恢复被中断（CLI进程被杀/机器重启）的运行；
  `wizard`是`run`的交互式问答版本，避免记忆一长串命令行参数；`list`/`status`
  只读查看运行状态；`cancel`终止一次运行（若训练子进程仍存活会
  SIGTERM/超时SIGKILL）。
- **运行状态持久化**（`orchestrator/run_registry.py`）：`StateMachine`在每次
  `_transition()`及Planning阶段每个中间产出后把`RunManifest`落盘到
  `runs/<run_id>/manifest.json`，记录当前`pipeline_state`/`stage_index`/
  `training_pid`等。`agents/execution/trainer_agent.py:launch_stage()`用
  `start_new_session=True`让训练子进程脱离CLI进程组独立运行，并通过一层bash
  包装把退出码写入`<stage_dir>/exit_code.txt`哨兵文件——这是resume场景下
  重新接管一个非本进程fork出来的训练pid、获取其真实退出码的必要前提（POSIX下
  只有父进程能`wait()`拿到子进程退出码，CLI进程重启后已不是该训练进程的
  父进程）。
- **resume的粒度**：不做任意时刻快照恢复，只恢复到"最后一次成功完成的状态
  转换点"。被打断的单次Planning Agent调用会完整重新调用一次；若被杀时训练
  子进程仍存活，`StateMachine._resume_training_loop()`会重新接管该pid继续
  监控，不会重启训练。
- **`on_event`/`on_transition`回调**（`orchestrator/tui.py`）：`StateMachine`
  可选接受这两个回调，转发给全部9处`*_agent.run()`/`poll_once`/
  `build_stage_config`/`run_and_save`调用与每次`_transition()`。`claude_engine.
  py`/`opencode_engine.py`原本就已完整实现`AgentEvent`流式回调，`orchestrator/
  tui.py`的`PipelineTUI`只是给这条早已存在的事件流接一个`rich.Live`消费者。
  `run`/`resume`默认启用（`--tui/--no-tui`），非tty环境（管道/CI/nohup）自动
  降级为纯文本输出。
- **`--interactive`人工确认**（`orchestrator/human_gate.py`）：默认`
  AutoHumanGate`原样采纳LLM在Plan Review/训练FAIL回退方向/达到最大重试次数
  三类判定点的结论，与不开启`--interactive`时行为完全一致；显式传入
  `--interactive`才会用`InteractiveHumanGate`在终端暂停等待确认，可强制覆盖
  LLM判定或延长重试次数上限。

## 关键设计决策

1. **Trainer/Monitor/Evaluator不重新解析`training_plan.md`的markdown表格**，
   而是直接消费`orchestrator/state_machine.py`持有的`TrainingPlanOutput`
   pydantic对象（同时落盘`training_plan.json`供跨进程/调试查阅）。整条流水线
   在同一Python进程内运行，避免了对markdown表格做正则解析的健壮性风险。
2. **本地日志格式**：`scripts/*.sh`把训练框架原始stdout `tee`到
   `<log_dir>/train.log`，`local_log_adapter.py`用best-effort正则从中提取
   六字段，非HF Trainer/ultralytics已知格式的日志无法可靠抽取（有意的降级
   行为）。
3. **GPU状态由Monitor Agent自己调用`nvidia-smi`**，不属于log_adapters职责；
   不可用时优雅降级为None，不影响其余分析流程。
4. **训练子进程的stdout/stderr重定向到`DEVNULL`**（不设PIPE），因为训练脚本
   自身已把日志`tee`到磁盘文件，Python侧无需再捕获一份——特意避免重蹈Engine
   桥接层里"stderr未被drain导致管道死锁"的覆辙。

## 已知限制

本轮（T3-T8）交付**未包含**Engine桥接层（`agents/engines/*`）的bug修复。审阅
发现的问题（进程重启计数器失效、通知handler内存泄漏、Windows下`cancel()`
失效、stderr管道死锁风险等）记录在仓库根目录
`多智能体训练框架-架构审阅与修订计划-v1.1.md`的T1.5阶段，将在下一轮单独处理。

T8端到端验证全部使用测试替身（`ScriptedEngine`模拟LLM调用、
`tests/fakes/fake_train_script.py`代替真实训练脚本），未做真实GPU训练或真实
`codex`/`claude`/`opencode`二进制调用验证——当前开发环境不具备设计文档假设的
生产GPU资源。`wandb_log_adapter.py`/`swanlab_log_adapter.py`对本地同步文件
格式的假设同样未经真实环境实测（见两个模块docstring中的"未验证声明"），
生产使用前需要针对实际安装版本重新核实。

CLI模式（`run_registry.py`/`cli.py`/`human_gate.py`/`tui.py`，见
`多智能体训练框架-CLI模式设计与实施计划-v1.md`）同样全部用`ScriptedEngine`+
fake训练脚本验证，未做真实CLI/GPU环境下的resume/cancel/TUI人工验收；
`start_new_session=True`+pid判活/终止是POSIX语义，当前不考虑Windows部署。
resume只保证恢复到"最后一次成功完成的状态转换点"，不做任意时刻快照，被打断的
单次Planning Agent调用会完整重跑一次。
