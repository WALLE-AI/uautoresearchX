# Task级执行方案：多智能体训练框架从零实现

将 `multi-agent-training-framework-c877cf.md` 的全部交付步骤（基础1-16 + CLI桥接升级17-22）拆解为有明确产出物、验证方式与依赖关系的可直接执行task，且engines从一开始就按新协议（JSON-RPC/stream-json/ACP）构建，不再分两阶段（先subprocess再升级）。

## 编号与依赖说明
- Task ID格式 `T<阶段>-<序号>`，阶段按依赖顺序推进：`T0`基础设施 → `T1`Engine桥接层 → `T2`Planning Agents → `T3`训练脚本 → `T4`Execution Agents → `T5`Orchestrator闭环 → `T6`Knowledge → `T7`CLI入口/文档 → `T8`端到端回归。
- 每个task标注【产出】【验证】【依赖】三项；【依赖】指必须先完成的Task ID。
- 原主方案步骤号在末尾括号标注（如 `(原1)`、`(原17)`）便于追溯。

## T0 基础设施
- **T0-0** UV项目初始化：仓库根目录执行`uv init --python 3.12`（无src/布局，`agents/`/`orchestrator/`/`scripts/`等目录直接置于根目录），生成`pyproject.toml`+`.python-version`（固定`3.12`）；追加核心依赖：运行期`pydantic`/`pyyaml`/`httpx`，开发期`uv add --dev pytest pytest-mock mypy`。`[project] name`拟定`uautoresearchx`。【产出】`pyproject.toml`、`.python-version`、`uv.lock`。【验证】`uv run python --version`输出3.12.x；`uv run pytest --version`可执行。【依赖】无。
- **T0-1** 目录骨架搭建：在T0-0生成的根目录基础上补充`configs/`、`agents/`、`scripts/`、`orchestrator/`、`knowledge_base/`、`logs/`、`runs/`、`docs/`。【产出】空目录+`.gitkeep`。【验证】`find`命令核对结构与方案目录树一致。【依赖】T0-0。(原1)
- **T0-2** `configs/agents.yaml`：六Planning+三Execution+knowledge_update共10个Agent条目，含`engine/model/timeout`及新协议字段（`permission_mode`/`sandbox`/`interval_minutes`）。【产出】yaml文件+pydantic Schema校验脚本。【验证】`uv run python configs/validate_agents.py`加载并按Schema校验通过。【依赖】T0-1。(原1)
- **T0-3** `configs/training_engines.yaml` + `configs/training_pipeline_patterns.yaml` + `configs/data_format_patterns.yaml`：三份静态配置库。【产出】yaml文件。【验证】人工审阅字段覆盖llamafactory/trl/transformers/ultralytics/verl(占位)、LLM-SFT系列模式、五种数据格式规范。【依赖】T0-1。(原1)

> 依赖归属细化（后续按需`uv add`，不在T0-0一次性加全）：T1无新增第三方依赖（标准库足够）；T2/T4已含`pydantic`，如需表格解析可选`markdown-it-py`（视T2-5实现再定）；T4-1的`wandb`/`swanlab`作为可选依赖，用`uv add --optional wandb swanlab`避免强制安装。

## T1 Engine桥接层（直接按新协议构建）
- **T1-1** `jsonrpc_transport.py`：通用NDJSON stdio JSON-RPC读写工具（请求/响应id匹配、通知路由、独立读线程）。【产出】模块+单元测试（mock管道）。【验证】`uv run pytest`覆盖请求超时/响应匹配/通知分发。【依赖】T0-1。(原17)
- **T1-2** `process_manager.py`：长驻子进程生命周期管理（启动/存活检测/优雅关闭/异常重启）。【产出】模块+单元测试（mock subprocess）。【验证】`uv run pytest`覆盖正常退出/超时SIGTERM/异常重启路径。【依赖】T0-1。(原17)
- **T1-3** `base_engine.py`：`BaseAgentEngine`抽象 + `AgentEvent`/`AgentResult`数据类。【产出】接口定义。【验证】mypy/类型检查通过，抽象方法齐全。【依赖】T1-1, T1-2。(原3, 原21)
- **T1-4** `codex_engine.py`：`codex app-server --stdio` JSON-RPC客户端（initialize握手→thread/start→turn/start→事件流→turn/completed）。【产出】实现+集成测试。【验证】先跑`codex app-server --help`确认`--stdio`可用（否则降级为`codex exec --json`并记录），再跑一次真实prompt验证`AgentResult`正确。【依赖】T1-3。(原2, 原18)
- **T1-5** `claude_engine.py`：`stream-json`双向NDJSON客户端（stdin写user_message，解析stream/delta/tool_use/result事件，`permission-mode`免审批）。【产出】实现+集成测试。【验证】真实调用一次claude验证流式文本可拼接为完整结果。【依赖】T1-3。(原2, 原19)
- **T1-6** `opencode_engine.py`：优先适配ACP，`opencode --help`/`opencode acp --help`实测确认可用性；不可用则降级为`opencode run`一次性subprocess并标注降级状态。【产出】实现+集成测试（含降级路径测试）。【验证】真实调用验证两种模式均可返回`AgentResult`。【依赖】T1-3。(原2, 原20)
- **T1-7** 三engine一致性集成测试：同一`run()`签名在三种engine下行为一致，`on_event`回调可选消费不影响返回值。【产出】测试用例。【验证】`uv run pytest`三engine跑通同一prompt并断言`AgentResult`结构一致。【依赖】T1-4, T1-5, T1-6。(原21)

## T2 Planning Agents
- **T2-1** `agents/base_agent.py`：BaseAgent抽象（构造prompt→调用engine.run→pydantic校验→重试）。【产出】模块。【验证】单元测试覆盖校验失败重试逻辑。【依赖】T1-3。(原3)
- **T2-2** `scenario_analysis_agent.py`：产出任务类型/行业/难度/风险JSON，要求WebSearch/WebFetch检索。【产出】模块+输出Schema。【验证】用示例任务描述跑通，输出含引用来源字段。【依赖】T2-1。(原4)
- **T2-3** `dataset_analysis_agent.py`：EDA统计+数据格式候选推荐（ShareGPT/Alpaca/COCO/YOLO/mask）。【产出】模块+输出Schema。【验证】用示例数据集样例跑通，候选格式列表非空。【依赖】T2-1, T0-3。(原4)
- **T2-4** `model_selection_agent.py`：推荐模型+资源估算+模型对数据格式的硬性要求。【产出】模块+输出Schema。【验证】跑通并断言硬性格式要求字段存在。【依赖】T2-1。(原4)
- **T2-5** `training_plan_generator.py`：汇总产出`training_plan.md`（含`pipeline_stages`+`数据格式`定案章节）。【产出】模块+Markdown模板渲染器。【验证】跑通生成的`training_plan.md`可被简单正则/表格解析工具正确抽取资源规划与Pipeline Stages表格。【依赖】T2-2, T2-3, T2-4, T0-3。(原4)
- **T2-6** `plan_reviewer_agent.py`：评审`training_plan.md`，输出`plan_review_log.md`，分层回退（P4/P3/P2）。【产出】模块。【验证】构造故意缺陷计划验证拒绝路径正确触发对应回退目标；正常计划验证通过。【依赖】T2-5。(原4, 原13)
- **T2-7** `report_writer_agent.py`：汇总生成`analysis_report.md`（四章节+引用来源）。【产出】模块。【验证】人工审阅报告分章节完整，含检索引用。【依赖】T2-6。(原4, 原11)

## T3 训练脚本
- **T3-1** `scripts/<engine>_run.sh`（llamafactory/trl/transformers/ultralytics）：统一接口`bash run.sh <run_dir> <config_path> <log_dir> <logger_type>`。【产出】4个脚本，可运行最简训练命令。【验证】本地跑一个mini样例验证进程可启动并写入指定logger目录。【依赖】T0-1。(原5)
- **T3-2** `scripts/verl_run.sh`：占位脚本，输出"verl未安装"提示。【产出】脚本。【验证】执行输出提示信息，退出码非0但不crash主流程。【依赖】T0-1。(原5)

## T4 Execution Agents
- **T4-1** `log_adapters/`（base/local/wandb/swanlab）：读取归一化六字段指标。【产出】4个模块+单元测试。【验证】构造三种示例日志数据，验证解析出的字段与预期一致。【依赖】T0-1。(原6, 原14)
- **T4-2** `data_format_converters/`（base/sharegpt/alpaca/coco/yolo/mask）：按字段映射规则转换数据。【产出】6个模块+单元测试。【验证】构造小型LLM对话样例与CV标注样例，验证输出文件符合目标格式规范。【依赖】T0-3。(原16)
- **T4-3** `trainer_agent.py`：解析`training_plan.md`→执行数据格式转换→按pipeline_stages逐阶段调用`scripts/<engine>_run.sh`。【产出】模块。【验证】用一个通过评审的示例`training_plan.md`跑通配置生成+脚本调用（mock或mini真实训练）。【依赖】T2-6, T3-1, T4-2, T1-4。(原6)
- **T4-4** `monitor_agent.py`：每`interval_minutes`调用engine.run做LLM全面分析，输出`monitor_reports/*.md`，Critical告警触发Evaluator提前介入。【产出】模块。【验证】构造正常/早熟/发散/GPU异常四种模拟指标序列，验证每轮报告正确+Critical场景触发提前介入。【依赖】T4-1, T1-5。(原6, 原15)
- **T4-5** `evaluator_agent.py`：解析归一化指标+Monitor报告，判定PASS/FAIL，维护`improve_guide.md`逐轮追加。【产出】模块。【验证】构造PASS和FAIL两条路径，验证`improve_guide.md`均正确追加记录。【依赖】T4-4。(原6, 原12)

## T5 Orchestrator闭环
- **T5-1** `orchestrator/state_machine.py`：串联`PLANNING→PLAN_REVIEW→TRAINING→MONITORING→EVALUATING→KNOWLEDGE_UPDATE`状态机，支持FAIL分层回退。【产出】模块。【验证】构造FAIL场景验证正确回退到Trainer重试或Planning重规划，超过最大重试次数终止。【依赖】T2-7, T4-3, T4-5。(原7)

## T6 Knowledge
- **T6-1** `knowledge_update_agent.py` + `knowledge_base/`索引：汇总生成Knowledge Card并更新`index.json`。【产出】模块+索引结构。【验证】跑通闭环后验证card正确写入且index可按任务类型检索命中。【依赖】T5-1。(原8)

## T7 CLI入口与文档
- **T7-1** `orchestrator/run_pipeline.py`：CLI入口，从用户输入到闭环执行。【产出】可执行脚本。【验证】`uv run python run_pipeline.py --task "..."`跑通完整流程无异常。【依赖】T5-1, T6-1。(原9)
- **T7-2** `docs/ARCHITECTURE.md`：架构说明与数据契约文档。【产出】文档。【验证】人工审阅覆盖目录结构/Agent职责/文件契约。【依赖】T7-1。(原9)

## T8 端到端回归
- **T8-1** 端到端demo：用一个小型示例任务跑通全流程，验证monitor/evaluator日志格式。【产出】demo运行记录。【验证】`runs/<run_id>/`产出完整（training_plan.md/monitor_reports/improve_guide.md/knowledge card）。【依赖】T7-1。(原10)
- **T8-2** 全量Agent + 新引擎桥接层回归测试：确认迁移到新协议后产物内容与迁移前等价，三engine均可用（或按记录的降级路径可用）。【产出】回归测试报告。【验证】对比`training_plan.md`/`analysis_report.md`/`monitor_reports/`关键字段跨engine一致。【依赖】T8-1, T1-7。(原22)

## 执行建议顺序
`T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8`，其中同阶段内多个task（如T2-2/T2-3/T2-4，T4-1/T4-2）可并行开发。
