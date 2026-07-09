# CLI引擎桥接适配升级：STDIO Streaming / JSON-RPC 通信层

将 `agents/engines/*` 中对 `codex`/`claude`/`opencode` 的简单 `subprocess.run` 一次性封装，升级为各CLI原生最强通信协议（codex 的 `app-server --stdio` JSON-RPC 2.0、claude 的 `stream-json` 双向NDJSON流、opencode 的 ACP/serve 能力）的长驻进程桥接层，通过统一的 `BaseAgentEngine` 接口对上层 Agent 屏蔽协议差异，所有 Agent（规划类+执行类）统一迁移到新桥接层。

## 调研结论（已通过WebSearch核实）
| CLI | 最强原生协议 | 关键特性 |
| --- | --- | --- |
| `codex` | `codex app-server --stdio` | 完整 JSON-RPC 2.0（隐藏`jsonrpc:"2.0"`头），需 `initialize`→`initialized`握手，`thread/start`→`turn/start`发起对话，服务端流式推送 `item/started`/`item/agentMessage/delta`/`item/completed`/`turn/completed` 通知，支持 `command/exec*` 系列方法执行/流式读取/终止子命令，`approvalPolicy`可设为`never`免交互审批 |
| `claude` | `claude -p --output-format stream-json --input-format stream-json --verbose --include-partial-messages` | NDJSON双向流：stdin写入`{"type":"user_message","content":...}`，stdout逐行JSON事件（`stream`/`delta`/`tool_use_start`/`result`等）；SDK MCP工具调用时会内嵌标准JSON-RPC 2.0请求（`sdk_control_request`/`mcp_message`），需按`permission-mode`配置规避交互式审批 |
| `opencode` | ACP (Agent Client Protocol) | 官方文档确认支持ACP，是编辑器/客户端通用的stdio JSON-RPC标准（与`claude --acp`/`codex acp`同族），具体子命令/参数需在实现阶段以当前安装版本CLI帮助信息验证（文档覆盖度不如前两者） |

**设计决策（已与用户确认）**：不强求三者收敛到统一底层协议（如全部走ACP），而是**各engine适配其原生最强协议，通过统一的 `BaseAgentEngine` 抽象接口封装差异**；且**所有Agent（Planning六个+Execution三个）统一切换到新桥接层**，即使一次性结构化输出场景也通过新协议获取（可选择不消费中间事件，等价于原有阻塞式调用）。

## 统一接口设计
```python
# agents/engines/base_engine.py
class AgentEvent:  # 统一事件模型，各engine内部协议差异在此层被归一化
    type: str        # "text_delta" | "tool_use_start" | "tool_use_end" | "exec_output" | "error" | "done"
    payload: dict
    raw: dict         # 原始协议消息，便于调试

class AgentResult:
    text: str                     # 拼接后的最终文本
    structured_output: dict | None
    usage: dict | None
    events: list[AgentEvent]      # 完整事件轨迹（用于日志/回放）

class BaseAgentEngine(ABC):
    def start(self) -> None: ...              # 拉起长驻子进程 + 协议握手（若协议需要）
    def run(self, system_prompt, user_prompt, output_schema=None,
            on_event: Callable[[AgentEvent], None] | None = None,
            timeout: float = ...) -> AgentResult: ...
    def cancel(self) -> None: ...              # 中途取消当前turn（仅JSON-RPC类协议支持）
    def stop(self) -> None: ...                # 优雅关闭：关闭stdin/发终止请求，超时后SIGTERM
```
- `run()` 对上层始终表现为阻塞调用（不传 `on_event` 等价于旧版 `subprocess.run` 语义），但内部通过流式协议读取，可选把中间事件转发给调用方（Monitor/Trainer 用于日志/实时展示，Planning类可忽略）。
- 若指定 `output_schema`，在拼接完整文本/收到 `result`/`turn/completed` 后用 pydantic 校验，失败则重试（保留现有重试语义）。

## 各Engine实现要点
- **`codex_engine.py`**：`subprocess.Popen(["codex", "app-server", "--stdio"], ...)`长驻；封装 `initialize`→`initialized`握手，每次`run()`发起`thread/start`（`approvalPolicy: {type: "never"}`, `sandbox`透传现有配置）→`turn/start`，读取通知流映射为`AgentEvent`，`turn/completed`时结束并汇总；`cancel()`调用中断方法（若协议提供，需实现阶段核实具体method名）。
- **`claude_engine.py`**：`subprocess.Popen(["claude", "-p", "--output-format", "stream-json", "--input-format", "stream-json", "--verbose", "--include-partial-messages", "--permission-mode", <配置的免审批模式>], ...)`；写入NDJSON `user_message`到stdin；逐行解析stdout：`stream`事件的`text_delta`映射为`AgentEvent`，`tool_use_start/end`同理，收到顶层`result`即结束该轮；若出现`sdk_control_request`（MCP工具回调）按文档协议应答，避免60秒超时。
- **`opencode_engine.py`**：优先尝试ACP子命令/参数（需实现阶段用 `opencode --help`/`opencode acp --help` 等实测确认，当前文档信息有限）；若ACP暂不可用或不稳定，降级为现有 `opencode run` 一次性subprocess封装并在日志中标注降级状态（保证功能不中断，风险见下）。
- **`jsonrpc_transport.py`**（新增，codex/opencode共用）：通用NDJSON stdio JSON-RPC读写工具——请求/响应按`id`匹配（维护`{id: Future}`表）、通知按`method`路由到回调、写入时补/剥`jsonrpc:"2.0"`字段以兼容codex的省略约定、支持独立读线程持续拉取stdout避免阻塞写入。
- **`process_manager.py`**（新增）：长驻子进程生命周期管理——启动、存活检测（定期心跳/管道健康检查）、优雅关闭（关stdin/发终止请求，超时SIGTERM→SIGKILL）、异常退出后的自动重启策略（对Planning类一次性调用可关闭重启；对Monitor长期运行的引擎需谨慎重启避免丢失上下文）。

## 目录结构变更（增量）
```
agents/engines/
├── base_engine.py        # 变更：BaseAgentEngine抽象升级为上述接口 + AgentEvent/AgentResult
├── jsonrpc_transport.py  # 新增：通用JSON-RPC 2.0 stdio帧读写/请求路由工具
├── process_manager.py    # 新增：长驻子进程生命周期管理
├── codex_engine.py       # 变更：改为 app-server --stdio JSON-RPC 客户端
├── claude_engine.py      # 变更：改为 stream-json 双向NDJSON客户端
└── opencode_engine.py    # 变更：优先ACP，不可用则降级为原subprocess封装
```

## 配置变更
`configs/agents.yaml` 每个Agent条目新增可选字段：
```yaml
scenario_analysis: {engine: claude, model: default, timeout: 120, permission_mode: bypassPermissions}
trainer:            {engine: codex, model: default, timeout: 60, sandbox: workspace-write}
monitor:            {engine: claude, model: default, timeout: 180, interval_minutes: 5, permission_mode: bypassPermissions}
```
- `permission_mode`/`sandbox`等透传给对应engine握手参数，替代原来仅用CLI flag拼接的方式。
- 新增全局开关（如 `engines.process_pool: per_call|warm`）控制引擎进程是每次`run()`临时拉起+关闭（默认，隔离性更好，与现状行为一致）还是维持常驻池复用（可选优化项，本阶段默认关闭）。

## 对上层Agent的影响
- `BaseAgent`（`agents/base_agent.py`）调用方式不变：仍是 `engine.run(system_prompt, user_prompt, output_schema=...)`，因此 Planning 六个 Agent、`plan_reviewer_agent.py`、`report_writer_agent.py`、`evaluator_agent.py`、`knowledge_update_agent.py` **无需修改调用代码**，仅底层协议升级。
- `monitor_agent.py`/`trainer_agent.py` 可选传入 `on_event` 回调，将中间事件（如Codex的`command/exec`输出流、Claude的分析文本delta）实时追加写入各自的日志文件（`monitor_reports/`执行过程日志、Trainer的训练启动过程日志），提升可观测性，但不改变最终产物契约（`monitor_reports/*.md`、`training_plan.md`等文件格式不变）。

## 交付步骤（增量，接续现有交付步骤）
17. `jsonrpc_transport.py` + `process_manager.py` 基础设施实现与单元测试（mock stdio管道，测试请求/响应匹配、通知路由、超时、优雅关闭）。
18. `codex_engine.py` 重写为 `app-server --stdio` 客户端：实测当前已安装 `codex`(0.128.0) 是否支持 `app-server --stdio`（文档显示v0.136+），若版本过低则记录降级方案（fallback到`codex exec --json`一次性NDJSON流，仍优于纯文本）。
19. `claude_engine.py` 重写为 `stream-json` 双向客户端：实测`--input-format stream-json`实际消息格式（官方文档不完整，参考已核实的第三方协议文档），验证`permission-mode`免审批配置生效。
20. `opencode_engine.py` 实测ACP支持情况（`opencode --help`/官方ACP文档 `opencode.ai/docs/acp/`），确定可用则实现JSON-RPC客户端，否则实现降级路径并记录已知限制。
21. `base_engine.py`/`AgentEvent`/`AgentResult` 统一接口 + 三个engine的集成测试（同一 `run()` 调用在三种engine下行为一致，`on_event`回调可选消费）。
22. Planning/Execution 全部现有Agent集成回归测试：验证迁移到新桥接层后功能不退化（`training_plan.md`/`analysis_report.md`/`monitor_reports/`等产物内容与迁移前等价）。

## 待确认/风险点
- 当前环境 `codex`(0.128.0) 是否已支持 `app-server --stdio`（该flag据资料在v0.136+才加入），需实现阶段用 `codex app-server --help` 实测确认，若不支持需降级为 `codex exec --json`（单向NDJSON事件流，非完整JSON-RPC）。
- `claude --input-format stream-json` 官方文档不完整（见 anthropics/claude-code#24594），需按第三方逆向文档（`claude-agent-sdk-go`/`claude-max-api-proxy`等）实测校验实际消息格式，存在协议细节随版本变化的风险。
- `opencode`(1.14.40) 的ACP具体调用方式（子命令/参数）文档覆盖不足，需实现阶段直接查CLI帮助确认；若不可用，`opencode_engine.py`降级为原一次性subprocess封装，三engine能力不对等需在文档中明确标注。
- 长驻子进程相比原一次性`subprocess.run`引入新的资源管理复杂度（僵尸进程、管道阻塞死锁风险），`process_manager.py`需覆盖异常退出/超时清理测试。
- Claude的`sdk_control_request`/MCP回调协议若需要处理（例如未来接入自定义工具），复杂度较高；当前阶段通过`permission_mode`免审批规避大部分交互，未来若需支持SDK级MCP工具再扩展。
- 三种协议的错误语义不同（JSON-RPC错误对象 vs NDJSON的`error`事件 vs 进程退出码），`AgentEvent(type="error")`需要设计统一的错误归一化字段，便于上层Agent一致处理重试逻辑。
