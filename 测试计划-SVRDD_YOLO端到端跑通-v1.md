# 测试多智能体训练框架：SVRDD_YOLO + YOLO 模型 + 全 claude 引擎 端到端真实跑通

## Context

`uautoresearchX` 是一个已经"设计完+代码写完"但**从未用真实 CLI 二进制、真实数据集、真实 GPU 跑过一次完整闭环**的多智能体自动化训练框架（Planning→Execution→Knowledge 三阶段，`orchestrator/state_machine.py` 驱动）。`docs/ARCHITECTURE.md` 和内部审阅文档 `多智能体训练框架-架构审阅与修订计划-v1.1.md` 都明确写着：现有测试全部基于 `tests/fakes/`（fake claude/codex/opencode 协议、fake 训练脚本），**没有任何一次调用过真实的 `claude`/`codex`/`opencode` 二进制**，且已知引擎桥接层存在若干 P0 级别风险点（stderr 管道死锁、Monitor 长轮询下的 notification-handler 泄漏等）。

本次任务目标：用 `/home/dataset1/gaojing/xibeiyuan/datasets/SVRDD/SVRDD_YOLO`（YOLO 格式检测数据集）+ `/home/dataset1/gaojing/xibeiyuan/models`（本地 YOLO 系列模型）作为真实输入，**把所有 Agent 强制切到 claude 引擎**（已与用户确认），跑通一次完整 pipeline（Planning→Execution 小规模真实训练→Knowledge），验证框架本身是否可用，暴露真实存在的问题。这是该框架的第一次真实验收测试，不是功能开发。

**已与用户确认的两个关键决策：**
1. 本次测试把 `configs/agents.yaml` 里全部 10 个 agent 的 `engine` 字段统一改为 `claude`（当前是 claude/codex/opencode 混合），专门验证 claude 引擎桥接层。
2. 跑通完整 pipeline，包含一次真实的小规模 YOLO 训练（不是只测 Planning 阶段的干跑）。

**已知阻塞/未验证项**：本次规划期间 Bash 工具因平台侧分类器故障持续不可用，未能实际执行 `find`/`ls` 查看 `SVRDD_YOLO` 目录结构、`models` 目录下的具体 YOLO checkpoint 文件名、`claude` CLI 是否已登录可用、GPU 是否空闲。这些是执行阶段的 **Phase 0**，必须最先跑，若发现与本计划假设不符（如数据集实际是分割格式而非检测框、模型目录里没有 `.pt` 权重只有 config），需要向用户反馈调整，而不是硬套本计划。

---

## Phase 0：环境与数据探查（执行阶段第一步，只读）

```bash
# 数据集结构 + 标注格式确认
find /home/dataset1/gaojing/xibeiyuan/datasets/SVRDD/SVRDD_YOLO -maxdepth 4
cat /home/dataset1/gaojing/xibeiyuan/datasets/SVRDD/SVRDD_YOLO/data.yaml   # 或 dataset.yaml，确认类别名列表、train/val 路径
head -5 <任意一个 labels/*.txt>                                            # 确认确实是 "class x y w h" 归一化格式

# 本地 YOLO 模型清单
find /home/dataset1/gaojing/xibeiyuan/models -maxdepth 4
# 找出可直接作为 start_from_path 的 .pt/.yaml 权重或结构文件

# claude CLI 可用性（框架实测基线是 claude-code 2.1.148，见 agents/engines/claude_engine.py 注释）
claude --version
echo '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"回复ok"}]}}' | \
  claude -p --input-format stream-json --output-format stream-json --verbose --permission-mode bypassPermissions

# ultralytics / GPU
python -c "import ultralytics; print(ultralytics.__version__)"
yolo version
nvidia-smi
```

根据结果确认：数据集类别数与名称、train 图片数量级、可用 GPU 数、`claude` CLI 是否需要额外登录步骤。若 `claude -p` 冒烟测试本身跑不通，后面所有阶段都无从谈起，必须先解决。

---

## Phase 1：全部 Agent 切到 claude 引擎

编辑 `configs/agents.yaml`：
- `dataset_analysis`: `engine: codex` → `engine: claude`，去掉 `sandbox: workspace-write`，加 `permission_mode: bypassPermissions`（与其余 claude agent 保持一致写法）。
- `trainer`: 同上（`engine: codex` → `claude`，去掉 `sandbox`，加 `permission_mode: bypassPermissions`）。
- `training_plan`: `engine: opencode` → `engine: claude`，加 `permission_mode: bypassPermissions`。
- 其余 6 个 agent（`scenario_analysis`/`model_selection`/`plan_reviewer`/`report_writer`/`monitor`/`evaluator`/`knowledge_update`）已经是 `claude`，不用动。

改完后全部 10 个 agent 都会走 `agents/engines/claude_engine.py`（`claude -p --input-format stream-json --output-format stream-json --include-partial-messages --verbose --permission-mode bypassPermissions`）。这一步同时验证 `agents/base_agent.py:build_engine()` 里 `codex`/`claude` 专属 kwargs 分支切换后不会遗留无效字段。

---

## Phase 2：把 SVRDD_YOLO 转成框架期望的中间记录格式

`TrainerAgent.prepare_data()` 期望的 `dataset_records`（经 `--dataset-records-file` 传入）是**通用 COCO-like 记录**，不是原始 YOLO txt：

```json
{"image": {"file_name": "0001.jpg", "width": 640, "height": 480},
 "objects": [{"category": "pothole", "bbox": [x_abs, y_abs, w_abs, h_abs]}]}
```

`agents/data_format_converters/yolo_converter.py` 会在 Trainer 阶段把这个中间格式**再转回** YOLO txt（`runs/<run_id>/data/yolo/*.txt` + `classes.txt`）。这是一次"YOLO→通用格式→YOLO"的往返，看似多余，但这正是框架设计要走的路径（Dataset-Analysis/Training-Plan 阶段只认这个通用格式），本次测试要按框架本来的路子走，以便真正验证转换器代码路径。

新增一个一次性脚本（测试用具，不放进 `agents/`，建议 `tools/prepare_svrdd_records.py` 或直接内联到运行步骤里）：
1. 读取 `data.yaml`（或 Phase 0 确认的实际文件名）拿到 `names`（class_id → 类别名映射）。
2. 遍历 train split 的 labels/*.txt，配合同名图片用 `PIL.Image.open(...).size` 拿宽高，把归一化的 `x_center,y_center,w,h` 还原成绝对像素 `[x,y,w,h]`（左上角+宽高）。
3. **为了满足"小规模测试"的约定，只抽样一小部分（如 50~100 张图）**，而不是全量数据集，保证训练阶段几分钟内跑完。
4. 输出两个东西：
   - `dataset_records.json`（喂给 `--dataset-records-file`）
   - 一段文本摘要（类别列表、样本数、每类目标数分布、1-2 条原始标注行示例），用于 `--dataset-sample`

**关键架构限制（需要写进 task-description/dataset-sample，而不是指望 agent 自己探查文件系统）**：`DatasetAnalysisAgent`/`ModelSelectionAgent`/`TrainerAgent` 在 `orchestrator/state_machine.py` 中构造时都没有传 `cwd`（`agents/base_agent.py` 的 `build_engine()` 只有传入非 None 的 `cwd` 时才会给 claude CLI 加 `--add-dir`），所以这些 agent 的 claude 子进程**看不到** `SVRDD_YOLO` 或 `models` 目录本身，只能依赖我们在 CLI 参数里塞的文本描述。因此 Phase 0 探查到的数据集/模型信息必须显式写进 `--dataset-path`/`--dataset-sample`/`--task-description`，尤其是本地可用模型清单要写进 `task-description`，否则 Model-Selection agent 会凭空推荐一个本地根本不存在的模型路径。

---

## Phase 3：组装并执行 CLI 调用

```bash
uv run python -m orchestrator.run_pipeline \
  --task-description "在 SVRDD_YOLO 数据集（<Phase0确认的领域，如道路缺陷/路面病害检测>，类别: <Phase0确认的类别列表>）上用 YOLO 做目标检测微调。本地可用基础模型清单（位于 /home/dataset1/gaojing/xibeiyuan/models）：<Phase0列出的具体文件路径>，请从中选择起点权重，不要假设网络可下载新权重。这是框架验收性质的小规模冒烟测试，目标是跑通完整流程而非追求最终精度，请用尽量小的 epoch 数（1-3）与小分辨率快速跑完。" \
  --dataset-path /home/dataset1/gaojing/xibeiyuan/datasets/SVRDD/SVRDD_YOLO \
  --dataset-sample "<Phase2生成的文本摘要>" \
  --dataset-records-file <Phase2生成的 dataset_records.json 路径> \
  --indicators "全流程无未处理异常跑到 DONE；产出至少一个有效 checkpoint 文件；mAP 非0即可，不要求收敛" \
  --resource-constraints "仅限使用1张GPU，训练阶段总时长不超过30分钟" \
  --available-resources "1x NVIDIA A100-40GB" \
  --logger-type local \
  --interval-minutes 1
```

`--interval-minutes 1` 让 `MonitorAgent` 真实 `time.sleep()` 轮询间隔缩到 1 分钟（仍是生产路径的真实 sleep，不是测试专用的 `poll_interval_seconds=0` 加速模式），在一个几分钟量级的小训练里能触发 2-3 次真实 Monitor 调用，足够验证长轮询链路（这正是 v1.1 审阅文档点名的高风险点：notification-handler 泄漏、stderr 管道死锁）而不用等 5 分钟一次。

先在前台跑（不要后台失联），全程盯着终端输出；预期耗时从 Planning 阶段几次 LLM 调用到训练完成，量级在 10-20 分钟。

---

## Phase 4：产物核对与已知风险观察点

跑完后依次检查：
- `runs/<run_id>/training_plan.md` + `training_plan.json`：`data_format.target_format` 是否命中 `yolo`（大小写不敏感子串匹配，见 `configs/data_format_patterns.yaml` 里的 `YOLO-txt`）；`pipeline_stages[].engine` 是否是 `ultralytics`。
- `runs/<run_id>/data/yolo/`：`YOLOConverter` 是否成功产出 `*.txt` + `classes.txt`。
- `runs/<run_id>/stage_1_*/config.yaml`：LLM 生成的 yaml 是否含 `task`/`model`/`data` 字段（`ultralytics_run.sh` 直接把 yaml 顶层键值拼成 `k=v` 传给 `yolo train`，list/dict 类型的值会被朴素 `str()` 化，若 LLM 生成了嵌套结构会导致 `yolo train` 参数不合法——留意这一点）。
- `logs/<run_id>/local/train.log`：应能看到真实 `yolo train` 的 epoch 进度输出和最后 `[ultralytics] training finished, exit_code=0`。
- `runs/<run_id>/monitor_reports/*.md` + `metrics.csv`：确认 `LocalLogAdapter` 真的从 `train.log` 里解析出了 epoch/loss 等字段。
- `runs/<run_id>/stage_1_*/checkpoints`：**注意** `orchestrator/state_machine.py:308-311` 假设下一阶段的 `start_from_path` 就是 `stage_dir/checkpoints`，但 `ultralytics_run.sh` 里 `yolo train project="$RUN_DIR"` 实际会把权重写到 `$RUN_DIR/train/weights/{last,best}.pt`（ultralytics 自己的默认子目录布局），路径对不上。如果 LLM 生成的 training_plan 只有一个 stage，这个问题不会暴露；如果有多个 stage（如"预热+主训练"两阶段），第二阶段拿到的 `start_from_path` 会指向一个不存在的目录——按现状先观察是否发生，不预先改代码。
- `knowledge_base/cards/<id>.json` + `knowledge_base/index.json`：Knowledge 阶段是否成功落盘。
- 全程留意 `agents/base_agent.py:BaseAgent.run()` 的结构化输出重试链路（`max_retries=2`，失败会把 pydantic 校验错误拼进下一次 prompt）：claude engine 没有走 `--json-schema`（`claude_engine.py` 注释里承认这一点，靠纯 prompt 指令 + 事后 `model_validate_json` 校验），真实模型输出偶尔带 ```` ```json ```` 代码块包裹是已知风险，观察是否触发重试甚至耗尽重试后抛 `AgentRunError`（`orchestrator/run_pipeline.py:main()` 没有 catch，会直接让 CLI 崩溃退出，这是预期行为不是要改的 bug）。

---

## Phase 5：问题归档，不做预防性大改

本次任务的产出是"跑通 + 发现了什么问题"的清单，不是把 `多智能体训练框架-架构审阅与修订计划-v1.1.md` 里列的 P0 bug 清单（T1.5）全部修一遍——那是范围明确更大的独立任务。原则：
- 阻塞性错误（导致流程根本跑不到底，比如 claude CLI 调不通、YOLOConverter 直接抛异常）：定位根因，做**最小化**修复，让流程能跑通，并在最终汇报里写清楚改了什么、为什么。
- 非阻塞性问题（如上面 Phase 4 提到的 checkpoint 路径假设、config.yaml 里 list 字段序列化、Monitor 长轮询下的已知 P0 风险点是否真的触发）：如实记录现象+可能原因，留给用户决定是否要开一个新任务专门修。

---

## 验证方式

- Phase 0/3 的命令输出即验证依据（不是"看代码猜"，是真实跑一遍）。
- 最终以 `runs/<run_id>/` 下产物是否完整、`orchestrator/run_pipeline.py` 打印的 `final_state=DONE` 作为"跑通"的判定标准；如果 `final_state=FAILED` 或进程直接崩溃，把完整 traceback/日志作为诊断材料附在汇报里。
- 不新增/修改 `tests/` 里的 fake 测试用例来"证明"框架可用——本次测试的意义就在于绕开 fakes，用真实二进制/真实数据/真实 GPU 验证。

计划批准后我会先把此文件复制一份到仓库当前目录（`uautoresearchX/`）方便你随时查阅，再按 Phase 0→5 顺序执行。
