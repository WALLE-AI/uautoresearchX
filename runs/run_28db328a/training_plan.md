# SVRDD_YOLO 路面病害检测（YOLO26n微调）- 训练计划

## TL;DR（人类速览）

- 任务：在北京五区（Fengtai/Chaoyang/Xicheng/Dongcheng/Haidian）道路病害数据集 SVRDD_YOLO 上，基于本地权重 yolo26n.pt 做 YOLO 目标检测微调，7类（Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole, Longitudinal Patch, Transverse Patch, Manhole Cover）。
- 性质：框架验收性质的真实全量数据端到端测试（train 6000 / val 1000），核心目标是跑通 train到val到评估到产出checkpoint 全链路且无未处理异常，而非追求高mAP。
- 起点权重：/home/dataset1/gaojing/xibeiyuan/models/yolo26n/yolo26n.pt（COCO预训练，5.5MB，nano级，本次指定不联网下载新权重）。
- 阶段序列：单次 model.train() 调用内完成主微调（含Mosaic/MixUp增强）到收尾稳定化（close_mosaic关闭强增强）两个逻辑阶段，不做独立的第二次resume训练，符合 Ultralytics 原生最佳实践，同时满足分阶段训练日历的可观察性要求。
- 训练引擎：Ultralytics YOLO（ultralytics Python包）。
- 资源：单卡GPU（CUDA_VISIBLE_DEVICES=0固定），imgsz=1024，batch=16，AMP混合精度。
- Epoch规划：总上限150 epoch（早停 patience=40，倒数20个epoch为close_mosaic收尾阶段），预计实际运行 6到10 小时。
- 数据格式定案：YOLO-txt（Ultralytics原生检测格式），以 Model-Selection 的硬性要求为准，覆盖 Dataset-Analysis 给出的 COCO / VOC-XML 候选格式（数据集本身已是YOLO-txt，无需转换）。
- 达标标准：全流程无未处理异常并到达DONE；runs/detect/train*/weights/ 下产出至少一个有效 best.pt/last.pt；model.val() 得到的整体 mAP50（或mAP50-95）非0即可，不要求收敛。

## 资源规划

| 项目 | 规划 |
|---|---|
| GPU | 1x GPU，CUDA_VISIBLE_DEVICES=0 固定物理GPU0，单卡训练，不使用DDP |
| 显存建议 | 大于等于12GB（推荐16GB以上，如RTX 3090/4080/V100/A100均可）；yolo26n nano级 加 imgsz=1024 加 AMP，预计batch=16时显存占用约6到10GB |
| CPU / DataLoader | workers=8（按主机核数调整，建议不超过物理核心数的50%，避免与主进程/其他任务争抢） |
| 存储 | 数据集7000张1024x1024 JPG 加 YOLO-txt标注（预估小于5GB，已在磁盘）；训练输出目录runs/detect/预留大于等于5GB（日志、可视化、last.pt/best.pt，nano权重约5到6MB/份，增长可忽略） |
| Batch Size | 16（imgsz=1024、AMP开启时的稳健取值；显存充裕可尝试32，正式训练前建议先跑1个mini-batch做显存探测） |
| 精度 | AMP混合精度（amp=True，FP16/FP32自动切换） |
| 训练引擎 | Ultralytics YOLO（ultralytics Python包，YOLO(...).train() API） |
| 并行策略 | 单卡，无需多卡/DDP |
| 断点续训 | 启用 resume=True 能力兜底：训练进程若因环境异常（如短暂OOM、磁盘IO抖动）中断，可从last.pt恢复，避免全量重跑 |

## Pipeline Stages（训练流程）

本任务风险清单显示：(1) COCO预训练权重与路面病害领域存在明显domain gap；(2) 47.6%目标为小目标、26.8%为极端细长目标（裂缝类），nano级检测头对其召回天然偏弱；(3) 需要避免旋转类增强破坏纵/横语义类别标签。综合 configs/training_pipeline_patterns.yaml 中的 CV-Finetune-EMA 模式（微调加EMA权重平滑，适用cv-detect），并结合 Dataset-Analysis 建议的训练末段关闭Mosaic/MixUp以稳定小目标收敛，裁剪为如下流程：不做独立预训练阶段（起点权重已是通用检测预训练权重，且任务明确要求以yolo26n.pt为起点、不联网下载），也不做人为拆分的多次resume训练，而是在同一次 model.train() 调用内通过 Ultralytics 原生 close_mosaic 参数实现主微调到收尾稳定化两个逻辑阶段——这是 Ultralytics 官方推荐的实现方式，比手动拆分两次训练更稳健（避免EMA/优化器状态在阶段切换时丢失）。knowledge_base 中未检索到 cv-detect 任务类型的历史成功案例，故本阶段划分主要依据任务本身特征、模式库基线与自身对 Ultralytics YOLO 训练机制的知识。

| 阶段 | 起点权重来源 | 训练目标 | 引擎 | 关键超参 | 预计耗时 |
|---|---|---|---|---|---|
| Stage 1: 主微调（Warmup加全量增强） | 基础模型 /home/dataset1/gaojing/xibeiyuan/models/yolo26n/yolo26n.pt（COCO预训练） | 从COCO通用检测知识迁移到路面病害领域，学习中大目标（如Manhole Cover）及初步裂缝/修补特征，缩小domain gap | Ultralytics YOLO (ultralytics) | imgsz=1024, batch=16, epochs=1到130(总150中前130), optimizer=AdamW, lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3, mosaic=1.0, mixup=0.05, hsv_h=0.015/hsv_s=0.7/hsv_v=0.4, degrees=0, flipud=0, fliplr=0.5, translate=0.1, scale=0.5, shear=0, perspective=0, amp=True, patience=40(早停,监控val mAP50-95), device=0 | 约6到9小时（early stopping patience=40可能提前触发，实际常短于上限） |
| Stage 2: 收尾稳定化（close_mosaic，同一次train()内自动衔接，非独立resume） | Stage 1 训练进程内部状态（同一次train()调用自动切换，非从独立checkpoint恢复） | 关闭Mosaic/MixUp强增强，使小目标(47.6%占比)与细长裂缝目标(26.8%占比)的定位边界在真实图像分布下收敛更稳定，减少最终checkpoint的边界框噪声 | Ultralytics YOLO (ultralytics, close_mosaic参数) | epoch 131到150(20个epoch), mosaic=0, mixup=0, 其余超参与lr调度延续Stage1的cosine尾段, patience同一早停计数器继续生效, degrees=0/flipud=0保持(避免破坏纵/横语义) | 约1到1.3小时（20 epoch，无mosaic增强数据加载更快） |

说明：若 Stage 1 阶段早停（patience=40）在到达 epoch 131 之前触发，则训练在触发点直接产出 best.pt/last.pt 并结束，Stage 2 的 close_mosaic 收尾不会被执行——这是预期行为（框架验收目标是流程健壮跑通加产出有效checkpoint，而非强制跑满全部阶段）。

## 数据格式

最终目标格式：YOLO-txt（Ultralytics原生检测标注格式）

依据 Model-Selection 的硬性要求（必须使用Ultralytics YOLO原生检测标注格式，不接受COCO json或VOC-XML作为训练输入格式），覆盖 Dataset-Analysis 候选格式清单中的 COCO / VOC-XML 选项。数据集在磁盘上已原生为 YOLO-txt 格式（class_id cx cy w h，[0,1]归一化），且已提供 train.txt/val.txt（相对路径）与 train_abs.txt/val_abs.txt（绝对路径）列表文件，无需任何格式转换，仅需组装 data.yaml。

字段映射规则（原始字段到目标格式字段）：

| 原始字段 | 目标字段 | 转换规则 |
|---|---|---|
| 标注行 class_id（0到6整数，SVRDD_YOLO原生标注） | data.yaml.names[class_id] 索引对应类别 | 恒等映射，顺序固定为 [0]Longitudinal Crack, [1]Transverse Crack, [2]Alligator Crack, [3]Pothole, [4]Longitudinal Patch, [5]Transverse Patch, [6]Manhole Cover，顺序错位将导致类别标签错乱，需在训练前核对 |
| 标注行 cx cy w h（[0,1]归一化中心点加宽高，SVRDD_YOLO原生标注） | Ultralytics训练输入的 cx cy w h | 恒等映射，无需反算像素坐标，直接复用原始归一化值 |
| train_abs.txt / val_abs.txt（图像绝对路径列表） | data.yaml.train / data.yaml.val | 直接引用为data.yaml的train/val字段值（本次不使用test集，即使test_abs.txt存在也不纳入训练配置） |
| 7类中文/英文类别名对照表 | data.yaml.names (list[str], 长度7) 与 data.yaml.nc (=7) | 按 class_id 0到6 顺序写入names列表；nc固定为7 |

## 训练日历（分阶段）

以任务启动日 2026-07-10 为 Day 0 估算（单卡，无排队等待假设）：

| 时间窗口 | 阶段 | 内容 | 里程碑产出 |
|---|---|---|---|
| Day 0, 0:00到0:30 | 准备 | 组装data.yaml（train/val路径加nc加names）、核对类别顺序、显存探测确定batch=16可行、加载本地yolo26n.pt（确认无联网下载行为） | data.yaml就绪，一次mini-batch前向/反向验证无OOM |
| Day 0, 0:30 到 Day 0约9:30 | Stage 1 主微调 | epoch 1到130（或early stopping提前触发），全量增强，监控train/val loss与val mAP50-95曲线 | 若干次best.pt更新（每次val mAP提升保存），last.pt持续更新 |
| （若未早停）Day 0约9:30到10:50 | Stage 2 收尾稳定化 | epoch 131到150，close_mosaic收尾 | 最终best.pt/last.pt |
| 训练结束后 0:15到0:30 | 验证与产出核对 | model.val()跑val集，输出整体mAP50/mAP50-95及per-class AP/Recall、混淆矩阵、PR曲线；核对checkpoint文件存在且可加载 | 验证报告 加 至少1个有效checkpoint确认 |

合计预计总耗时约 6.5到11 小时（含准备与验证），符合无严格总时长上限但避免不必要超大epoch空转的约束（150 epoch为上限，early stopping为主要收敛/停止机制）。

## 验证与达标标准

1. 无未处理异常跑到DONE：训练脚本对model.train()调用做try/except包裹，捕获并记录（而非静默吞掉）CUDA OOM、磁盘写满等异常；发生可恢复异常（如短暂OOM）时通过resume=True从last.pt续训，而非直接判定任务失败；训练进程正常结束或早停触发均视为DONE。
2. 至少一个有效checkpoint：训练结束后核查 runs/detect/train*/weights/best.pt 与 last.pt 均存在、文件大小与nano模型量级相符（约5到6MB量级）、可被 YOLO(path) 重新加载且能对至少1张val图像跑通推理。
3. mAP非0：对val集执行 model.val()，整体 mAP50 与 mAP50-95 至少一项显著大于0（不要求收敛或达到Model-Selection估计的0.55到0.68区间，该区间仅作参考基线）。
4. 辅助复核（非硬性DONE条件，但需记录）：
   - 输出per-class AP/Recall，重点关注 Pothole（794样本）、Alligator Crack（1546样本）两个稀有类是否出现召回率为0（若为0需在报告中标注为已知风险而非流程缺陷，对应 Scenario-Analysis 风险预判）；
   - 记录train/val loss曲线与PR曲线，供人工判断模型是否学到有效特征而非仅拟合高频类别；
   - 记录单帧推理耗时作为边缘/车载场景的参考指标（Scenario-Analysis中提出的隐含需求）。

## 决策依据与引用来源

- knowledge_base 检索结果：未找到与任务类型 cv-detect 匹配的历史成功案例（已确认为空），因此阶段序列与超参未直接复用历史案例，改为综合任务特征、模式库与自身知识裁剪得出。
- 训练流程模式库：configs/training_pipeline_patterns.yaml 中的 CV-Finetune-EMA（微调加EMA权重平滑，适用 cv-detect/cv-segment/cv-classify）作为基线，裁剪点：将微调加EMA落地为 Ultralytics 单次 train() 调用内含 close_mosaic 收尾的两阶段等效实现（EMA为Ultralytics训练默认内置机制，无需额外配置）；未采用 CV-Pretrain-Finetune 模式，因为本次已有可用的COCO预训练起点权重，无需从零预训练。
- 上游三方输出交叉引用：Scenario-Analysis风险预判（mAP偏低属预期、需warmup应对domain gap、需关注per-class指标、需断点续训兜底）均已在超参（warmup_epochs=3, lr0=0.001）与验证标准中落地；Dataset-Analysis的小目标(47.6%)/细长目标(26.8%)统计与增强建议（保留imgsz=1024原图、保守裁剪、禁用旋转/竖直翻转、close_mosaic收尾）已直接映射为 degrees=0, flipud=0, shear=0, perspective=0, fliplr=0.5 及 Stage 2 设计，类别不均衡(5.1:1)对应验证标准中的per-class复核要求；Model-Selection的起点权重路径、YOLO-txt硬性格式要求、GPU/显存/batch建议、8到15小时训练时长估计均已直接采用或细化到本计划的资源规划与训练日历中。
- 自身知识补充（未使用WebSearch/WebFetch，运行环境不可用）：Ultralytics YOLO 官方训练机制惯例——COCO预训练权重微调场景下适度降低lr0并配合warmup_epochs、使用close_mosaic在训练尾段关闭强增强以稳定小目标收敛、AMP混合精度与resume断点续训机制——均基于训练前已具备的通用知识给出，非本次检索获得。