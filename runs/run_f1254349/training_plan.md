# SVRDD 道路路面病害检测 - 训练计划

## TL;DR（人类速览）
- **任务类型**：CV 目标检测（7 类别道路病害）
- **推荐模型**：YOLOv8n（轻量实时）
- **起点权重**：YOLOv8n 官方预训练权重（COCO 预训练）
- **流程模式**：CV-Finetune （单阶段微调）
- **输入分辨率**：320×320（快速验证）
- **预计耗时**：2-4 小时（A100, 1k-10k 样本）
- **预期指标**：mAP@50≈0.82-0.87, mAP@50-95≈0.55-0.62

## 资源规划
| 资源项 | 规划 |
|--------|------|
| GPU | 1× A100-40GB 或 1× RTX 3090/4090-24GB |
| 显存 | ~12-16GB (batch_size=32, 320×320) |
| CPU | 8 核以上 |
| 存储 | 50GB+（数据集+权重+ checkpoint）|
| 训练时长 | 2-4 小时（1k-10k 样本）|

## Pipeline Stages（训练流程）

### Stage 1: YOLOv8n 微调
- **起点权重**：YOLOv8n 官方 COCO 预训练权重
- **目标**：在 SVRDD 道路病害数据集上监督微调，适配 7 类别检测
- **引擎**：ultralytics (YOLOv8 Training API)
- **关键超参**：
  - batch_size: 32
  - epochs: 100
  - lr0: 0.01 (初始学习率)
  - lrf: 0.01 (最终学习率)
  - optimizer: SGD with momentum
  - weight_decay: 0.0005
  - image_size: 320
  - augmentation: Mosaic(1.0), MixUp(0.1), Copy-Paste(0.1)
- **预计耗时**：2-4 小时（1× A100）

## 数据格式

### 最终目标格式：YOLO-txt
**选择理由**：
1. Model-Selection 硬性要求 YOLOv8 原生支持 YOLO-txt 格式
2. Dataset-Analysis 候选格式中 YOLO-txt 是唯一推荐
3. YOLOv8n 与 YOLO-txt 完美兼容，无需格式转换

### 字段映射规则
| 原始字段 | 目标字段 | 转换规则 |
|----------|----------|----------|
| image_path | img_path / img_id | 直接映射，img_id 提取文件名 |
| bbox_2d (x1,y1,x2,y2) | class_id x_center y_center width height | 转换为归一化坐标 (0-1)：
- x_center = (x1+x2)/2 / img_width
- y_center = (y1+y2)/2 / img_height
- width = (x2-x1) / img_width
- height = (y2-y1) / img_height
| 病害类别名称 | class_id | 映射到 0-6：裂缝=0, 坑槽=1, 松散=2, 龟裂=3, 坑凼=4, 沉陷=5, 修补=6 |

## 训练日历（分阶段）

| 阶段 | 内容 | 时长 | 产出 |
|------|------|------|------|
| Day 1 | 数据准备 + 格式转换 + 验证集划分 | 4-8 小时 | 标准化的 YOLO-txt 数据集 |
| Day 2 | Stage 1 微调训练（100 epochs） | 2-4 小时 | final.weights, last.weights |
| Day 3 | 模型验证 + 指标评估 + 推理测试 | 2-4 小时 | 评估报告+推理 demo |

## 验证与达标标准
- **主要指标**：mAP@50 ≥ 0.80
- **次要指标**：mAP@50-95 ≥ 0.50, Inference FPS ≥ 30 (320×320, T4)
- **验收标准**：验证集 mAP@50 达到预期区间 0.82-0.87，推理速度满足实时性要求

## 决策依据与引用来源
- **模式选择**：CV-Finetune（基于 Model-Selection 推荐 YOLOv8n 单阶段微调方案）
- **模型选择**：YOLOv8n（社区生态成熟、推理速度快、支持小分辨率输入）
- **数据格式**：YOLO-txt（Model-Selection 硬性要求 + Dataset-Analysis 候选推荐）
- **无历史案例**：knowledge_base 中未找到 cv-detect 类型案例，基于领域知识补充