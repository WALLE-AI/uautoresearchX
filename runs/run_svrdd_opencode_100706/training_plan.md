# SVRDD Road Defect Detection - Training Plan

## TL;DR（人类速览）
在 100 个 SVRDD 路面病害样本上，使用 YOLO26n 轻量模型进行 2 epoch 快速微调，验证 ultralytics CLI 命令（run/list/status/logs/cancel/resume）功能。采用 AMP 混合精度节省显存、lr=0.001 微调学习率、256 分辨率快速迭代。

## 资源规划
- GPU: 1x A100 40GB
- Batch Size: -1 (自动最优，预计 16-32)
- Epochs: 2 (功能验证)
- Precision: AMP (混合精度)
- Storage: 用于数据、日志、checkpoint 约 2GB

## Pipeline Stages（训练流程）
- CV-Finetune-EMA
- Stage 1: Baseline Detection Training
  - Start From: /home/dataset1/gaojing/xibeiyuan/models/yolo26n/yolo26n.pt (COCO 80 类预训练)
  - Goal: CLI 功能验证，输出至少一个 checkpoint，mAP > 0
  - Engine: ultralytics (yolo detect train)
  - Key Hyperparams: epochs=2, lr0=0.001, amp=True, imgsz=256
  - Duration: 2-5 分钟
- Decision References: [知识补充] ultralytics YOLO 微调最佳实践（官方文档推荐lr=0.001、使用AMP）

## 数据格式
- Target Format: YOLO-txt
- Rationale: Model-Selection 硬性要求 ultralytics 标准格式，Dataset-Analysis 已满足此格式
- Field Mapping:
  - Source: image_name.txt (YOLO 标准格式：class_id x_center y_center w h)
  - Target: ultralytics YOLO txt 格式（每行归一化坐标）
  - Rule: 直接使用，无需转换。data.yaml 中定义 classes=7 和.names 列表匹配任务 7 类病害
- Note: COCO 80 类权重作为初始化，训练时映射到 7 类目标类别，ultralytics 自动处理类别迁移

## 训练日历（分阶段）
- 0-5 分钟: Stage 1 (baseline)，2 epoch，CLI 验证
- 后续迭代: 全量数据集 + 8 倍 imgsz + 10-20 epochs → mAP 持续提升

## 验证与达标标准
- 成功条件: 无未处理异常、至少 1 个 checkpoint 文件、mAP > 0
- 失败条件: 显存溢出、训练崩溃 2 轮以上

## 决策依据与引用来源
- 模型选择: YOLO26n 参数量最小，适合快速验证
- 学习率修正: lr=0.001 而非 0.01，基于 ultralytics 官方微调建议
- 精度修正: AMP 混合精度节省显存，适合单 GPU 训练
- 数据迁移理解: 使用 COCO 80 类预训练权重初始化，ultralytics 自动处理类别对齐到 7 类目标