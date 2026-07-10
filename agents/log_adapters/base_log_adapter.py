"""LogAdapter统一接口：把三种日志形态（local/wandb/swanlab）归一化为同一份
六字段指标，供Monitor Agent读取。不负责GPU信息采集（由Monitor Agent自己调
`nvidia-smi`），也不负责跨轮次的趋势累积（由Monitor Agent自己写`metrics.csv`
逐轮积累）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NormalizedMetrics:
    """六字段归一化指标，任意字段在无法从日志中获取时为None。"""

    epoch: float | None = None
    loss: float | None = None
    metric: float | None = None
    speed: float | None = None
    memory: float | None = None
    checkpoint_path: str | None = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (self.epoch, self.loss, self.metric, self.speed, self.memory, self.checkpoint_path)
        )


class BaseLogAdapter(ABC):
    """所有日志形态适配器的统一接口。"""

    @abstractmethod
    def read_latest_metrics(self, log_dir: Path) -> NormalizedMetrics:
        """从`log_dir`读取当前最新一次可用的指标，归一化为六字段。

        `log_dir`语义因适配器而异：
            local  -> 包含`train.log`的目录
            wandb  -> 包含`wandb/`同步目录的父目录
            swanlab-> 包含swanlab本地同步目录的父目录
        读取失败/文件不存在时返回全None的`NormalizedMetrics`，不抛异常
        （Monitor Agent的LLM分析阶段仍可基于"暂无可用指标"这一事实继续工作）。
        """
        raise NotImplementedError
