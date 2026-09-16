"""原始/重接配对的自监督学习；没有自然幻觉标签或金标片段。"""

from pathlib import Path

from torch import Tensor

from .data import MatchedPair
from .model import EvidenceSpanScorer


def contrastive_loss(observed_logits: Tensor, reconnected_logits: Tensor) -> Tensor:
    """平衡两类配对，学习条件相容性，不计算特征重构误差。"""
    raise NotImplementedError


def train_scorer(
    model: EvidenceSpanScorer,
    pairs: list[MatchedPair],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> EvidenceSpanScorer:
    """只更新检测器；正式实现按样本小批读取，不将全部原图放入显存。"""
    raise NotImplementedError


def save_scorer(model: EvidenceSpanScorer, output: Path) -> None:
    """保存检测器权重和实际训练配置，不另建复杂身份封装。"""
    raise NotImplementedError
