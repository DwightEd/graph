"""利用整段关系得分联合定位；后文可影响早期评分，但不是早期预测。"""

import numpy as np

from .data import DetectionResult, SpanView, TokenGraph
from .model import EvidenceSpanScorer


def score_spans(
    model: EvidenceSpanScorer,
    graph: TokenGraph,
    spans: list[SpanView],
) -> np.ndarray:
    """把每个区间的负匹配 logit 作为原始关系异常分数。"""
    raise NotImplementedError


def fit_unlabeled_reference(
    scores: np.ndarray,
    lengths: np.ndarray,
    source_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """按长度、来源平衡校准混合样本分数；不筛选已知正确回答。"""
    raise NotImplementedError


def decode_spans(
    graph: TokenGraph,
    spans: list[SpanView],
    scores: np.ndarray,
    reference: dict[str, np.ndarray],
    boundary_penalty: float,
) -> DetectionResult:
    """用片段能量联合选择边界，保留单词错误及整答不报警的解。"""
    raise NotImplementedError
