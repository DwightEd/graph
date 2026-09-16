"""先提出未知真假的区间，再读取整段及其后文；不使用金标边界。"""

import numpy as np

from .data import SpanView, TokenGraph


def propose_spans(graph: TokenGraph, max_length: int) -> np.ndarray:
    """枚举长度预算内的连续区间，保留单 token，不要求先有高熵入口。"""
    raise NotImplementedError


def find_selector_candidates(graph: TokenGraph, start: int, end: int) -> np.ndarray:
    """用原生读取关系提出选择位置；它们只是候选，不宣称是纯指针。"""
    raise NotImplementedError


def build_span_view(
    graph: TokenGraph,
    start: int,
    end: int,
    later_readers: dict[int, np.ndarray],
    future_budget: int,
) -> SpanView:
    """同时保留片段内部复用、外部证据及后文中的支持、切换或纠正线索。"""
    raise NotImplementedError
