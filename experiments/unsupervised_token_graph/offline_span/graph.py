"""保留实际读取边，另建离线分析连接；不模拟反向因果传播。"""

import numpy as np

from .data import Sample, TokenGraph


def build_token_graph(sample: Sample, observations: dict[str, np.ndarray]) -> TokenGraph:
    """按 token、层和计算阶段建立原生图，保留 head 身份和缺失质量。"""
    raise NotImplementedError


def attach_source_context(graph: TokenGraph) -> TokenGraph:
    """接入被读词附近的原始记录和限定上下文，不将同句关系当成真值。"""
    raise NotImplementedError


def index_later_readers(graph: TokenGraph) -> dict[int, np.ndarray]:
    """找到后面哪些节点读取了当前节点；返回索引，不添加反向原生边。"""
    raise NotImplementedError
