"""独立检测器学习来源与片段是否对应；不重构节点/边，不更新原 LLM。"""

from torch import Tensor, nn

from .data import SpanView, TokenGraph


class EvidenceSpanScorer(nn.Module):
    """分别编码证据与选择—复用视图，再学习它们的联合匹配。"""

    def __init__(self, node_size: int, message_size: int, hidden_size: int):
        """后续仅在此定义检测端的投影和关系评分层。"""
        raise NotImplementedError

    def encode_evidence(self, graph: TokenGraph, span: SpanView) -> Tensor:
        """保留对象、条件及取值的上下文，不输入绝对来源编号。"""
        raise NotImplementedError

    def encode_reuse(self, graph: TokenGraph, span: SpanView) -> Tensor:
        """读取整段与后续关联节点，逐关系/head计算后再汇总。"""
        raise NotImplementedError

    def forward(self, graph: TokenGraph, span: SpanView) -> Tensor:
        """输出原始配对相对重接对照的匹配 logit，不是假话概率。"""
        raise NotImplementedError
