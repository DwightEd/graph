"""只重接检测器的证据/复用配对，不改写或伪装原模型的物理运算。"""

from .data import MatchedPair, SpanView, TokenGraph


def match_evidence_controls(graph: TokenGraph, span: SpanView) -> list[SpanView]:
    """寻找同材料、角色和规模接近的来源组合，避免只靠话题差异识别对照。"""
    raise NotImplementedError


def match_reuse_controls(graph: TokenGraph, span: SpanView) -> list[SpanView]:
    """保持后续链内部结构，将其与另一相近选择根配对。"""
    raise NotImplementedError


def build_contrastive_pairs(graph: TokenGraph, spans: list[SpanView]) -> list[MatchedPair]:
    """统一形成原始/重接对照；原始不等于正确，重接不等于已知幻觉。"""
    raise NotImplementedError
