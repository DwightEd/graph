"""定义无标签输入、原生图和片段视图；本文件不读取标注。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    """原始材料和完整回答；offsets 对应回答字符，不包含幻觉标签。"""

    response_id: str
    source_id: str
    task: str
    split: str
    evidence_text: str
    response_text: str
    token_ids: np.ndarray
    prompt_length: int
    offsets: np.ndarray


@dataclass
class TokenGraph:
    """原生边和检测用连接分开保存，坐标只用于组图、不输入配对评分器。"""

    sample: Sample
    node_features: np.ndarray
    node_coordinates: np.ndarray
    native_edges: np.ndarray
    native_attributes: dict[str, np.ndarray]
    analysis_links: np.ndarray
    analysis_link_types: list[str]


@dataclass
class SpanView:
    """回答区间 [start, end) 及其证据、选择位置、后续读取节点。"""

    start: int
    end: int
    evidence_nodes: np.ndarray
    selector_nodes: np.ndarray
    response_nodes: np.ndarray
    later_nodes: np.ndarray
    native_edge_ids: np.ndarray
    analysis_link_ids: np.ndarray


@dataclass
class MatchedPair:
    """同一张图中的原始配对和重接对照；control_kind 不进入模型。"""

    graph: TokenGraph
    observed: SpanView
    reconnected: SpanView
    control_kind: str


@dataclass
class DetectionResult:
    """离线分数及片段边界；分数不是已校准的事实错误概率。"""

    response_id: str
    token_scores: np.ndarray
    span_bounds: np.ndarray
    span_scores: np.ndarray
    covered_tokens: np.ndarray


def load_samples(cache_root: Path) -> list[Sample]:
    """复用现有样本接口和原 offsets，绝不载入 labels 字段。"""
    raise NotImplementedError


def load_observations(sample: Sample, cache_root: Path) -> dict[str, np.ndarray]:
    """读取已存在的逐头 attention 和表征；缺少的观测不伪造。"""
    raise NotImplementedError
