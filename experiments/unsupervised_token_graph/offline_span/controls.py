"""困难对照只重接检测侧配对，不篡改原 attention 或新增真假标签。"""

from dataclasses import replace

import numpy as np

from .data import MatchedPair


def match_evidence_controls(graph, span, index):
    """同一材料内找词项、长度最接近的其他来源窗口；不随机跨话题拼接。"""
    if span.evidence_group < 0:
        return []
    original = set(graph.sample.token_ids[span.evidence_nodes].tolist())
    candidates = []
    for group, nodes in index.source_nodes.items():
        if group == span.evidence_group:
            continue
        if abs(len(nodes) - len(span.evidence_nodes)) > max(2, len(span.evidence_nodes) // 4):
            continue
        tokens = set(graph.sample.token_ids[nodes].tolist())
        similarity = len(original & tokens) / max(1, len(original | tokens))
        candidates.append((similarity, group, nodes))
    if not candidates:
        return []
    _, group, nodes = max(candidates, key=lambda item: (item[0], -item[1]))
    return [replace(span, evidence_nodes=nodes, evidence_group=group)]


def match_reuse_controls(graph, span, all_views):
    """保持整个段内与后文视图，替换同答相近长度/距离的选择根。"""
    candidates = []
    for other in all_views:
        if other.end - other.start != span.end - span.start:
            continue
        if other.start < span.end and span.start < other.end:
            continue
        distance = abs(other.start - span.start)
        if distance > 64 or other.evidence_group == span.evidence_group:
            continue
        candidates.append((distance, other))
    if not candidates:
        return []
    other = min(candidates, key=lambda item: item[0])[1]
    return [replace(span, selector_nodes=other.selector_nodes)]


def build_contrastive_pairs(graph, spans, index, maximum=128, seed=0):
    """随机采样原区间，不按风险挑选；两类对照均保留身份外的实际内容。"""
    random = np.random.default_rng(seed)
    chosen = random.choice(len(spans), min(maximum, len(spans)), replace=False)
    pairs = []
    by_length = {}
    for span in spans:
        by_length.setdefault(span.end - span.start, []).append(span)
    for number in chosen:
        span = spans[int(number)]
        controls = [('evidence', match_evidence_controls(graph, span, index)),
                    ('reuse', match_reuse_controls(graph, span, by_length[span.end - span.start]))]
        for name, alternatives in controls:
            if alternatives:
                pairs.append(MatchedPair(span, alternatives[0], name))
    return pairs
