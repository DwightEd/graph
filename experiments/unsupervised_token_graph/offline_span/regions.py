"""候选区间不依赖金标或高熵门槛；一次索引服务整条回答。"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .data import SpanView
from .graph import index_later_readers


@dataclass
class SpanIndex:
    source_nodes: dict
    source_ids: np.ndarray
    source_mass: sparse.csr_matrix
    later_readers: dict


def prepare_span_index(graph):
    group_ids = np.unique(graph.source_groups[graph.source_groups >= 0])
    groups = {int(group): np.flatnonzero(graph.source_groups == group) for group in group_ids}
    source, query = graph.edges[:, 1], graph.edges[:, 2]
    usable = source < graph.sample.prompt_length
    usable[usable] &= graph.source_groups[source[usable]] >= 0
    group_columns = np.searchsorted(group_ids, graph.source_groups[source[usable]])
    mass = sparse.csr_matrix((graph.weights[usable], (query[usable], group_columns)),
                             shape=(len(graph.sample.token_ids), len(group_ids)))
    return SpanIndex(groups, group_ids, mass, index_later_readers(graph))


def propose_spans(graph, max_length):
    """所有合法起点与长度；观测缺口保留为未覆盖，而不是补零后打分。"""
    spans = []
    for start in range(graph.sample.response_length):
        for end in range(start + 1, min(start + max_length, graph.sample.response_length) + 1):
            if not graph.coverage[end - 1]:
                break
            spans.append((start, end))
    return np.asarray(spans, dtype=np.int32).reshape(-1, 2)


def find_selector_candidates(graph, start, end, index):
    """保留首尾预测位置及区间内来源读取最强的位置，不将它称为已验证指针。"""
    queries = graph.sample.prompt_length + np.arange(start, end) - 1
    prompt_mass = np.asarray(index.source_mass[queries].sum(axis=1)).ravel()
    strongest = queries[int(np.argmax(prompt_mass))]
    return np.unique([queries[0], strongest, queries[-1]]).astype(np.int32)


def build_span_view(graph, start, end, index, future_budget):
    selectors = find_selector_candidates(graph, start, end, index)
    group_mass = np.asarray(index.source_mass[selectors].sum(axis=0)).ravel()
    if len(group_mass) and group_mass.max() > 0:
        group = int(index.source_ids[np.argmax(group_mass)])
        evidence = index.source_nodes[group]
    else:
        group = -1
        evidence = np.empty(0, np.int32)

    response = graph.sample.prompt_length + np.arange(start, end)
    readers = []
    for token in response:
        readers.extend(index.later_readers.get(int(token), ()))
    readers = np.asarray(sorted(set(readers)), np.int32)
    readers = readers[readers >= graph.sample.prompt_length + end][:future_budget]
    return SpanView(start, end, evidence, selectors, response.astype(np.int32), readers, group)


def build_views(graph, max_length=32, future_budget=8):
    index = prepare_span_index(graph)
    views = [build_span_view(graph, int(start), int(end), index, future_budget)
             for start, end in propose_spans(graph, max_length)]
    return views, index
