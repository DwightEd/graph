"""逐头保留实际 attention 端点；检测端层序聚合不冒充 WV/WO 原生信息流。"""

import numpy as np

from ..channels import iter_channels
from .data import TokenGraph


def retain_partition_edges(rows, columns, weights, prompt_length, budget):
    """每行分别保留 prompt/history 的强边；记录原权重，不把剩余边归一化。"""
    if budget == 0:
        return np.arange(len(weights))
    groups = rows * 2 + (columns >= prompt_length)
    order = np.lexsort((columns, -weights, groups))
    sorted_groups = groups[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_groups))]
    ranks = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    return order[ranks < budget]


def attach_source_context(sample, observations, width):
    """优先原记录组；否则使用明确标为近似的 prompt token 窗口，不声称已抽取命题。"""
    prompt_length = sample.prompt_length
    supplied = observations['source_groups']
    groups = np.arange(prompt_length) // width if supplied is None else np.asarray(supplied, dtype=int).copy()
    if groups.shape != (prompt_length,):
        raise ValueError('source_groups must align with prompt tokens')
    if observations['source_mask'] is not None:
        mask = np.asarray(observations['source_mask'], dtype=bool)
        if mask.shape != groups.shape:
            raise ValueError('source_mask must align with prompt tokens')
        groups[~mask] = -1
    return groups


def build_token_graph(sample, observations, edges_per_partition=2, context_width=32, layers=None, heads=None):
    channels, all_edges, all_weights, all_masses = [], [], [], []
    coverage = np.zeros(sample.response_length, bool)
    seen = set()
    prompt_length = sample.prompt_length

    for record in observations['records']:
        for channel in iter_channels(record, layers, heads):
            identity = (channel.layer, channel.head)
            if identity in seen:
                raise ValueError(f'duplicate physical channel {identity} for answer {sample.response_id}')
            seen.add(identity)
            channels.append(identity)
            matrix = channel.attention
            rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
            columns = matrix.indices
            weights = matrix.data
            query_positions = channel.queries[rows]
            prediction_positions = channel.queries + 1 - prompt_length
            valid_rows = (prediction_positions >= 0) & (prediction_positions < sample.response_length)
            coverage[prediction_positions[valid_rows]] = True

            prompt_mass = np.bincount(rows, weights * (columns < prompt_length), minlength=matrix.shape[0])
            history_mass = np.bincount(rows, weights * (columns >= prompt_length), minlength=matrix.shape[0])
            masses = np.full((sample.response_length, 4), np.nan, np.float32)
            row_total = prompt_mass + history_mass
            masses[prediction_positions[valid_rows], :3] = np.column_stack(
                (prompt_mass, history_mass, np.maximum(0, 1 - row_total)))[valid_rows]

            if len(weights):
                retained = retain_partition_edges(rows, columns, weights, prompt_length, edges_per_partition)
                retained_mass = np.bincount(rows[retained], weights[retained], minlength=matrix.shape[0])
                channel_ids = np.full(len(retained), len(channels) - 1)
                all_edges.append(np.column_stack((channel_ids, columns[retained], query_positions[retained])))
                all_weights.append(weights[retained])
            else:
                retained_mass = np.zeros(matrix.shape[0])
            masses[prediction_positions[valid_rows], 3] = (row_total - retained_mass)[valid_rows]
            all_masses.append(masses)

    if not channels:
        raise ValueError('requested physical channels are absent from this answer')
    edges = np.concatenate(all_edges).astype(np.int32) if all_edges else np.empty((0, 3), np.int32)
    weights = np.concatenate(all_weights).astype(np.float32) if all_weights else np.empty(0, np.float32)
    groups = attach_source_context(sample, observations, context_width)
    return TokenGraph(sample, np.asarray(channels, np.int32), edges, weights,
                      np.stack(all_masses), coverage, groups, observations['node_features'],
                      observations['entropy'], 'hidden' if observations['node_features'].shape[1] else 'tokens')


def index_later_readers(graph):
    """原生 j→q 保持不变，另建 j→{较晚q} 的只读索引，供离线检测取上下文。"""
    readers = {}
    for source, query in np.unique(graph.edges[:, 1:3], axis=0):
        if query > source:
            readers.setdefault(int(source), []).append(int(query))
    return {source: np.asarray(queries, np.int32) for source, queries in readers.items()}
