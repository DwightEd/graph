"""Finite DAG paths and head/word/lag matched edge rewiring."""
import numpy as np


def coordinate_projection(width=128, rank=4):
    matrix = np.random.default_rng(42).normal(size=(width, rank))
    return np.linalg.qr(matrix)[0].astype(np.float32)


def matched_mappings(token_ids, prompt, source_mask, heads=32):
    """Same key ID, domain and floor(log2(lag)); lag1 and future keys fixed."""
    token_ids = np.asarray(token_ids)
    source_mask = np.asarray(source_mask, bool)
    domain = np.ones(len(token_ids), int)
    domain[:prompt][source_mask] = 0
    domain[prompt:] = 2
    generator = np.random.default_rng(73)
    mappings, movable = [], []
    for target in range(len(token_ids) - prompt):
        query = prompt + target - 1
        keys = np.arange(query + 1)
        lag = query - keys + 1
        bins = np.floor(np.log2(lag)).astype(int)
        groups = {}
        for key in keys:
            identity = (int(token_ids[key]), int(domain[key]), int(bins[key]))
            groups.setdefault(identity, []).append(int(key))
        mapping = np.tile(np.arange(len(token_ids)), (heads, 1))
        can_move = np.zeros(len(token_ids), bool)
        for group in groups.values():
            if len(group) < 2:
                continue
            order = generator.random((heads, len(group))).argsort(1)
            mapping[:, group] = np.asarray(group)[order]
            can_move[group] = True
        mappings.append(mapping)
        movable.append(can_move)
    return mappings, movable


def rooted_paths(source, weights, discount=.5):
    """R=(I-discount*W)^(-1)S; arrays [T,H,D], [T,H,T]."""
    result = source.copy()
    for target in range(len(source)):
        result[target] += discount * np.einsum(
            'hj,jhd->hd', weights[target, :, :target], result[:target])
    return result
