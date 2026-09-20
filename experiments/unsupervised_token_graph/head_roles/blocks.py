"""Equal-width ordinary key blocks: fixed destinations, moved raw key content."""

import numpy as np


def ordinary_blocks(token_ids, query, excluded, width):
    ordinary = ~np.isin(token_ids[:query], excluded)
    changes = np.diff(np.r_[False, ordinary, False].astype(int))
    blocks = []
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        blocks.extend((left, left + width) for left in range(start, end - width + 1, width))
    return np.asarray(blocks, int).reshape(-1, 2)


def sample_swaps(blocks, count, seed):
    pairs = np.column_stack(np.triu_indices(len(blocks), 1))
    random = np.random.default_rng(seed)
    chosen = random.choice(len(pairs), min(count, len(pairs)), replace=False)
    return blocks[pairs[chosen]]


def permutation(length, pair):
    first, second = pair
    if first[1] - first[0] != second[1] - second[0]:
        raise ValueError("Local key swaps require equal block lengths")
    order = np.arange(length)
    order[first[0]:first[1]] = np.arange(*second)
    order[second[0]:second[1]] = np.arange(*first)
    return order
