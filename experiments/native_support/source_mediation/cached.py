"""Use measured finite cuts; never infer missing joint or sham-single experiments."""

import numpy as np

from .decomposition import CACHED_TOKENS, gate_scores


def measured_blocks(reader, directory, views, token_mode):
    if token_mode:
        for target in range(len(views["answer_ids"])):
            saved = reader.arrays(f"{directory}/token_{target:06d}.npz")
            values = {name: saved[name][..., 0, None] for name in ("keep", "joint", "sham", "single")}
            yield np.array([target]), saved, values
    else:
        for unit in views["units"]:
            saved = reader.arrays(f"{directory}/unit_{unit['start']:06d}/measurements.npz")
            yield np.arange(unit["start"], unit["stop"]), saved, saved


def score_cached(reader, directory, views, token_mode):
    count = len(views["answer_ids"])
    scores = {name: np.empty(count) for name in CACHED_TOKENS}
    components = {"selected_edge_count": np.zeros(count, dtype=int)}
    edges = []
    for targets, saved, values in measured_blocks(reader, directory, views, token_mode):
        if not np.array_equal(np.atleast_1d(saved["target"]), targets):
            raise ValueError("Carrier target identities differ")
        actual = np.asarray(views["answer_ids"])[targets]
        if not np.array_equal(np.atleast_1d(saved["token_id"]), actual):
            raise ValueError("Carrier token identities differ")
        block_scores, block = gate_scores(*(values[name] for name in ("keep", "joint", "sham", "single")))
        for name, value in block_scores.items():
            scores[name][targets] = value
        for name, value in block.items():
            if value.ndim == 1:
                components.setdefault(name, np.empty(count))[targets] = value
        edge_ids = saved["edges"]
        components["selected_edge_count"][targets] = len(edge_ids)
        for index, edge in enumerate(edge_ids):
            layer, head, key = int(edge[0]), int(edge[1]), int(edge[-1])
            receiver = int(edge[2]) if token_mode else -1
            for column, target in enumerate(targets):
                edges.append((target, layer, head, receiver, key,
                              *block["single_effect"][:, index, column],
                              block["head_interaction"][index, column]))
    columns = ("target", "layer", "head", "receiver", "key", "with_source", "without_source", "interaction")
    matrix = np.asarray(edges, dtype=float).reshape(-1, len(columns))
    sparse = {name: matrix[:, index].astype(int if index < 5 else float)
              for index, name in enumerate(columns)}
    return scores, components, sparse
