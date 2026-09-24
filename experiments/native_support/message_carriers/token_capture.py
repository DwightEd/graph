"""Freeze a token's alternative and history edges, then intervene at exact queries."""

import numpy as np

from .token_native import deleted_values, observe_token
from .token_selection import candidate_table, select_edges


def empty_selection(model):
    return dict(edges=np.empty((0, 4), dtype=np.int64), sham_edges=np.empty((0, 4), dtype=np.int64),
        approximation=np.empty((0, 2)), sham_approximation=np.empty((0, 2)),
        route_mass=np.empty((0, 2)), message_energy=np.empty((0, 2)),
        sham_route_mass=np.empty((0, 2)), sham_message_energy=np.empty((0, 2)),
        importance=np.empty(0), candidate_count=0,
        reconstruction_error=np.zeros((2, len(model.layers))), reconstruction_measured=False)


def measure_cuts(model, prompt, answer, target, foil, selected, keep):
    edges = selected["edges"]
    single = [deleted_values(model, prompt, answer, target, foil, [edge]) for edge in edges]
    if not len(edges):
        return dict(single=np.empty((0, 2)), joint=keep.copy(), sham=keep.copy())
    joint = single[0] if len(edges) == 1 else deleted_values(model, prompt, answer, target, foil, edges)
    sham = deleted_values(model, prompt, answer, target, foil, selected["sham_edges"])
    return dict(single=np.stack(single), joint=joint, sham=sham)


def capture_token(model, views, target, top_k, receiver_budget, selection, seed):
    prompts = [views[f"prompt_{name}"] for name in ("with_source", "without_source")]
    answer = views["answer_ids"]
    if max(map(len, prompts)) + target > model.native.config.max_position_embeddings:
        raise ValueError("Observed prefix exceeds model context; no truncation")
    first = observe_token(model, prompts[0], answer, target)
    second = observe_token(model, prompts[1], answer, target, first["foil"])
    conditions = [first, second]
    if target:
        table = candidate_table(model, conditions, target, receiver_budget)
        selected = select_edges(table, top_k, seed, selection)
        del table
    else:
        selected = empty_selection(model)
    keep = np.stack([condition["values"] for condition in conditions])
    foil = first["foil"]
    del first, second, conditions
    cuts = [measure_cuts(model, prompt, answer, target, foil, selected, keep[index])
            for index, prompt in enumerate(prompts)]
    return dict(**selected, keep=keep, single=np.stack([cut["single"] for cut in cuts]),
        joint=np.stack([cut["joint"] for cut in cuts]), sham=np.stack([cut["sham"] for cut in cuts]),
        target=np.asarray(target), token_id=np.asarray(answer[target]), foil_id=np.asarray(foil),
        prompt_lengths=np.asarray(list(map(len, prompts))),
        head_shape=np.asarray([len(model.layers), model.head_layout(0)[0]]))
