"""Each control changes one input/module; no gold labels are inspected."""

import numpy as np


# Model-side switches are explicit in model.py, not a feature-generation pipeline.
MODEL_ABLATIONS = ('full', 'no_graph', 'no_node', 'no_edge', 'no_source',
                   'no_mark', 'no_residual', 'no_relay', 'one_layer')
GRAPH_ABLATIONS = ('no_prompt', 'no_history', 'local', 'rewire',
                   'head_mean', 'head_names', 'coupled_heads', 'independent_heads')


def groups(graph):
    source, target = graph['edge_index']
    role = source >= int(graph['prompt_length'])
    lag = np.ceil(np.log2(target - source)).astype(int)
    _, group = np.unique(np.column_stack((target, role, lag)), axis=0, return_inverse=True)
    order = np.argsort(group, kind='stable')
    return np.split(order, np.flatnonzero(np.diff(group[order])) + 1) if len(order) else []


def remove_edges(graph, keep):
    return dict(graph, edge_index=graph['edge_index'][:, keep],
                edge_attr=graph['edge_attr'][keep], edge_mark=graph['edge_mark'][keep])


def permute_heads(graph, independent, rng):
    """Keep each head's weights within target/role/lag; vary joint endpoint pairing."""
    values = graph['edge_attr'].copy()
    for indices in groups(graph):
        if independent:
            for head in range(values.shape[1]):
                values[indices, head] = graph['edge_attr'][rng.permutation(indices), head]
        else:
            values[indices] = graph['edge_attr'][rng.permutation(indices)]
    return dict(graph, edge_attr=values)


def move_history(graph, local, rng):
    """Same historical coarse lag bands as the old rewire_in control."""
    source, target = graph['edge_index']
    prompt = int(graph['prompt_length'])
    moved = source.copy()
    for query in np.unique(target[source >= prompt]):
        edges = np.flatnonzero((target == query) & (source >= prompt))
        if local:
            moved[edges] = np.arange(query - len(edges), query)
            continue
        band = np.searchsorted([1, 4, 16, 64], query - source[edges], side='left')
        candidates = np.arange(prompt, query)
        candidate_band = np.searchsorted([1, 4, 16, 64], query - candidates, side='left')
        for value in np.unique(band):
            selected = edges[band == value]
            moved[selected] = rng.choice(candidates[candidate_band == value], len(selected), replace=False)
    return dict(graph, edge_index=np.stack((moved, target)))


def change_graph(graph, name, seed):
    rng = np.random.default_rng(seed)
    source = graph['edge_index'][0]
    prompt = int(graph['prompt_length'])
    if name == 'no_prompt':
        return remove_edges(graph, source >= prompt)
    if name == 'no_history':
        return remove_edges(graph, source < prompt)
    if name in ('local', 'rewire'):
        return move_history(graph, name == 'local', rng)
    if name in ('coupled_heads', 'independent_heads'):
        return permute_heads(graph, name == 'independent_heads', rng)
    if name == 'head_names':
        heads, layers = int(graph['heads']), int(graph['layers'])
        order = np.concatenate([layer * heads + rng.permutation(heads) for layer in range(layers)])
        return dict(graph, x=graph['x'][:, order], edge_attr=graph['edge_attr'][:, order])
    if name == 'head_mean':
        def collapse(values):
            shape = values.shape
            by_head = values.reshape(-1, int(graph['layers']), int(graph['heads']))
            return np.broadcast_to(by_head.mean(axis=2, keepdims=True), by_head.shape).reshape(shape).copy()
        return dict(graph, x=collapse(graph['x']), edge_attr=collapse(graph['edge_attr']))
    return graph


def topology_change(original, changed):
    """Changing edge slots is NOT the same as changing the adjacency set."""
    prompt = int(original['prompt_length'])
    def rr_edges(graph):
        return {tuple(pair) for pair in graph['edge_index'].T if pair[0] >= prompt}
    before, after = rr_edges(original), rr_edges(changed)
    count = len(before)
    removed = len(before - after)
    slot_changes = None
    if original['edge_index'].shape == changed['edge_index'].shape:
        slot_changes = int(np.any(original['edge_index'] != changed['edge_index'], axis=0).sum())
    return dict(rr_edges=count, rr_removed=removed, rr_added=len(after - before),
                removed_fraction=removed / count if count else None, changed_edge_slots=slot_changes)
