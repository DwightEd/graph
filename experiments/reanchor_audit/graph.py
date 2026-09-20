"""Sparse layer-ordered Doob conditioning; capacities are descriptive proxies."""

import numpy as np


def capacities(graph, kind):
    mass = graph["attention"]
    heads = mass.shape[1]
    if kind == "attention":
        return mass / (heads + 1), np.ones(graph["residual_norm"].shape) / (heads + 1)
    projected = np.take_along_axis(graph["projected_value_norm"][:, :, None, :],
                                   graph["source"], axis=-1)
    denominator = graph["full_message_norm_sum"] + graph["residual_norm"]
    edges = mass * projected / denominator[:, None, :, None]
    return edges, graph["residual_norm"] / denominator


def backward_potential(source, capacity, residual, target):
    layers, _, length, _ = source.shape
    potential = np.zeros((layers + 1, length), dtype=np.float64)
    potential[-1, target] = 1.
    for layer in reversed(range(layers)):
        weights = capacity[layer] * potential[layer + 1][None, :, None]
        arriving = np.bincount(source[layer].ravel(), weights=weights.ravel(), minlength=length)
        potential[layer] = arriving + residual[layer] * potential[layer + 1]
    return potential


def conditioned_flow(source, capacity, residual, target, seed_mode="potential"):
    potential = backward_potential(source, capacity, residual, target)
    nodes = np.zeros_like(potential)
    roots = potential[0] if seed_mode == "potential" else (potential[0] > 0).astype(float)
    if roots.sum() == 0:
        return dict(potential=potential, nodes=nodes, edges=np.zeros_like(capacity),
                    residual=np.zeros_like(residual), reachable=False)
    nodes[0] = roots / roots.sum()
    edge_flow = np.zeros_like(capacity, dtype=np.float64)
    residual_flow = np.zeros_like(residual, dtype=np.float64)
    for layer in range(len(residual)):
        factor = np.divide(nodes[layer], potential[layer], out=np.zeros_like(nodes[layer]),
                           where=potential[layer] > 0)
        edge_flow[layer] = factor[source[layer]] * capacity[layer] * potential[layer + 1][None, :, None]
        residual_flow[layer] = factor * residual[layer] * potential[layer + 1]
        nodes[layer + 1] = edge_flow[layer].sum(axis=(0, 2)) + residual_flow[layer]
    return dict(potential=potential, nodes=nodes, edges=edge_flow, residual=residual_flow, reachable=True)


def analyze_graph(graph):
    result = {}
    for kind in ("attention", "message_norm"):
        capacity, residual = capacities(graph, kind)
        for seed_mode in ("potential", "uniform"):
            flow = conditioned_flow(graph["source"], capacity, residual,
                                    len(graph["prefix_ids"]) - 1, seed_mode)
            for name, value in flow.items():
                result[kind + "_" + seed_mode + "_" + name] = value
    return result


def flow_checks(flow):
    rows = []
    for kind in ("attention", "message_norm"):
        for seed in ("potential", "uniform"):
            key = kind + "_" + seed + "_"
            nodes = flow[key + "nodes"]
            rows.append(dict(capacity=kind, seed=seed, reachable=bool(flow[key + "reachable"]),
                sink_flow=float(nodes[-1, -1]),
                max_layer_mass_error=float(np.abs(nodes.sum(axis=1) - nodes[0].sum()).max()),
                initial_potential_sum=float(flow[key + "potential"][0].sum())))
    return rows
