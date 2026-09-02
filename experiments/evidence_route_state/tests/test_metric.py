from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from experiments.evidence_route_state.graph import GraphSequence
from experiments.evidence_route_state.metric import (
    BLOCK_NAMES,
    RouteMetric,
    block_distances,
)


def graph(level: float = 0.0) -> GraphSequence:
    tokens, layers, heads, channels, hidden = 3, 2, 2, 4, 3
    prediction = torch.arange(20, 20 + tokens)
    return GraphSequence(
        query_position=prediction - 1,
        prediction_position=prediction,
        node_embedding=torch.full((tokens, channels, hidden), level),
        residual_gram=torch.full((tokens, layers + 1, channels, channels), level),
        head_write_gram=torch.full((tokens, layers, heads, channels, channels), level),
        route_topology=torch.full((tokens, layers, heads, channels, 7), level),
        mlp_relation=torch.full((tokens, layers, channels + 1), level),
        margin_contribution=torch.full((tokens, channels), level),
        valid=torch.ones(tokens, dtype=torch.bool),
    )


def test_every_complete_tensor_block_has_equal_access_to_the_metric():
    baseline = graph()
    for expected, name in enumerate(BLOCK_NAMES):
        tensor = getattr(baseline, name).clone()
        tensor[1].reshape(-1)[0] = 1.0
        changed = replace(baseline, **{name: tensor})
        distance = block_distances((baseline, 1), (changed, 1))

        assert distance[expected] > 0
        assert np.count_nonzero(distance) == 1


def test_metric_preserves_head_layer_origin_order_and_signed_direction():
    baseline = graph()
    topology = baseline.route_topology.clone()
    topology[1, 0, 0, 0, 0] = 1.0
    topology[1, 0, 1, 0, 0] = 2.0
    permuted = topology.clone()
    permuted[1, 0, [0, 1]] = permuted[1, 0, [1, 0]]

    gram = baseline.head_write_gram.clone()
    gram[1, 1, 0, 0, 1] = 1.0
    opposed = gram.clone()
    opposed[1, 1, 0, 0, 1] = -1.0

    metric = RouteMetric(np.ones(len(BLOCK_NAMES)))
    assert (
        metric.distance(
            (replace(baseline, route_topology=topology), 1),
            (replace(baseline, route_topology=permuted), 1),
        )
        > 0
    )
    assert (
        metric.distance(
            (replace(baseline, head_write_gram=gram), 1),
            (replace(baseline, head_write_gram=opposed), 1),
        )
        > 0
    )


def test_block_scales_are_label_free_median_pair_distances():
    baseline = graph(0.0)
    one = graph(1.0)
    three = graph(3.0)
    metric = RouteMetric.fit([((baseline, 1), (one, 1)), ((baseline, 1), (three, 1))])

    np.testing.assert_allclose(metric.scale, 2.0)
    np.testing.assert_allclose(metric.distance((baseline, 1), (three, 1)), 1.5)


def test_vectorized_prototype_distance_matches_individual_distance():
    query = graph(0.5)
    references = (graph(1.0), graph(3.0))
    metric = RouteMetric(np.arange(1, len(BLOCK_NAMES) + 1, dtype=np.float64))
    batch = {
        name: np.stack([getattr(item, name)[1].numpy() for item in references])
        for name in BLOCK_NAMES
    }

    actual = metric.distances_to_batch(query, 1, batch)
    expected = [metric.distance((query, 1), (item, 1)) for item in references]
    np.testing.assert_allclose(actual, expected)
