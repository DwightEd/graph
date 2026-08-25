from collections import Counter

import pytest
import torch

from experiments.grounded_route.config import (
    GroundedRouteConfig,
    InterventionConfig,
    TrainConfig,
)
from experiments.grounded_route.controls import (
    lag_bucket,
    rewire_endpoints_keep_roles,
    shuffle_weights_keep_endpoints,
    source_role,
)
from experiments.grounded_route.pipeline import (
    change_fraction,
    controlled_graph,
    require_effective_control,
)
from experiments.grounded_route.tests.helpers import (
    make_graph,
    make_rewirable_graph,
    make_weight_shuffle_graph,
)


def row_mass(graph):
    result = torch.zeros_like(graph.diagonal)
    result.index_put_(
        (
            graph.edges.target - graph.response_start,
            graph.edges.layer,
            graph.edges.head,
        ),
        graph.edges.weight,
        accumulate=True,
    )
    return result


def endpoint_degree(values):
    return Counter(map(int, values.detach().cpu().tolist()))


def grouped_weight_multisets(graph):
    groups = {}
    role = source_role(graph)
    for index in range(graph.edge_count):
        key = (
            int(graph.edges.target[index]),
            int(graph.edges.layer[index]),
            int(graph.edges.head[index]),
            int(role[index]),
        )
        groups.setdefault(key, []).append(float(graph.edges.weight[index]))
    return {key: sorted(value) for key, value in groups.items()}


def test_weight_shuffle_keeps_endpoints_support_and_all_row_nuisances():
    graph = make_weight_shuffle_graph()
    for seed in range(32):
        shuffled = shuffle_weights_keep_endpoints(
            graph,
            torch.Generator().manual_seed(seed),
        )
        if not torch.equal(shuffled.edges.weight, graph.edges.weight):
            break

    assert torch.equal(shuffled.edges.source, graph.edges.source)
    assert torch.equal(shuffled.edges.target, graph.edges.target)
    assert torch.equal(shuffled.edges.layer, graph.edges.layer)
    assert torch.equal(shuffled.edges.head, graph.edges.head)
    assert not torch.equal(shuffled.edges.weight, graph.edges.weight)
    assert grouped_weight_multisets(shuffled) == grouped_weight_multisets(graph)
    assert torch.allclose(row_mass(shuffled), row_mass(graph))
    assert torch.equal(shuffled.diagonal, graph.diagonal)
    assert torch.equal(shuffled.unresolved, graph.unresolved)


def test_endpoint_double_swap_preserves_roles_degrees_weights_and_causality():
    graph = make_rewirable_graph()
    original_source = graph.edges.source.clone()
    rewired = rewire_endpoints_keep_roles(
        graph,
        torch.Generator().manual_seed(29),
        passes=1,
    )

    assert not torch.equal(rewired.edges.source, original_source)
    assert torch.equal(graph.edges.source, original_source)
    assert torch.equal(rewired.edges.target, graph.edges.target)
    assert torch.equal(rewired.edges.layer, graph.edges.layer)
    assert torch.equal(rewired.edges.head, graph.edges.head)
    assert torch.equal(rewired.edges.weight, graph.edges.weight)
    assert torch.equal(source_role(rewired), source_role(graph))
    assert torch.equal(
        lag_bucket(rewired.edges.source, rewired.edges.target),
        lag_bucket(graph.edges.source, graph.edges.target),
    )
    assert endpoint_degree(rewired.edges.source) == endpoint_degree(graph.edges.source)
    assert endpoint_degree(rewired.edges.target) == endpoint_degree(graph.edges.target)
    assert bool((rewired.edges.source < rewired.edges.target).all())
    endpoint = set(
        zip(
            rewired.edges.source.tolist(),
            rewired.edges.target.tolist(),
            rewired.edges.layer.tolist(),
            rewired.edges.head.tolist(),
            strict=True,
        )
    )
    assert len(endpoint) == rewired.edge_count
    assert torch.allclose(row_mass(rewired), row_mass(graph))
    assert torch.equal(rewired.diagonal, graph.diagonal)
    assert torch.equal(rewired.unresolved, graph.unresolved)


@pytest.mark.parametrize(
    ("variant", "graph"),
    (
        ("weight_shuffle", make_weight_shuffle_graph()),
        ("endpoint_rewire", make_rewirable_graph()),
    ),
)
def test_control_reports_a_nonzero_changed_fraction(variant, graph):
    for seed in range(32):
        config = GroundedRouteConfig(
            train=TrainConfig(seed=seed),
            intervention=InterventionConfig(
                variant=variant,
                minimum_changed_fraction=0.01,
            ),
        )
        _, changed, total = controlled_graph(graph, config)
        if changed:
            break

    assert total == graph.edge_count
    assert 0 < changed <= total
    assert change_fraction(changed, total) == changed / total
    require_effective_control(config.intervention, changed, total)


def test_control_is_rejected_when_its_realized_change_is_too_small():
    graph = make_graph(layers=1, heads=1)
    config = GroundedRouteConfig(
        intervention=InterventionConfig(
            variant="weight_shuffle",
            minimum_changed_fraction=0.01,
        )
    )
    _, changed, total = controlled_graph(graph, config)

    assert changed == 0
    with pytest.raises(RuntimeError, match="changed fewer than 1.0%"):
        require_effective_control(config.intervention, changed, total)
