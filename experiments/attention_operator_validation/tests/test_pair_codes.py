import pytest

torch = pytest.importorskip("torch")

from experiments.attention_operator_validation.pair_codes import (
    PAIR_RETAINED,
    PAIR_SELF,
    build_pair_code_field,
)
from experiments.grounded_route.tests.helpers import make_graph


def test_pair_codes_group_heads_without_averaging_them():
    graph = make_graph(layers=2, heads=3, response_count=4).canonicalize()
    field = build_pair_code_field(graph, imputation="zero")

    retained = field.kind == PAIR_RETAINED
    assert bool(retained.any())
    selected = torch.nonzero(retained, as_tuple=False)[0, 0]
    layer = field.layer[selected]
    target = field.target[selected]
    source = field.source[selected]
    matching = (
        (graph.edges.layer == layer)
        & (graph.edges.target == target)
        & (graph.edges.source == source)
    )
    expected = torch.zeros(graph.head_count)
    expected.index_add_(0, graph.edges.head[matching], graph.edges.weight[matching])

    assert torch.allclose(field.code[selected], expected)
    assert torch.equal(field.observed[selected], expected > 0)
    assert torch.allclose(
        field.direction[selected].sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_self_pair_code_is_exact_diagonal_vector():
    graph = make_graph(layers=2, heads=3, response_count=4)
    field = build_pair_code_field(graph)
    self_pair = field.kind == PAIR_SELF

    assert int(self_pair.sum()) == graph.layer_count * graph.response_count
    first = torch.nonzero(self_pair, as_tuple=False)[0, 0]
    response = int(field.target[first]) - graph.response_start
    layer = int(field.layer[first])
    assert torch.allclose(field.code[first], graph.diagonal[response, layer])
    assert bool(field.observed[first].all())


def test_censoring_imputation_changes_code_direction_not_observed_route_mass():
    graph = make_graph(layers=1, heads=3, response_count=4).canonicalize()
    zero = build_pair_code_field(graph, imputation="zero")
    floor = build_pair_code_field(graph, imputation="floor")

    assert torch.allclose(zero.magnitude, floor.magnitude)
    retained = zero.kind == PAIR_RETAINED
    partially_observed = retained & (~zero.observed.all(dim=1))
    if bool(partially_observed.any()):
        assert not torch.allclose(
            zero.direction[partially_observed],
            floor.direction[partially_observed],
        )
