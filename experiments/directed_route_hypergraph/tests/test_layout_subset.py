import torch

from experiments.directed_route_hypergraph.config import LearningConfig, ModelConfig
from experiments.directed_route_hypergraph.layout import (
    endpoint_layout_plan,
    ordered_endpoint_layout,
)
from experiments.directed_route_hypergraph.learning import (
    endpoint_layout_loss,
    sample_layout_rows,
    self_supervised_loss,
)
from experiments.directed_route_hypergraph.model import DirectedRouteHypergraphEncoder
from experiments.grounded_route.tests.helpers import make_graph


def make_model(graph):
    return DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(dropout=0.0),
    )


def test_selected_endpoint_layout_exactly_matches_full_rows_in_both_orders():
    graph = make_graph(layers=4, heads=3, response_count=9)
    requested = torch.tensor([7, 1, 7, 4])

    for order in (None, tuple(reversed(range(graph.layer_count)))):
        full = ordered_endpoint_layout(graph, layer_order=order).distribution
        selected = ordered_endpoint_layout(
            graph,
            layer_order=order,
            response_index=requested,
        ).distribution

        assert selected.shape == (len(requested), graph.token_count + 1)
        assert torch.allclose(
            selected,
            full[requested],
            atol=1e-6,
            rtol=1e-6,
        )


def test_selected_layout_plan_reduces_dense_teacher_work():
    graph = make_graph(layers=4, heads=3, response_count=12)
    full = endpoint_layout_plan(graph, torch.arange(graph.response_count))
    selected = endpoint_layout_plan(graph, torch.tensor([2, 8]))

    assert selected.work_element_count < full.work_element_count
    assert selected.peak_state_elements < full.peak_state_elements
    assert torch.equal(selected.required_rows[-1], torch.tensor([2, 8]))


def test_budgeted_layout_sampling_shrinks_instead_of_aborting_training():
    torch.manual_seed(71)
    graph = make_graph()
    full = endpoint_layout_plan(graph, torch.arange(graph.response_count))
    first = endpoint_layout_plan(graph, torch.tensor([0]))
    budget = max(first.work_element_count, full.work_element_count // 3)
    assert budget < full.work_element_count

    selected = sample_layout_rows(
        graph,
        graph.response_count,
        torch.Generator().manual_seed(73),
        max_elements=10_000,
        max_work_elements=budget,
    )
    selected_plan = endpoint_layout_plan(graph, selected)

    assert 0 < len(selected) < graph.response_count
    assert selected_plan.work_element_count <= budget

    model = make_model(graph).eval()
    output = model(graph)
    objective = endpoint_layout_loss(
        model,
        output,
        graph,
        response_index=selected,
        max_elements=10_000,
        max_work_elements=budget,
    )

    assert objective.row_count == len(selected)
    assert torch.isfinite(objective.loss)


def test_pipeline_skips_only_layout_when_even_one_exact_row_exceeds_budget():
    torch.manual_seed(79)
    graph = make_graph()
    model = make_model(graph).train()
    output = self_supervised_loss(
        model,
        graph,
        LearningConfig(
            layout_rows_per_graph=graph.response_count,
            layout_max_elements=1,
            layout_max_work_elements=1,
        ),
        torch.Generator().manual_seed(83),
    )

    assert torch.isfinite(output.loss)
    assert output.layout.item() == 0.0
    assert output.layout_row_count == 0
    assert output.row_count > 0
    assert output.flow.item() > 0.0
