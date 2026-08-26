from dataclasses import replace

import pytest
import torch

from experiments.grounded_route.artifacts import EncodedTokenGraph
from experiments.grounded_route.controls import (
    rewire_endpoints_keep_roles,
    shuffle_weights_keep_endpoints,
)
from experiments.grounded_route.graph_effectiveness.control_audit import (
    verify_control_graph,
)
from experiments.grounded_route.tests.helpers import (
    make_rewirable_graph,
    make_weight_shuffle_graph,
)


def _encoded(graph):
    lineage = torch.zeros(graph.response_count, graph.layer_count, graph.head_count, 3)
    lineage[..., 0] = 1.0
    return EncodedTokenGraph(
        sample_id=graph.sample_id,
        source_id=graph.source_id,
        task_type=graph.task_type,
        response_start=graph.response_start,
        layer_count=graph.layer_count,
        head_count=graph.head_count,
        attention_floor=graph.attention_floor,
        token_ids=graph.token_ids,
        node_embedding=torch.randn(graph.token_count, 8),
        edge_index=torch.stack((graph.edges.source, graph.edges.target)),
        edge_layer=graph.edges.layer,
        edge_head=graph.edges.head,
        edge_weight=graph.edges.weight,
        diagonal=graph.diagonal,
        unresolved=graph.unresolved,
        lineage=lineage,
    )


def test_saved_endpoint_control_preserves_registered_invariants():
    real = make_rewirable_graph()
    control = rewire_endpoints_keep_roles(
        real,
        torch.Generator().manual_seed(29),
        passes=1,
    )
    assert verify_control_graph(
        _encoded(real),
        _encoded(control),
        "endpoint_rewire",
    ) > 0


def test_saved_weight_control_preserves_registered_invariants():
    real = make_weight_shuffle_graph()
    for seed in range(32):
        control = shuffle_weights_keep_endpoints(
            real,
            torch.Generator().manual_seed(seed),
        )
        if not torch.equal(real.edges.weight, control.edges.weight):
            break
    assert verify_control_graph(
        _encoded(real),
        _encoded(control),
        "weight_shuffle",
    ) > 0


def test_saved_weight_control_rejects_changed_group_multiset():
    graph = _encoded(make_weight_shuffle_graph())
    changed_weight = graph.edge_weight.clone()
    changed_weight[0] += 0.01
    changed = replace(graph, edge_weight=changed_weight)

    with pytest.raises(ValueError, match="weight multiset"):
        verify_control_graph(graph, changed, "weight_shuffle")


def test_no_message_control_uses_the_exact_same_saved_graph():
    real = _encoded(make_weight_shuffle_graph())
    control = replace(real, node_embedding=real.node_embedding + 1.0)

    assert verify_control_graph(real, control, "no_message") == 0
