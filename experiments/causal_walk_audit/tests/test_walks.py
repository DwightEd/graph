import torch

from experiments.causal_walk_audit.anchors import uniform_prompt_anchors
from experiments.causal_walk_audit.lineage import propagate_anchor_lineage
from experiments.causal_walk_audit.walks import (
    build_layer_event_graph,
    build_nested_features,
    causal_walk_contexts,
)

from .helpers import routing_state


def test_debruijn_predecessors_and_nested_shapes():
    routing = routing_state()
    event_graph = build_layer_event_graph(routing)
    assert event_graph.num_events > 0
    assert event_graph.predecessor.numel() > 0

    order2, order3 = causal_walk_contexts(event_graph)
    assert order2.shape == (3, 3, 4)
    assert order3.shape == (3, 3, 4)

    anchors = uniform_prompt_anchors(
        2,
        max_anchors=2,
        chunk_tokens=1,
        device=torch.device("cpu"),
    )
    lineage = propagate_anchor_lineage(routing, anchors)
    nested = build_nested_features(
        routing,
        lineage,
        event_graph,
        max_anchors=2,
    )
    assert nested.order1.shape[0] == 3 * 2
    assert nested.order1.shape[1] < nested.order2.shape[1] < nested.order3.shape[1]
    assert nested.target.shape[0] == nested.order1.shape[0]
