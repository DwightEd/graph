import torch

from experiments.causal_walk_audit.anchors import uniform_prompt_anchors
from experiments.causal_walk_audit.lineage import propagate_anchor_lineage

from .helpers import routing_state


def test_lineage_conserves_mass_and_tracks_relay_depth():
    routing = routing_state()
    anchors = uniform_prompt_anchors(
        2,
        max_anchors=2,
        chunk_tokens=1,
        device=torch.device("cpu"),
    )
    trace = propagate_anchor_lineage(routing, anchors)
    total = trace.state.sum(dim=(-1, -2)) + trace.unresolved
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    assert float(trace.relay_anchor()[1, 1].sum()) > 0
    assert float(trace.multihop_anchor()[2, 2].sum()) > 0
    assert float(trace.response_base_multihop()[2, 2].sum()) > 0
