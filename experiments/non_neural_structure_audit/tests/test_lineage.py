import torch

from experiments.non_neural_structure_audit.lineage import (
    LINEAGE_INDEX,
    propagate_lineage,
)

from .helpers import routing_state


def test_response_relay_preserves_prompt_lineage_instead_of_response_base():
    routing = routing_state(
        layers=[0, 1],
        heads=[0, 0],
        queries=[0, 1],
        sources=[0, 1],
        weights=[1.0, 1.0],
        diagonal=torch.zeros((2, 2, 1)),
    )

    trace = propagate_lineage(routing)
    final = trace.state[1, -1]

    assert final[LINEAGE_INDEX["prompt_relay"]] == 1.0
    assert final[LINEAGE_INDEX["response_relay_one_hop"]] == 0.0


def test_response_relay_of_local_token_tracks_response_base_and_conserves_mass():
    diagonal = torch.zeros((2, 2, 1))
    diagonal[0, 0, 0] = 1.0
    routing = routing_state(
        layers=[1],
        heads=[0],
        queries=[1],
        sources=[1],
        weights=[1.0],
        diagonal=diagonal,
    )

    trace = propagate_lineage(routing)

    torch.testing.assert_close(trace.state.sum(dim=-1), torch.ones((2, 2)))
    assert trace.state[1, -1, LINEAGE_INDEX["response_relay_one_hop"]] == 1.0


def test_layer_shuffle_changes_ordered_lineage_without_changing_input_routes():
    routing = routing_state(
        layers=[0, 1],
        heads=[0, 0],
        queries=[0, 1],
        sources=[0, 1],
        weights=[1.0, 1.0],
        diagonal=torch.zeros((2, 2, 1)),
    )

    ordered = propagate_lineage(routing, layer_order=(0, 1))
    shuffled = propagate_lineage(routing, layer_order=(1, 0))

    assert ordered.state[1, -1, LINEAGE_INDEX["prompt_relay"]] == 1.0
    assert shuffled.state[1, -1, LINEAGE_INDEX["prompt_relay"]] == 0.0
