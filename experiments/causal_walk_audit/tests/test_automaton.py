import torch

from experiments.causal_walk_audit.automaton import (
    P_PLUS,
    R_NEAR,
    run_typed_automaton,
)
from experiments.causal_walk_audit.graph import (
    PROMPT,
    RESPONSE_NEAR,
    RoutingGraph,
)


def test_typed_automaton_preserves_mass_and_relay_type():
    prompt = torch.tensor([[[0.5], [0.5]], [[0.0], [0.0]]])
    response = torch.tensor([[[0.0], [0.0]], [[0.5], [0.5]]])
    self_mass = torch.full((2, 2, 1), 0.2)
    unresolved = 1.0 - prompt - response - self_mass
    graph = RoutingGraph(
        sample_id="sample",
        response_idx=1,
        num_response_tokens=2,
        num_tokens=3,
        num_layers=2,
        num_heads=1,
        attention_floor=0.01,
        recent_lag=4,
        source=torch.tensor([0, 0, 1, 1]),
        target=torch.tensor([1, 1, 2, 2]),
        layer=torch.tensor([0, 1, 0, 1]),
        head=torch.zeros(4, dtype=torch.long),
        relation=torch.tensor([PROMPT, PROMPT, RESPONSE_NEAR, RESPONSE_NEAR]),
        weight=torch.full((4,), 0.5),
        prompt_mass=prompt,
        response_mass=response,
        self_mass=self_mass,
        unresolved_mass=unresolved,
    ).validate()

    trace = run_typed_automaton(graph)
    torch.testing.assert_close(
        trace.route.sum(dim=-1),
        torch.ones((2, 2, 1)),
    )
    assert trace.route[1, 0, 0, R_NEAR] > 0
    assert trace.route[1, 1, 0, P_PLUS] > 0
