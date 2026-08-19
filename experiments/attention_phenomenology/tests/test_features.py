import torch

from experiments.attention_phenomenology.features import analyze_routing
from experiments.attention_phenomenology.hypotheses import FEATURE_NAMES
from experiments.attention_phenomenology.routing import collect_routing_edges

from .helpers import SyntheticAttention, SyntheticSample


def test_analysis_preserves_token_layer_feature_and_known_role_geometry():
    attention = SyntheticAttention(
        num_layers=2,
        num_heads=2,
        num_response_tokens=3,
        response_idx=1,
        attention_diagonal=torch.full((2, 2, 4), 0.1),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 2],
            [0, 1, 0, 2],
            [0.6, 0.5, 0.4, 0.3],
        ),
    )
    analysis = analyze_routing(collect_routing_edges(sample))
    assert analysis.layer_features.shape == (3, 2, len(FEATURE_NAMES))
    assert analysis.routing.role_probability.shape[:3] == (3, 2, 2)
    assert analysis.routing.known_role_probability.shape[-1] + 1 == (
        analysis.routing.role_probability.shape[-1]
    )
    torch.testing.assert_close(
        analysis.routing.role_probability.sum(dim=3),
        torch.ones((3, 2, 2)),
    )


def test_exact_source_velocity_changes_when_anchor_changes():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=1,
        num_response_tokens=3,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 1, 4)),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0],
            [0, 0],
            [1, 2],
            [1, 2],
            [1.0, 1.0],
        ),
    )
    analysis = analyze_routing(collect_routing_edges(sample))
    velocity = analysis.source_dynamics.distribution_velocity[:, 0]
    assert velocity[2] > 0
