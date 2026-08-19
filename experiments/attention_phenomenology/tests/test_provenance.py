import torch

from experiments.attention_phenomenology.features import analyze_routing
from experiments.attention_phenomenology.routing import collect_routing_edges

from .helpers import SyntheticAttention, SyntheticSample


def test_layered_prompt_provenance_tracks_two_layer_relay():
    attention = SyntheticAttention(
        num_layers=2,
        num_heads=1,
        num_response_tokens=2,
        response_idx=1,
        attention_diagonal=torch.zeros((2, 1, 3)),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 1],
            [0, 0],
            [0, 1],
            [0, 1],
            [1.0, 1.0],
        ),
    )
    analysis = analyze_routing(collect_routing_edges(sample))
    lower = analysis.provenance.aggregate_lower.cpu().numpy()
    assert lower[0, 1] == 1.0
    assert lower[1, 1] == 0.0
    assert lower[1, 2] == 1.0


def test_unsupported_feedback_counts_rr_not_self_attention():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=1,
        num_response_tokens=1,
        response_idx=1,
        attention_diagonal=torch.tensor([[[0.8, 0.8]]]),
    )
    sample = SyntheticSample(
        attention,
        edges=([0], [0], [0], [0], [0.2]),
    )
    analysis = analyze_routing(collect_routing_edges(sample))
    assert analysis.provenance.unsupported_rr_lower[0, 0, 0] == 0.0
