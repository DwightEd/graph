import numpy as np
import torch

from experiments.attention_phenomenology.config import PhenomenologyConfig
from experiments.attention_phenomenology.features import analyze_routing
from experiments.attention_phenomenology.nulls import rewire_exact_endpoints
from experiments.attention_phenomenology.routing import collect_routing_edges

from .helpers import SyntheticAttention, SyntheticSample


def test_endpoint_null_preserves_roles_and_changes_sources():
    config = PhenomenologyConfig(
        null_prompt_position_bins=2,
        null_response_lag_bins=2,
        random_seed=9,
    )
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=4,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 2, 5)),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [3, 3, 3, 3],
            [1, 2, 1, 2],
            [0.4, 0.3, 0.2, 0.1],
        ),
    )
    edges = collect_routing_edges(sample, config=config)
    null = rewire_exact_endpoints(edges, config=config, seed=3)
    real = analyze_routing(edges, config=config)
    rewired = analyze_routing(null.edges, config=config)

    np.testing.assert_allclose(
        real.routing.role_probability.cpu().numpy(),
        rewired.routing.role_probability.cpu().numpy(),
        atol=1e-6,
    )
    assert null.changed_fraction > 0


def test_prompt_rewire_never_crosses_its_position_bin():
    config = PhenomenologyConfig(null_prompt_position_bins=4)
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=1,
        num_response_tokens=1,
        response_idx=7,
        attention_diagonal=torch.zeros((1, 1, 8)),
    )
    sample = SyntheticSample(
        attention,
        edges=([0], [0], [0], [3], [0.8]),
    )
    edges = collect_routing_edges(sample, config=config)

    rewired = rewire_exact_endpoints(edges, config=config, seed=1).edges

    original_bin = edges.source * 4 // edges.response_idx
    rewired_bin = rewired.source * 4 // edges.response_idx
    torch.testing.assert_close(rewired_bin, original_bin)
