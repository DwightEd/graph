from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from experiments.attention_phenomenology.config import PhenomenologyConfig
from experiments.attention_phenomenology.features import analyze_routing
from experiments.attention_phenomenology.geometry import (
    hellinger_distance_matrix,
    zero_dimensional_persistence,
)
from experiments.attention_phenomenology.routing import (
    collect_routing_edges,
    rewire_exact_endpoints,
)


@dataclass
class SyntheticAttention:
    num_layers: int
    num_heads: int
    num_response_tokens: int
    response_idx: int
    attention_diagonal: torch.Tensor
    attention_floor: float = 0.01

    @property
    def num_tokens(self):
        return self.response_idx + self.num_response_tokens

    @property
    def num_channels(self):
        return self.num_layers * self.num_heads

    @property
    def response_values(self):
        return torch.empty(0)


class SyntheticSample:
    def __init__(self, attention, edges):
        self._attention = attention
        self.edges = edges

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        layer, head, query, source, weight = self.edges
        yield SimpleNamespace(
            layer=torch.tensor(layer, dtype=torch.long),
            head=torch.tensor(head, dtype=torch.long),
            query=torch.tensor(query, dtype=torch.long),
            source=torch.tensor(source, dtype=torch.long),
            weight=torch.tensor(weight, dtype=torch.float32),
        )


def test_zero_dimensional_persistence_detects_head_coalitions():
    coherent = torch.tensor(
        [[[0.50, 0.50], [0.49, 0.51], [0.51, 0.49], [0.50, 0.50]]]
    )
    fractured = torch.tensor(
        [[[0.99, 0.01], [0.98, 0.02], [0.01, 0.99], [0.02, 0.98]]]
    )
    coherent_deaths = zero_dimensional_persistence(
        hellinger_distance_matrix(coherent)
    )
    fractured_deaths = zero_dimensional_persistence(
        hellinger_distance_matrix(fractured)
    )
    assert fractured_deaths.max() > coherent_deaths.max()


def test_layered_prompt_provenance_tracks_a_two_layer_relay():
    # Prompt token 0; response tokens are absolute positions 1 and 2.
    # Layer 0: response 0 reads prompt. Layer 1: response 1 reads response 0.
    diagonal = torch.zeros((2, 1, 3), dtype=torch.float32)
    attention = SyntheticAttention(
        num_layers=2,
        num_heads=1,
        num_response_tokens=2,
        response_idx=1,
        attention_diagonal=diagonal,
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
    lower = analysis.provenance.aggregate_lower.detach().cpu().numpy()
    assert lower[0, 1] == 1.0
    assert lower[1, 1] == 0.0
    assert lower[1, 2] == 1.0


def test_endpoint_rewire_preserves_roles_but_changes_exact_sources():
    config = PhenomenologyConfig(prompt_bins=2, rr_lag_bins=2, random_seed=9)
    diagonal = torch.zeros((1, 2, 5), dtype=torch.float32)
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=4,
        response_idx=1,
        attention_diagonal=diagonal,
    )
    # Query 3 has two RR candidates (response sources 0 and 1) in the same
    # coarse lag bin; rewiring can exchange their exact identities.
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
    rewired = rewire_exact_endpoints(edges, config=config, seed=3)
    real = analyze_routing(edges, config=config)
    null = analyze_routing(rewired, config=config)

    np.testing.assert_allclose(
        real.routing.role_probability.detach().cpu().numpy(),
        null.routing.role_probability.detach().cpu().numpy(),
        atol=1e-6,
    )
    assert not np.array_equal(
        edges.source.detach().cpu().numpy(), rewired.source.detach().cpu().numpy()
    )


def test_sample_analysis_keeps_token_layer_feature_geometry():
    diagonal = torch.full((2, 2, 4), 0.1, dtype=torch.float32)
    attention = SyntheticAttention(
        num_layers=2,
        num_heads=2,
        num_response_tokens=3,
        response_idx=1,
        attention_diagonal=diagonal,
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
    assert analysis.layer_features.shape[:2] == (3, 2)
    assert analysis.routing.role_probability.shape[:3] == (3, 2, 2)
    assert torch.isfinite(analysis.layer_features).all()


def test_censoring_is_an_explicit_probability_role_and_bounds_are_ordered():
    diagonal = torch.full((1, 2, 3), 0.2, dtype=torch.float32)
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=2,
        response_idx=1,
        attention_diagonal=diagonal,
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0.3, 0.4],
        ),
    )
    analysis = analyze_routing(collect_routing_edges(sample))
    probability_sum = analysis.routing.role_probability.sum(dim=3)
    torch.testing.assert_close(probability_sum, torch.ones_like(probability_sum))
    assert torch.all(
        analysis.provenance.head_lower <= analysis.provenance.head_upper
    )
    assert torch.all(analysis.routing.unresolved_mass >= 0)


def test_unlabeled_reference_standardization_preserves_layer_feature_shape():
    from experiments.attention_phenomenology.config import FEATURE_NAMES
    from experiments.attention_phenomenology.reference import (
        PhenomenologyReference,
        family_atypicality,
        standardize_features,
    )

    layers = 2
    features = len(FEATURE_NAMES)
    reference = PhenomenologyReference(
        task=np.asarray(["__all__"]),
        bucket=np.asarray([0], dtype=np.int16),
        center=np.zeros((1, layers, features), dtype=np.float32),
        scale=np.ones((1, layers, features), dtype=np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        family_names=np.asarray(["fracture", "integration", "lockin", "all"]),
        config_json="{}",
    )
    values = np.ones((3, layers, features), dtype=np.float32)
    standardized = standardize_features(
        values,
        task="QA",
        buckets=np.asarray([0, 1, 2], dtype=np.int16),
        reference=reference,
    )
    assert standardized.shape == values.shape
    assert family_atypicality(standardized).shape == (3, 4)
