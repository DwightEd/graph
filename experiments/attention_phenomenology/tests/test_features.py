import torch

from experiments.attention_phenomenology.experiment import _detail_arrays
from experiments.attention_phenomenology.features import analyze_routing
from experiments.attention_phenomenology.hypotheses import FEATURE_INDEX, FEATURE_NAMES
from experiments.attention_phenomenology.routing import collect_routing_edges

from .helpers import SyntheticAttention, SyntheticSample


EXPECTED_FEATURES = (
    "prompt_mass_mean",
    "prompt_effective_sources_mean",
    "prompt_top1_share_mean",
    "prompt_source_velocity_mean",
    "prompt_response_head_disagreement",
    "prompt_mass_head_std",
    "prompt_provenance_head_std",
    "prompt_anchor_head_agreement",
    "prompt_provenance_lower_mean",
    "prompt_provenance_uncertainty",
    "unsupported_response_mass_mean",
    "response_takeover_mean",
    "response_effective_sources_mean",
    "response_top1_share_mean",
    "recent_response_share_mean",
    "response_mean_lag_mean",
    "response_source_velocity_mean",
    "response_anchor_head_agreement",
    "self_mass_mean",
    "unresolved_mass_mean",
    "known_mass_mean",
)


def _feature(analysis, name):
    return analysis.layer_features[:, :, FEATURE_INDEX[name]]


def test_analysis_exposes_small_interpretable_feature_contract():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=3,
        response_idx=2,
        attention_diagonal=torch.zeros((1, 2, 5)),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 2, 1],
            [0.6, 0.6, 0.6, 0.6],
        ),
    )

    analysis = analyze_routing(collect_routing_edges(sample))

    assert FEATURE_NAMES == EXPECTED_FEATURES
    assert analysis.layer_features.shape == (3, 1, len(EXPECTED_FEATURES))
    assert analysis.routing.role_names == (
        "prompt",
        "response_history",
        "self",
        "unresolved",
    )
    assert analysis.routing.role_probability.shape == (3, 1, 2, 4)
    torch.testing.assert_close(
        analysis.routing.role_probability.sum(dim=-1),
        torch.ones((3, 1, 2)),
    )


def test_prompt_and_response_routes_remain_distinct():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=2,
        response_idx=2,
        attention_diagonal=torch.zeros((1, 2, 4)),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 2, 1],
            [0.6, 0.6, 0.6, 0.6],
        ),
    )

    analysis = analyze_routing(collect_routing_edges(sample))

    assert _feature(analysis, "prompt_mass_mean")[0, 0] == 0.6
    assert _feature(analysis, "response_takeover_mean")[0, 0] == 0.0
    assert _feature(analysis, "prompt_mass_mean")[1, 0] == 0.3
    assert _feature(analysis, "response_takeover_mean")[1, 0] == 0.5
    assert _feature(analysis, "prompt_response_head_disagreement")[1, 0] > 0.9


def test_response_takeover_ignores_heads_with_no_observed_prompt_or_response_edge():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=2,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 2, 3)),
    )
    sample = SyntheticSample(
        attention,
        edges=([0], [0], [1], [1], [0.7]),
    )

    analysis = analyze_routing(collect_routing_edges(sample))

    assert _feature(analysis, "response_takeover_mean")[1, 0] == 1.0


def test_exact_response_sources_keep_the_head_dimension():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=3,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 2, 4)),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 2, 2],
            [1, 1, 1, 2],
            [0.7, 0.7, 0.7, 0.7],
        ),
    )

    analysis = analyze_routing(collect_routing_edges(sample))

    assert analysis.response_sources.top_source.shape == (3, 1, 2)
    assert analysis.response_sources.top_source[1, 0].tolist() == [0, 0]
    assert analysis.response_sources.top_source[2, 0].tolist() == [0, 1]
    assert _feature(analysis, "response_anchor_head_agreement")[1, 0] == 1.0
    assert _feature(analysis, "response_anchor_head_agreement")[2, 0] == 0.0


def test_exact_source_velocity_detects_anchor_change_per_head():
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

    velocity = analysis.response_sources.velocity[:, 0, 0]
    assert velocity[1] == 0.0
    assert velocity[2] == 1.0


def test_detail_artifact_contains_interpretable_source_fields():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=1,
        num_response_tokens=2,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 1, 3)),
    )
    sample = SyntheticSample(
        attention,
        edges=([0], [0], [1], [1], [0.8]),
    )
    analysis = analyze_routing(collect_routing_edges(sample))

    detail = _detail_arrays(analysis)

    assert "prompt_top_source" in detail
    assert "response_top_source" in detail
    assert "known_persistence_deaths" not in detail
    assert "source_mass" not in detail
