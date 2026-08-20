import torch

from experiments.attention_phenomenology.head_resolved import (
    HeadResolvedFeatureExtractor,
)

from .helpers import SyntheticAttention, SyntheticSample


def test_extractor_preserves_different_head_routes():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=2,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 2, 3)),
    )
    sample = SyntheticSample(
        attention,
        edges=(
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0.8, 0.2, 0.6],
        ),
    )

    result = HeadResolvedFeatureExtractor(reuse_top_k=2).extract(sample)
    prompt_mass = result.feature("prompt_mass")
    response_mass = result.feature("response_mass")

    assert result.values.shape == (2, 1, 2, len(result.feature_names))
    torch.testing.assert_close(prompt_mass[0, 0], torch.tensor([0.8, 0.2]))
    torch.testing.assert_close(response_mass[1, 0], torch.tensor([0.6, 0.0]))


def test_response_reuse_coordinates_are_strictly_causal_and_head_resolved():
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
            [0, 0],
            [0, 0],
            [1, 2],
            [1, 1],
            [0.6, 0.4],
        ),
    )

    result = HeadResolvedFeatureExtractor(reuse_top_k=2).extract(sample)
    first_rank = result.feature("response_reuse_rank_1")[:, 0]
    second_rank = result.feature("response_reuse_rank_2")[:, 0]

    torch.testing.assert_close(first_rank[:, 0], torch.tensor([0.0, 0.3, 1.0 / 3.0]))
    torch.testing.assert_close(first_rank[:, 1], torch.zeros(3))
    torch.testing.assert_close(second_rank, torch.zeros_like(second_rank))


def test_appending_future_edges_cannot_change_prefix_features():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=1,
        num_response_tokens=3,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 1, 4)),
    )
    prefix = SyntheticSample(
        attention,
        edges=([0], [0], [1], [1], [0.6]),
    )
    extended = SyntheticSample(
        attention,
        edges=([0, 0], [0, 0], [1, 2], [1, 1], [0.6, 0.9]),
    )
    extractor = HeadResolvedFeatureExtractor(reuse_top_k=2)

    before = extractor.extract(prefix).values[:2]
    after = extractor.extract(extended).values[:2]

    torch.testing.assert_close(before, after)


def test_reuse_coordinates_can_be_removed_for_a_control_experiment():
    attention = SyntheticAttention(
        num_layers=1,
        num_heads=2,
        num_response_tokens=2,
        response_idx=1,
        attention_diagonal=torch.zeros((1, 2, 3)),
    )
    sample = SyntheticSample(
        attention,
        edges=([0], [1], [1], [1], [0.7]),
    )

    result = HeadResolvedFeatureExtractor(reuse_top_k=0).extract(sample)

    assert result.values.shape == (2, 1, 2, 16)
    assert result.feature_names[-1] == "response_active"
