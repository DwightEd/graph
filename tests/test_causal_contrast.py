import math

import pytest
import torch

from route_graph.causal_contrast import ContinuationContrast, message_gate_delta


def test_contrast_uses_complete_disjoint_events_and_last_shared_query():
    contrast = ContinuationContrast.from_sequences(
        [[1, 2, 3, 4, 0], [1, 2, 3, 5, 6, 0], [1, 2, 3, 7, 0]], 2
    )
    assert contrast.prefix == (1, 2, 3)
    assert contrast.shared_queries == (1, 2)
    score = contrast.score(
        [
            [math.log(0.2), math.log(0.5)],
            [math.log(0.1), math.log(0.5), math.log(0.5)],
            [math.log(0.15), math.log(0.5)],
        ]
    )
    assert score == pytest.approx(0.0)  # .1 / (.025 + .075), no length mean.
    with pytest.raises(ValueError, match="exactly one"):
        contrast.score([[-1], [-1], [-1]])


@pytest.mark.parametrize(
    "sequences",
    [
        [[1, 2, 3], [1, 2, 3, 4]],
        [[1, 2, 3], [1, 2, 4], [1, 2, 4]],
        [[1, 2, 3], [1, 5, 3]],
    ],
)
def test_reject_overlapping_events_and_changed_prompt(sequences):
    with pytest.raises(ValueError):
        ContinuationContrast.from_sequences(sequences, 2)


def test_route_conserves_mass_content_does_not_redistribute_and_gqa_is_preserved():
    a = torch.tensor([[[0.2, 0.3]], [[0.4, 0.1]], [[0.1, 0.4]], [[0.25, 0.25]]])
    v = torch.tensor([[[1.0], [10.0]], [[3.0], [30.0]]])
    route, stats = message_gate_delta(a, v, [True, False], 0, "route")
    content, _ = message_gate_delta(a, v, [True, False], 0, "content")
    assert torch.allclose(route.squeeze(), torch.tensor([0.4, 0.8, 2.0, 5.0]))
    assert torch.allclose(content.squeeze(), torch.tensor([-0.2, -0.4, -1.0, -2.5]))
    assert stats["mass_error"] < 1e-6
    for kind in ("route", "content"):
        sham, _ = message_gate_delta(a, v, [True, False], 1, kind)
        assert torch.equal(sham, torch.zeros_like(sham))


def test_route_cannot_delete_last_visible_key_but_zero_mass_is_valid():
    a = torch.tensor([[[0.5, 0.0], [0.0, 0.0]]])
    v = torch.tensor([[[1.0]], [[3.0]]])
    with pytest.raises(ValueError, match="no key"):
        message_gate_delta(a, v, [True, False], 0, "route")
    delta, _ = message_gate_delta(a[:, 1:], v, [True, True], 0, "route")
    assert torch.equal(delta, torch.zeros_like(delta))


def test_route_has_no_effect_when_value_messages_are_identical():
    a = torch.tensor([[[0.2, 0.3]]])
    v = torch.tensor([[[2.0, 4.0]], [[2.0, 4.0]]])
    delta, _ = message_gate_delta(a, v, [True, False], 0, "route")
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_extreme_finite_sequence_scores_keep_alternative_count_or_reject_overflow():
    contrast = ContinuationContrast.from_sequences([[1, 2], [1, 3], [1, 4]], 1)
    assert contrast.score([[-1e308], [-1e308], [-1e308]]) == pytest.approx(-math.log(2))
    long = ContinuationContrast.from_sequences([[1, 2, 5], [1, 3, 5]], 1)
    with pytest.raises(ValueError, match="float64 range"):
        long.score([[-1e308, -1e308], [-1e308, -1e308]])


def test_float64_messages_preserve_range_and_invalid_probability_mass_is_rejected():
    a = torch.tensor([[[0.25, 0.25]]], dtype=torch.float64)
    v = torch.tensor([[[1e100]], [[2e100]]], dtype=torch.float64)
    delta, _ = message_gate_delta(a, v, [True, False], 0, "route")
    assert delta.dtype == torch.float64
    assert delta.item() == pytest.approx(0.25e100)
    with pytest.raises(ValueError, match="total attention mass"):
        message_gate_delta(torch.tensor([[[0.5, 0.505]]]), v, [True, False], 0, "route")


def test_unrepresentable_message_delta_is_rejected():
    a = torch.tensor([[[0.75, 0.25]]], dtype=torch.float64)
    v = torch.tensor([[[-1.7e308]], [[1.7e308]]], dtype=torch.float64)
    with pytest.raises(ValueError, match="working dtype range"):
        message_gate_delta(a, v, [True, False], 0, "route")


def test_paired_transfer_moves_only_the_fixed_outgoing_mass_to_destination():
    a = torch.tensor([[[0.2, 0.3, 0.1]]])
    v = torch.eye(3).reshape(3, 1, 3)
    delta, stats = message_gate_delta(
        a, v, [True, False, False], 0.5, "route", [False, True, False]
    )
    assert torch.allclose(delta.squeeze(), torch.tensor([-0.1, 0.1, 0.0]))
    assert stats["mass_error"] < 1e-6
    with pytest.raises(ValueError, match="disjoint"):
        message_gate_delta(a, v, [True, False, False], 0, "route", [True, False, False])
    with pytest.raises(ValueError, match="receiving mass"):
        message_gate_delta(
            a, v, [True, False, False], 0, "route", [False, False, False]
        )
