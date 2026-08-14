import math

import torch

from cache import AttentionSample
from experiments.mechanism_validation.mechanisms import (
    MECHANISM_FAMILY_SLICES,
    MECHANISM_NAMES,
    compact_token_features,
    extract_token_mechanisms,
)


def sample() -> AttentionSample:
    # R0: P0=.2, P1=.3, diagonal=.1, OTHER=.4.
    # R1: P0=.1, R0=.4, diagonal=.2, OTHER=.3.
    diagonal = torch.zeros((1, 1, 4), dtype=torch.float16)
    diagonal[0, 0, 2:] = torch.tensor([.1, .2], dtype=torch.float16)
    return AttentionSample(
        "response", "source", 2, torch.arange(4, dtype=torch.int32), diagonal,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        torch.tensor([.2, .3, .1, .4], dtype=torch.float16), .01,
    )


def test_sparse_mechanisms_keep_layer_head_and_censor_missing_mass():
    result = extract_token_mechanisms(sample())
    index = {name: MECHANISM_NAMES.index(name) for name in MECHANISM_NAMES}

    assert result.values.shape == (2, 1, 1, len(MECHANISM_NAMES))
    assert MECHANISM_FAMILY_SLICES["routing"] == slice(0, 4)
    assert_close(float(result.values[0, 0, 0, index["retained_prompt_mass"]]), .5)
    assert_close(float(result.values[1, 0, 0, index["retained_history_mass"]]), .4)
    assert_close(float(result.values[0, 0, 0, index["retained_mass"]]), .6)
    assert_close(float(result.values[1, 0, 0, index["unresolved_mass"]]), .3)
    assert_close(float(result.values[0, 0, 0, index["retained_prompt_share"]]), .5 / .6)
    assert_close(float(result.values[0, 0, 0, index["prompt_share_lower_bound"]]), .5)
    assert_close(float(result.values[0, 0, 0, index["prompt_share_upper_bound"]]), .9)
    assert_close(float(result.values[0, 0, 0, index["retained_length_normalized_lookback"]]), 5 / 7)
    expected_entropy = -sum(value * math.log(value) for value in (.2, .3, .1, .4))
    assert_close(float(result.values[0, 0, 0, index["coarsened_entropy_lower_bound"]]), expected_entropy)


def test_zero_denominator_lookback_uses_attention_floor_and_is_valid():
    empty = AttentionSample(
        "empty", "source", 1, torch.arange(2, dtype=torch.int32),
        torch.zeros((1, 1, 2), dtype=torch.float16),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.float16), .01,
    )

    result = extract_token_mechanisms(empty)
    lookback = MECHANISM_NAMES.index("retained_length_normalized_lookback")

    assert_close(float(result.values[0, 0, 0, lookback]), empty.attention_floor)
    assert bool(result.valid[0, 0, 0, lookback])


def test_compact_features_include_masked_temporal_and_layer_drift_features():
    raw = extract_token_mechanisms(sample())
    compact = compact_token_features(raw, ema_decay=.5)

    assert compact.values.shape[0] == 2
    assert compact.values.shape == compact.valid.shape
    delta = compact.names.index("retained_prompt_mass:token_delta")
    innovation = compact.names.index("retained_prompt_mass:ema_innovation")
    drift = compact.names.index("retained_prompt_mass:early_late_layer_drift")
    assert not bool(compact.valid[0, delta])
    assert not bool(compact.valid[0, innovation])
    assert bool(compact.valid[1, delta])
    assert not bool(compact.valid[:, drift].any())
    for family, selection in compact.family_slices.items():
        expected = set(MECHANISM_NAMES[MECHANISM_FAMILY_SLICES[family]])
        assert all(name.split(":", 1)[0] in expected for name in compact.names[selection])


def test_ema_carries_valid_history_across_invalid_token():
    values = torch.tensor([[[[1.]]], [[[0.]]], [[[3.]]]])
    valid = torch.tensor([[[[True]]], [[[False]]], [[[True]]]])
    compact = compact_token_features(type("Raw", (), {
        "values": values, "valid": valid, "names": ("x",),
        "family_slices": {"routing": slice(0, 1)},
    })(), ema_decay=.5)

    innovation = compact.names.index("x:ema_innovation")
    assert not bool(compact.valid[1, innovation])
    assert bool(compact.valid[2, innovation])
    assert_close(float(compact.values[2, innovation]), 2.0)


def test_overfull_retained_mass_invalidates_only_cache_bounds():
    diagonal = torch.zeros((1, 1, 3), dtype=torch.float16)
    overfull = AttentionSample("id", "source", 1, torch.arange(3, dtype=torch.int32), diagonal,
        torch.tensor([0, 2, 2], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([.7, .7], dtype=torch.float16), .01)
    result = extract_token_mechanisms(overfull)
    names = {name: index for index, name in enumerate(MECHANISM_NAMES)}
    assert bool(result.valid[0, 0, 0, names["retained_length_normalized_lookback"]])
    assert bool(result.valid[0, 0, 0, names["retained_prompt_share"]])
    assert not bool(result.valid[0, 0, 0, names["prompt_share_lower_bound"]])
    assert not bool(result.valid[0, 0, 0, names["coarsened_hhi_upper_bound"]])


def assert_close(actual, expected):
    assert abs(actual - expected) < 2e-3
