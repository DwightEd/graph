from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.ragtruth_flow.native import (
    cached_values,
    forward_target,
    prepare_target,
)
from experiments.ragtruth_flow.screen import row_features
from experiments.ragtruth_flow.confirm import target_rows
from experiments.ragtruth_flow.report import paired_functional, competition


def test_screen_features_split_prompt_self_recent_remote():
    class Answer:
        prompt_length = 3
    keys = np.array([0, 2, 3, 4, 5, 6])
    weights = np.array([.1, .2, .05, .1, .15, .4])
    # response position 4 => absolute query 6
    values = row_features(Answer(), 4, keys, weights, recent=2)
    np.testing.assert_allclose(values, [.3, .4, .25, .05, 1.0])


def test_target_rows_separates_onset_and_mid_continuation():
    pair = SimpleNamespace(
        error=SimpleNamespace(start=4, end=9, length=5),
        control=SimpleNamespace(start=14, end=19, length=5),
    )
    assert target_rows(pair) == [
        ("error_onset", 4),
        ("control_onset", 14),
        ("error_continuation", 6),
        ("control_continuation", 16),
    ]


def test_cached_values_repeat_gqa_heads():
    values = torch.arange(40.0).reshape(1, 2, 5, 4)
    cache = SimpleNamespace(
        layers=[SimpleNamespace(values=values)]
    )
    repeated = cached_values(cache, 0, heads=4, kv_heads=2)
    assert repeated.shape == (1, 4, 5, 4)
    torch.testing.assert_close(repeated[:, 0], repeated[:, 1])
    torch.testing.assert_close(repeated[:, 2], repeated[:, 3])


def test_cached_query_matches_full_eager_on_tiny_llama():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    prefix = [2, 3, 4, 5, 6]
    target = 7
    groups = dict(
        source=np.array([0]),
        other_prompt=np.array([1, 2]),
        history=np.array([3]),
        query_self=np.array([4]),
        selected_heads=((0, 1),),
    )

    model.set_attn_implementation("eager")
    with torch.inference_mode():
        full = model(torch.tensor([prefix])).logits[0, -1]
        expected = float(full.double().log_softmax(-1)[target])

    prepared = prepare_target(model, prefix)
    actual, run = forward_target(model, prepared, target, groups)
    changed, _ = forward_target(
        model,
        prepared,
        target,
        groups,
        dict(layer=0, head=1, source_group="source"),
    )

    assert actual == pytest.approx(expected, abs=2e-5)
    assert np.isfinite(actual - changed)
    local = next(
        row for row in run.local
        if row["layer"] == 0
        and row["head"] == 1
        and row["source_group"] == "source"
    )
    assert np.isfinite(local["local_support"])

def test_pair_report_uses_same_response_coordinates():
    table = pd.DataFrame([
        dict(response_id="a", source_id="s", role="error_onset", layer=1, head=2,
             source_group="source", final_support=.7, local_support=.8,
             attention_mass=.4, downstream_reversal=False),
        dict(response_id="a", source_id="s", role="control_onset", layer=1, head=2,
             source_group="source", final_support=.2, local_support=.3,
             attention_mass=.5, downstream_reversal=True),
    ])
    row = paired_functional(table).iloc[0]
    assert row.final_support_difference == pytest.approx(.5)
    assert row.local_support_difference == pytest.approx(.5)
    assert row.attention_mass_difference == pytest.approx(-.1)


def test_competition_is_zero_for_one_direction_and_one_when_balanced():
    table = pd.DataFrame([
        dict(response_id="a", role="error_onset", source_group="history", final_support=2.0),
        dict(response_id="a", role="error_onset", source_group="history", final_support=-2.0),
        dict(response_id="b", role="error_onset", source_group="history", final_support=1.0),
        dict(response_id="b", role="error_onset", source_group="history", final_support=3.0),
    ])
    result = competition(table).set_index("response_id")
    assert result.loc["a", "competition"] == 1
    assert result.loc["b", "competition"] == 0
