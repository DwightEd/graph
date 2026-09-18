from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_path_conflict import fixture
from experiments.ragtruth_flow.native import forward_target
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


def install_base_forward(model):
    def base_forward(base, input_ids, attention_mask=None, use_cache=False,
                     output_attentions=True, return_dict=True):
        state = model.embedding(input_ids)
        attentions = []
        for layer in base.layers:
            state, attention = layer(state)
            attentions.append(attention)
        state = base.norm(state)
        return SimpleNamespace(last_hidden_state=state, attentions=tuple(attentions))
    model.model.forward = MethodType(base_forward, model.model)


def test_actual_token_support_cut_matches_direct_logprob_change():
    model, probe = fixture()
    install_base_forward(model)
    groups = dict(
        source=np.array([0]),
        other_prompt=np.array([1, 2]),
        history=np.array([3]),
        query_self=np.array([4]),
        selected_heads=((0, 1),),
    )
    target = 7
    base, run = forward_target(model, probe["prefix_ids"], target, groups)
    changed, _ = forward_target(
        model,
        probe["prefix_ids"],
        target,
        groups,
        dict(layer=0, head=1, source_group="source"),
    )
    local = next(
        row for row in run.local
        if row["layer"] == 0 and row["head"] == 1 and row["source_group"] == "source"
    )
    assert np.isfinite(base - changed)
    assert np.isfinite(local["local_support"])
    manual = model(torch.tensor([probe["prefix_ids"]])).logits[0, -1].double().log_softmax(-1)[target]
    assert base == pytest.approx(float(manual), abs=2e-6)


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
