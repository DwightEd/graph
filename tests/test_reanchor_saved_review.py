"""Saved evidence must distinguish a lookback head from its selected source."""

import numpy as np

from experiments.reanchor_audit.review_results import candidate_rows, previous_mass_bounds


def test_positive_head_gain_does_not_make_a_decreasing_bos_edge_a_lookback():
    graph = dict(source=np.array([[[[0, 0], [0, 1], [0, 1], [0, 2]]]]),
                 attention=np.array([[[[1., 0.], [.8, .2], [.7, .3], [.7, .3]]]]))
    context = dict(token_text=["<|begin_of_text|>", " evidence", " reread", " answer"],
                   prompt_length=2, roles=dict(scope=[1]))
    plan = [dict(event=0, entry=dict(layer=0, head=0, receiver=2, sources=[0]),
                 selected=dict(selection="lookback_target_flow", lookback_gain=.1))]
    rows, _ = candidate_rows(graph, context, plan, 10, dict(case_id="test"))
    assert rows[0]["lookback_gain"] > 0
    assert rows[0]["source_special"]
    assert not rows[0]["source_in_lookback_set"]
    assert rows[0]["selected_mass_change_upper"] < 0
    assert rows[0]["retained_scope_mass_lower_bound"] == .3


def test_unretained_previous_attention_is_bounded_not_filled_with_zero():
    graph = dict(source=np.array([[[[0, 0], [0, 1], [0, 2], [0, 1]]]]),
                 attention=np.array([[[[1., 0.], [.8, .2], [.7, .3], [.7, .2]]]]))
    lower, upper = previous_mass_bounds(graph, 0, 0, 3, 1)
    assert lower == 0. and upper == .3
    assert previous_mass_bounds(graph, 0, 0, 3, 2) == (.3, .3)
