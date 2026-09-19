"""Scientific invariants: mass identities, signed writes and four-world effects."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_path_conflict import fixture
from test_path_conflict_focused import focused_fixture
from experiments.path_conflict.operators import local_readout_direction, lens_margin
from experiments.path_conflict.native import forward
from experiments.path_conflict.flow import final_head_trials
from experiments.path_conflict.flow_inputs import add_flow_groups
from experiments.path_conflict.cooperation import select_pairs, run_pairs, interaction_values
from experiments.path_conflict.flow_report import supervised_alignment
from experiments.ragtruth_flow.grounding_dynamics import (
    transition_xy, closure_residual, route_row, load_dynamics, score_answer,
    freeze_run_settings,
)


def test_current_complementary_masses_are_a_target_identity():
    rng = np.random.default_rng(3)
    routes = rng.dirichlet(np.ones(4), size=(20, 2, 3))
    changes = np.diff(routes, axis=0)
    np.testing.assert_allclose(changes[..., 3], -changes[..., :3].sum(-1), atol=1e-15)
    sparse = routes * rng.uniform(.3, .9, size=(20, 2, 3, 1))
    sparse_changes = np.diff(sparse, axis=0)
    np.testing.assert_allclose(sparse_changes.sum(-1), closure_residual(sparse), atol=1e-15)


def test_forecast_never_reads_current_or_future_routes():
    rng = np.random.default_rng(5)
    routes = rng.dirichlet(np.ones(4), size=(8, 2, 3))
    before, _ = transition_xy(routes)
    routes[4:] = 99
    after, _ = transition_xy(routes)
    np.testing.assert_array_equal(before[:4], after[:4])


def test_prompt_last_query_is_not_double_counted():
    answer = SimpleNamespace(prompt_length=3)
    row = route_row(answer, 0, np.arange(3), np.array([.2, .3, .5]), np.ones(3, bool))
    np.testing.assert_allclose(row, [.5, 0, 0, .5])


def test_old_grounding_model_cannot_resume_new_forecast(tmp_path):
    path = tmp_path / "model.npz"
    np.savez(path, geometry=[2, 4], regression=np.zeros((2, 16, 4)))
    with pytest.raises(ValueError, match="past-only"):
        load_dynamics(path)


def test_forecast_resume_keeps_frozen_population(tmp_path):
    args = SimpleNamespace(train_cache=tmp_path / "train", test_cache=tmp_path / "test",
                           tokenizer="observer", grounding_ridge=.001)
    freeze_run_settings(tmp_path, args, ["a"], ["b"])
    freeze_run_settings(tmp_path, args, ["a"], ["b"])
    with pytest.raises(ValueError, match="inputs/settings changed"):
        freeze_run_settings(tmp_path, args, ["a", "c"], ["b"])


def test_local_common_direction_matches_gradient_and_adds():
    model, _ = fixture()
    value = torch.randn(16, requires_grad=True)
    expected, = torch.autograd.grad(lens_margin(model, value, 7, 9), value)
    actual = local_readout_direction(model, value.detach(), 7, 9)
    torch.testing.assert_close(actual, expected)
    messages = torch.randn(4, 16)
    torch.testing.assert_close((messages @ actual).sum(), messages.sum(0) @ actual)


def test_native_llama_rmsnorm_direction_matches_autograd():
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    model, _ = fixture()
    model.model.norm = LlamaRMSNorm(16)
    value = torch.randn(16, requires_grad=True)
    expected, = torch.autograd.grad(lens_margin(model, value, 7, 9), value)
    actual = local_readout_direction(model, value.detach(), 7, 9)
    torch.testing.assert_close(actual, expected)


def test_native_observation_sums_head_projections_not_lens_deletions():
    model, probe = fixture()
    _, run = forward(model, probe["prefix_ids"], probe)
    writes = pd.DataFrame(run.writes)
    for _, rows in writes.groupby(["layer", "source_group"]):
        heads = rows[rows["head"].ge(0)].local_linear_support.sum()
        total = rows[rows["head"].eq(-1)].local_linear_support.iloc[0]
        assert heads == pytest.approx(total, abs=2e-6)


def test_four_world_interaction_direction():
    # F(a,b)=a+b+3ab. Both retained has extra +3 support.
    values = interaction_values(5., 1., 1., 0.)
    assert values == dict(left_support=4., right_support=4., joint_support=5., interaction=3.)


def test_same_shape_different_models_cannot_align_heads(tmp_path):
    path = tmp_path / "heads.csv"
    pd.DataFrame(dict(layer=[31], head=[31])).to_csv(path, index=False)
    output = supervised_alignment(pd.DataFrame(dict(layer=[31], head=[31])), path)
    assert "not aligned" in output.iloc[0].note


def test_real_small_model_head_coalitions_and_resume(tmp_path):
    model, probe = focused_fixture()
    probe = add_flow_groups(probe)
    identity = dict(case_id="c", source_id="s", side="supported", panel="natural",
                    seed=0, trace="t.npz", query=4)
    from experiments.path_conflict.scoring import evaluate_candidates
    baseline, run = evaluate_candidates(model, probe)
    (tmp_path / "runs").mkdir()
    heads = [(0, 1), (1, 2)]
    final_head_trials(model, probe, identity, baseline, heads, tmp_path)
    plans = select_pairs(pd.DataFrame(run.writes), heads, pairs_per_group=1)
    results = run_pairs(model, probe, identity, baseline, plans, tmp_path)
    assert len(results) == 4
    assert {row["metric"] for row in results} == {"next_margin", "sequence_margin"}
    repeated = run_pairs(model, probe, identity, baseline, plans, tmp_path)
    assert results == repeated
    for row in results:
        assert np.isfinite(row["interaction"])


def test_forecast_scoring_does_not_access_gold():
    class Answer:
        response_id = "a"
        source_id = "s"
        task = "QA"
        generator = "g"
        response_ids = np.arange(4)
        text = "abcd"
        offsets = [(i, i + 1) for i in range(4)]

        @property
        def error_mask(self):
            raise AssertionError("scorer accessed natural labels")

    rng = np.random.default_rng(6)
    routes = rng.dirichlet(np.ones(4), size=(4, 1, 2))
    models = [(np.zeros((8, 2)), np.zeros(2))]
    reference = [(np.zeros(2), np.eye(2))]
    result = score_answer(Answer(), routes, models, reference, reference)
    assert "gold" not in result and "previous_gold" not in result
    assert np.isfinite(result.raw_surprise).all()
