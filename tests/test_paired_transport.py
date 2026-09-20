"""Scientific checks for causal indexing, paired identity and finite head effects."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_target_transport import native_model
from test_path_conflict import WordTokenizer
from experiments.path_conflict.native import Intervention
from experiments.path_conflict.paired import run_phase
from experiments.path_conflict.paired_inputs import compile_pair, phase_positions, source_groups
from experiments.path_conflict.paired_plan import choose_paired_heads, head_units, joint_plan, target_head_scores
from experiments.path_conflict.paired_report import paired_differences, source_intervals, classify_interactions, report_pairs
from experiments.path_conflict.paired_screen import confidence_controls
from experiments.path_conflict.paired_trials import unit_action
from experiments.path_conflict.scoring import evaluate_candidates


def cached_trace():
    rng = np.random.default_rng(4)
    prompt, length, layers, heads = 3, 7, 3, 4
    attention = np.zeros((layers, heads, length, prompt + length))
    for position in range(length):
        attention[:, :, position, :prompt + position] = rng.dirichlet(
            np.ones(prompt + position), size=(layers, heads))
    return dict(prompt_length=np.array(prompt), attention=attention,
                logit_entropy=np.linspace(.2, 1., length))


def test_confidence_controls_use_saved_prediction_rows_and_no_future():
    trace = cached_trace()
    scores, selected, previous = confidence_controls(trace, 41)
    np.testing.assert_array_equal(previous[:, :, 2], trace["attention"][:, :, 2, 4].astype(np.float32))
    expected = np.log(41) - trace["logit_entropy"][0] * np.log(2)
    np.testing.assert_allclose(scores[0], -np.log(expected))
    trace["attention"][:, :, 4:] = 100
    trace["logit_entropy"][4:] = 100
    changed, changed_heads, _ = confidence_controls(trace, 41)
    np.testing.assert_array_equal(scores[:4], changed[:4])
    np.testing.assert_array_equal(selected[:4], changed_heads[:4])


def test_phases_and_claim_history_never_include_unconsumed_target():
    roles = dict(scope=np.array([0]), supported_value=np.array([1]), value_source=np.array([2]))
    groups = source_groups(3, 6, roles, (2, 5))
    np.testing.assert_array_equal(groups["history"], [3, 4, 5])
    np.testing.assert_array_equal(groups["query_self"], [5])
    np.testing.assert_array_equal(groups["prior_history"], [3, 4])
    np.testing.assert_array_equal(groups["claim_history"], [5])
    assert 6 not in groups["head_total"]
    phases = phase_positions((0, 1), 2, np.array([False, True]))
    assert phases == {"onset": 0}  # No invented before/back-half or EOS recovery.


def test_saved_tokens_define_both_sides_without_inventing_a_missing_phase(tmp_path):
    tokenizer = WordTokenizer()
    prompt_ids = tokenizer.encode("Only ruler. Ordinary caps. Royal gold. Output:")
    samples = []
    for seed, response in [(0, "They wear caps with cloth. End"), (1, "They instead wear gold")]:
        ids = prompt_ids + tokenizer.encode(response)
        length = len(ids) - len(prompt_ids)
        name = f"{seed}.npz"
        np.savez(tmp_path / name, token_ids=ids, prompt_length=len(prompt_ids),
                 token_text=[tokenizer.decode([token]) for token in ids],
                 chosen_logit=np.zeros(length), log_normalizer=np.zeros(length),
                 top_ids=np.zeros((length, 5), dtype=int), top_logits=np.zeros((length, 5)),
                 special_mask=np.zeros(len(ids), dtype=bool))
        samples.append(dict(source_id="one", seed=seed, trace=name, response=response))
    case = dict(case_id="claim", source_id="one", supported=dict(seed=0, target="caps with cloth"),
                unsupported=dict(seed=1, target="gold"), candidates=[" caps with cloth", " gold"],
                evidence=["Only ruler.", "Ordinary caps."], value_source=["Royal gold."],
                source_roles=dict(scope=["Only ruler."], supported_value=["Ordinary caps."],
                                  value_source=["Royal gold."]))
    pair = compile_pair(tmp_path, tokenizer, samples, case)
    assert set(pair["supported"]) == {"before_claim", "onset", "back_half", "post_claim"}
    assert set(pair["unsupported"]) == {"before_claim", "onset"}
    for phases in pair.values():
        assert tokenizer.decode(phases["onset"]["prefix_ids"]).endswith("wear")
        assert phases["onset"]["groups"]["claim_history"].size == 0
        for probe in phases.values():
            assert probe["prefix_ids"][:len(prompt_ids)] == prompt_ids
            assert probe["groups"]["head_total"][-1] == len(probe["prefix_ids"]) - 1


def test_selection_keeps_both_sides_and_samples_controls_outside_gradient_table():
    left = pd.DataFrame([dict(layer=0, head=1, final_linear_support=3.)])
    right = pd.DataFrame([dict(layer=2, head=3, final_linear_support=-4.)])
    heads = choose_paired_heads([left, right], 3, 4, 2, 3, 7)
    assert {(row["layer"], row["head"]) for row in heads[:2]} == {(0, 1), (2, 3)}
    assert all(row["selection"] == "model_head_control" for row in heads[2:])
    assert len({(row["layer"], row["head"]) for row in heads}) == 5
    arrays = dict(target_sources=np.array([[[1., -2.], [0., .1]]]))
    assert target_head_scores(arrays).final_linear_support.tolist() == [3., .1]


def test_conditional_sign_and_redundancy_do_not_follow_interaction_sign():
    frame = pd.DataFrame([
        dict(left_support=0., right_support=0., joint_support=2., interaction=-2.,
             left_conditional=2., right_conditional=2.),
        dict(left_support=1., right_support=-1., joint_support=0., interaction=0.,
             left_conditional=1., right_conditional=-1.),
    ])
    result = classify_interactions(frame, .05)
    assert result.small_singles_large_joint.tolist() == [True, False]
    assert result.opposed_finite_effects.tolist() == [False, True]
    assert result.nonadditive_effect.tolist() == [True, False]


def paired_probe(probe, phase="onset"):
    result = dict(probe, phase=phase, position=2, actual_token=7, claim_span=[2, 5],
                  readout="sequence_margin" if phase == "onset" else "observed_logp")
    result["groups"] = dict(evidence=np.array([0, 1]), inapplicable_source=np.array([2]),
                           history=np.array([3, 4]), claim_history=np.array([], dtype=int),
                           query_self=np.array([4]), prior_history=np.array([3]),
                           head_total=np.arange(5))
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_each_candidate_branch_restores_its_own_message_and_mlp(native_model, dtype):
    model, original = native_model
    model.to(dtype=dtype)
    probe = paired_probe(original)
    heads = [dict(layer=0, head=1, selection="paired_gradient")]
    unit = head_units(heads, probe)[-1]
    probe["capture_units"] = [unit]
    baseline, run = evaluate_candidates(model, probe)
    for site in ("message", "mlp"):
        name = ("mlp_" if site == "mlp" else "") + unit["unit"]
        replacement = {branch: writes[name] for branch, writes in run.branch_message_writes.items()}
        assert set(replacement) == {"prefix", "correct", "wrong"}
        action = unit_action(unit, operation="replace", replacement=replacement)
        if site == "mlp":
            action = Intervention(unit["layer"], ("mlp",), queries=(unit["receiver"],),
                                  operation="replace", replacement=replacement)
        sham, _ = evaluate_candidates(model, probe, (action,))
        assert sham["sequence_margin"] == pytest.approx(baseline["sequence_margin"], abs=1e-7)
    # Missing branch data must fail, rather than silently reusing prefix data.
    del replacement["wrong"]
    with pytest.raises(KeyError, match="wrong"):
        evaluate_candidates(model, probe, (action,))


def test_real_llama_paired_worlds_conditional_effects_and_resume(native_model, tmp_path, monkeypatch):
    model, original = native_model
    probe = paired_probe(original)
    heads = [dict(layer=0, head=1, selection="paired_gradient"),
             dict(layer=2, head=2, selection="paired_gradient")]
    pairs = joint_plan(heads)
    args = SimpleNamespace(doses=[.25, 1.], seed=0, sham_atol=1e-5, mediation_pairs=1)
    directory = tmp_path / "pairs" / "claim" / "supported" / "onset"
    identity = dict(case_id="claim", source_id="source", side="supported", phase="onset")
    run_phase(model, probe, heads, pairs, head_units(heads, probe), identity, directory, args)
    rows = pd.read_csv(directory / "interactions.csv")
    np.testing.assert_allclose(rows.interaction, rows.full - rows.without_left - rows.without_right + rows.without_both, atol=1e-12)
    assert rows.numeric_ok.all()
    adaptation = pd.read_csv(directory / "adaptation.csv")
    assert set(adaptation.site) == {"message", "mlp"}
    assert adaptation.sham_error.max() < 1e-7
    coverage = pd.read_csv(directory / "coverage.csv")
    assert not coverage[coverage.unit.str.endswith("claim_history")].measured.any()
    def no_forward(*args, **kwargs):
        raise AssertionError("resume ran the model")
    monkeypatch.setattr(model, "forward", no_forward)
    run_phase(model, probe, heads, pairs, head_units(heads, probe), identity, directory, args)
    pd.DataFrame([dict(case_id="claim", source_id="source")]).to_csv(tmp_path / "pair_inventory.csv", index=False)
    report = report_pairs(tmp_path, repeats=20)
    assert report["sources"] == 1 and sum(report["failed_control_rows"].values()) == 0
    assert (tmp_path / "paired_review.tar.gz").is_file()


def test_source_statistics_do_not_count_panels_as_independent_sources():
    rows = []
    for source, count, delta in [("s1", 5, 1.), ("s2", 1, 3.)]:
        for case in range(count):
            for side, value in [("supported", 0.), ("unsupported", delta)]:
                rows.append(dict(source_id=source, case_id=source + str(case), side=side, head=2, support=value))
    paired = paired_differences(pd.DataFrame(rows), ["head"], "support")
    summary = source_intervals(paired, ["head"], repeats=30)
    assert summary.effect.iloc[0] == 2.  # Not the panel-weighted mean 8/6.
    assert summary.sources.iloc[0] == 2
    one = source_intervals(paired[paired.source_id == "s1"], ["head"], repeats=30)
    assert np.isnan(one.lower.iloc[0])


def test_onset_carry_changes_later_readout_only_through_remaining_layers(native_model, tmp_path):
    model, original = native_model
    onset = paired_probe(original)
    later = paired_probe(original, phase="back_half")
    later.update(prefix_ids=original["prefix_ids"] + [7, 8], candidates=[[13], [13]],
                 position=4, actual_token=13)
    later["groups"].update(history=np.arange(3, 7), claim_history=np.array([5, 6]),
                           query_self=np.array([6]), prior_history=np.arange(3, 6),
                           head_total=np.arange(7))
    heads = [dict(layer=0, head=1, selection="paired_gradient"),
             dict(layer=2, head=2, selection="paired_gradient")]
    args = SimpleNamespace(doses=[1.], seed=0, sham_atol=1e-5, mediation_pairs=0)
    identity = dict(case_id="claim", source_id="source", side="unsupported", phase="back_half")
    run_phase(model, later, heads, joint_plan(heads), head_units(heads, onset),
              identity, tmp_path, args)
    carried = pd.read_csv(tmp_path / "persistence.csv")
    assert set(carried.receiver) == {4}  # Earlier onset query, not current query 6.
    assert (carried.null_error == 0).all()
    # The final layer at a past position has no remaining cross-position path.
    assert carried[carried.layer == 2].support.iloc[0] == 0
    assert abs(carried[carried.layer == 0].support.iloc[0]) > 1e-8
    effects = pd.read_csv(tmp_path / "effects.csv")
    assert set(effects.receiver) == {6}
    assert set(effects.readout) == {"observed_logp"}
