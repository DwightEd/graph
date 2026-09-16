"""Exercise the actual frozen CHARM, preserved-entropy controls and saved-run CLI."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import expit
import torch

from experiments.charm_structure_audit.model import CHARM
from experiments.charm_structure_audit.learned_audit.controls import (
    edge_groups, permute_weights, compare_inputs, match_sources)
from experiments.charm_structure_audit.learned_audit.features import graph_features, head_overlap
from experiments.charm_structure_audit.learned_audit.messages import capture_states, predict, replace_source_states
from experiments.charm_structure_audit.learned_audit.explain import fit_explanations
from experiments.charm_structure_audit.learned_audit.run import main, load_sample


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def graph(seed=2, layers=2, heads=2):
    rng = np.random.default_rng(seed)
    count, prompt = 14, 2
    source, target = [], []
    for query in range(prompt, count):
        source.extend(range(query))
        target.extend([query] * query)
    channels = layers * heads
    weights = rng.uniform(.06, .15, (len(source), channels)).astype(np.float32)
    source, target = np.asarray(source), np.asarray(target)
    labels = np.array([False, True, True, False, True, False, False, True, False, False, True, False])
    onset = labels & ~np.r_[False, labels[:-1]]
    offsets = np.column_stack((np.arange(count - prompt), np.arange(1, count - prompt + 1)))
    return dict(x=rng.uniform(.1, .3, (count, channels)).astype(np.float32),
        edge_index=np.stack((source, target)), edge_attr=weights,
        edge_mark=np.column_stack((source < prompt, source >= prompt)).astype(np.float32),
        prompt_length=np.asarray(prompt), layers=np.asarray(layers), heads=np.asarray(heads),
        token_ids=np.arange(count), gold=labels, onset=onset, offsets=offsets,
        response=np.asarray("abcdefghijkl"), spans=np.array([[1, 3], [4, 5], [7, 8], [10, 11]]))


def model(channels=4, layers=3):
    torch.manual_seed(8)
    return CHARM(channels, channels, dict(hidden_dim=8, gnn_layers=layers), "out", edge_chunk=3).eval()


def test_entropy_matches_normalized_saved_row():
    g = graph()
    features, columns = graph_features(g)
    query, channel = 7, 2
    values = np.r_[g["edge_attr"][g["edge_index"][1] == query, channel], g["x"][query, channel]].astype(float)
    values /= values.sum()
    expected = -(values * np.log(values)).sum()
    assert features[query - 2, columns["entropy"][channel]] == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["vector", "layer", "head", "positive_head"])
def test_per_head_distributions_entropy_degrees_and_input_are_preserved(mode):
    g = graph()
    before = {name: value.copy() for name, value in g.items()}
    changed = permute_weights(g, mode, 3)
    for group in edge_groups(g):
        np.testing.assert_array_equal(np.sort(g["edge_attr"][group], axis=0), np.sort(changed["edge_attr"][group], axis=0))
    np.testing.assert_array_equal(g["edge_index"], changed["edge_index"])
    np.testing.assert_array_equal(g["x"], changed["x"])
    assert compare_inputs(g, changed)["max_entropy_change"] < 1e-5
    for name, value in before.items():
        np.testing.assert_array_equal(g[name], value)


def test_whole_vectors_and_layer_permutations_keep_within_layer_agreement():
    g = graph()
    original, observed = head_overlap(g)
    for mode in ("vector", "layer"):
        altered, covered = head_overlap(permute_weights(g, mode, 3))
        np.testing.assert_array_equal(observed, covered)
        np.testing.assert_allclose(original, altered, atol=1e-6)
    altered, _ = head_overlap(permute_weights(g, "head", 3))
    assert np.max(abs(original - altered)) > 1e-3


def test_positive_head_keeps_zero_locations():
    g = graph()
    g["edge_attr"][::3, 0] = 0
    g["edge_attr"][1::3, 1] = 0
    changed = permute_weights(g, "positive_head", 2)
    np.testing.assert_array_equal(g["edge_attr"] > 0, changed["edge_attr"] > 0)
    assert compare_inputs(g, changed)["changed_support"] == 0


def test_head_names_preserve_per_layer_entropy_multiset():
    g = graph(heads=4)
    a, columns = graph_features(g)
    b, _ = graph_features(permute_weights(g, "head_names", 2))
    np.testing.assert_allclose(np.sort(a[:, columns["entropy"]].reshape(12, 2, 4), axis=2),
                               np.sort(b[:, columns["entropy"]].reshape(12, 2, 4), axis=2), atol=1e-6)


def test_single_head_controls_cannot_claim_within_layer_collaboration():
    g = graph(heads=1)
    left = permute_weights(g, "layer", 9)
    right = permute_weights(g, "head", 9)
    np.testing.assert_array_equal(left["edge_attr"], right["edge_attr"])
    assert not head_overlap(g)[1].any()


def test_controls_never_inspect_gold():
    g = graph()
    a = permute_weights(g, "head", 2)
    g["gold"] = ~g["gold"]
    b = permute_weights(g, "head", 2)
    np.testing.assert_array_equal(a["edge_attr"], b["edge_attr"])


def test_source_donors_share_eligibility_and_match_roles():
    g = graph()
    features, _ = graph_features(g)
    features[:] = 0  # Matching fixture holds the coarse descriptors identical.
    donors = match_sources(g, features)
    source, target = g["edge_index"]
    valid = donors["eligible"]
    assert valid.any()
    same, other = donors["same_label"], donors["other_label"]
    np.testing.assert_array_equal(same[~valid], source[~valid])
    np.testing.assert_array_equal(other[~valid], source[~valid])
    np.testing.assert_array_equal(g["gold"][same[valid] - 2], g["gold"][source[valid] - 2])
    assert np.all(g["gold"][other[valid] - 2] != g["gold"][source[valid] - 2])
    assert np.all(same < target) and np.all(other < target)


def test_actual_native_model_identity_patch_preserves_logits_and_no_gradients():
    g, m = graph(), model()
    original_parameters = {key: value.clone() for key, value in m.state_dict().items()}
    base = predict(m, g)
    captured, states = capture_states(m, g)
    np.testing.assert_allclose(base, captured)
    for layer in range(3):
        changed = replace_source_states(m, g, layer, states, edge_groups(g), g["edge_index"][0])
        np.testing.assert_allclose(base, changed, atol=1e-6)
        assert not m.mp_layers[layer].msg_mlp._forward_pre_hooks
    for key, value in m.state_dict().items():
        torch.testing.assert_close(value, original_parameters[key])
    assert all(parameter.grad is None for parameter in m.parameters())


def test_source_patch_cleans_hook_after_exception(monkeypatch):
    from experiments.charm_structure_audit.learned_audit import messages
    g, m = graph(), model()
    _, states = capture_states(m, g)
    def fail(*args):
        raise RuntimeError("forced test failure")
    monkeypatch.setattr(messages, "predict", fail)
    with pytest.raises(RuntimeError, match="forced"):
        replace_source_states(m, g, 1, states, edge_groups(g))
    assert not m.mp_layers[1].msg_mlp._forward_pre_hooks


def test_known_cross_head_function_changes_while_entropy_function_does_not():
    g = graph(layers=1, heads=2)
    m = model(channels=2, layers=1)
    with torch.no_grad():
        for parameter in m.parameters():
            parameter.zero_()
        # MLP detects both heads simultaneously strong on the SAME message slot.
        message = m.mp_layers[0].msg_mlp
        message[0].weight[0, 8:10] = 1
        message[0].bias[0] = -.19
        message[2].weight[0, 0] = 1
        update = m.mp_layers[0].up_mlp
        update[0].weight[0, 8] = 1
        update[2].weight[0, 0] = 1
        m.pred[0].weight[0, 0] = 1
        m.pred[3].weight[0, 0] = 1
    original = predict(m, g)
    coordinated = permute_weights(g, "vector", 3)
    independent = permute_weights(g, "head", 3)
    np.testing.assert_allclose(original, predict(m, coordinated), atol=1e-6)
    assert np.max(abs(original - predict(m, independent))) > 1e-4
    entropy_a, columns = graph_features(g)
    entropy_b, _ = graph_features(independent)
    np.testing.assert_allclose(entropy_a[:, columns["entropy"]], entropy_b[:, columns["entropy"]], atol=1e-6)
    # Any fixed linear function of these entropies is unchanged by the control.
    np.testing.assert_allclose(entropy_a[:, columns["entropy"]].sum(1), entropy_b[:, columns["entropy"]].sum(1), atol=1e-6)


@pytest.fixture
def saved_run(tmp_path):
    prepared, model_dir = tmp_path / "data", tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "test/samples").mkdir(parents=True)
    m = model()
    checkpoint = model_dir / "checkpoint.pt"
    torch.save(dict(model_state=m.state_dict(), hp=dict(hidden_dim=8, gnn_layers=3)), checkpoint)
    records = []
    parts = dict(fit=["0", "1", "2", "3"], select=["4", "5"], calibration=["6"], test=["7", "8"])
    for index in range(9):
        g = graph(seed=index)
        split = "test" if index >= 7 else "train"
        row = dict(id=str(index), source_id="source_" + str(index), split=split)
        folder = prepared / "graphs" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{index}.npz", **g)
        records.append(row)
        if split == "test":
            np.savez_compressed(model_dir / "test/samples" / f"{index}.npz", score=expit(predict(m, g)),
                                gold=g["gold"], offsets=g["offsets"])
    (prepared / "index.json").write_text(json.dumps(records))
    (model_dir / "training.json").write_text(json.dumps(dict(variant="charm_out", partitions=parts)))
    (model_dir / "test/prediction_settings.json").write_text(json.dumps(dict(records=parts["test"], threshold=dict(value=.6))))
    return prepared, model_dir, tmp_path / "audit"


def arguments(saved_run, stage="all"):
    prepared, model_dir, output = saved_run
    return ["--prepared", str(prepared), "--model-dir", str(model_dir), "--output", str(output),
            "--device", "cpu", "--seeds", "0", "--bootstrap", "2", "--stage", stage]


def test_real_saved_checkpoint_end_to_end_and_resume(saved_run):
    prepared, model_dir, output = saved_run
    before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for folder in (prepared, model_dir)
              for file in folder.rglob("*") if file.is_file()}
    main(arguments(saved_run))
    assert (output / "score_explanation.json").exists()
    assert (output / "token_effects.csv").exists()
    results = json.loads((output / "mechanisms.json").read_text())
    assert "head_s0_minus_layer_s0" in results["paired_contrasts"]
    assert "other_label_g0_minus_same_label_g0" in results["paired_contrasts"]
    assert results["responses"] == 2
    original_time = (output / "samples/7.npz").stat().st_mtime_ns
    main(arguments(saved_run, "intervene"))
    assert (output / "samples/7.npz").stat().st_mtime_ns == original_time
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in before} == before


def test_mismatched_original_scores_stop_attribution(saved_run):
    prepared, model_dir, output = saved_run
    g = graph(seed=7)
    np.savez_compressed(model_dir / "test/samples/7.npz", score=np.zeros(12), gold=g["gold"], offsets=g["offsets"])
    with pytest.raises(AssertionError, match="no longer reproduces"):
        main(arguments(saved_run, "intervene"))
    assert not (output / "samples/7.npz").exists()


def test_cli_runs_cpu_interventions_without_tokenizer(saved_run):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "-m", "experiments.charm_structure_audit.learned_audit.run",
        *arguments(saved_run, "intervene")], cwd=root, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert (saved_run[2] / "README_RESULTS.md").exists()


def test_probability_ties_are_not_broken_by_using_logits_for_baseline():
    from experiments.charm_structure_audit.learned_audit.report import binary_metrics
    measured = binary_metrics([0, 1], np.array([99., 100.]), .5, scores=np.array([1., 1.]))
    assert measured["auroc"] == .5
    assert measured["ap"] == .5


def test_explanation_selection_ignores_gold_and_preserves_context_columns():
    from experiments.charm_structure_audit.learned_audit.explain import fit_explanations
    rows = []
    columns = None
    for index in range(5):
        g = graph(seed=index)
        features, columns = graph_features(g)
        rows.append(dict(id=str(index), gold=g["gold"], features=features,
                         logits=2 * features[:, 9] - features[:, 3]))
    parts = dict(fit=["0", "1", "2"], select=["3"], test=["4"])
    _, a = fit_explanations(rows, parts, columns, maximum=12)
    for row in rows:
        row["gold"] = ~row["gold"]
    rows[-1]["logits"][:] = 999
    _, b = fit_explanations(rows, parts, columns, maximum=12)
    assert a == b
    assert all(set(range(9)) <= set(row["selected_columns"]) for row in a.values())


def test_partial_report_reads_only_finalized_samples(saved_run):
    main(arguments(saved_run, "intervene"))
    output = saved_run[2]
    (output / "samples/8.npz").unlink()
    (output / "samples/8.partial").write_bytes(b"unfinished")
    main(arguments(saved_run, "report") + ["--completed-only"])
    report = json.loads((output / "mechanisms.json").read_text())
    assert report["responses"] == 1
    assert report["completed_preview"] is True


def test_shell_reuses_saved_diagnosis_and_passes_paths(tmp_path):
    import os
    tool = tmp_path / "fake python"
    log = tmp_path / "calls.jsonl"
    tool.write_text("#!/usr/bin/env python\nimport os,sys,json\nwith open(os.environ['CALL_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    tool.chmod(0o755)
    env = dict(os.environ, PY=str(tool), CALL_LOG=str(log), MODEL_DIR="model with spaces", OUTPUT="audit folder", STAGE="report")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(["bash", "experiments/charm_structure_audit/learned_audit/run.sh"],
                               cwd=root, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][2] == "experiments.charm_structure_audit.diagnose_saved"
    assert calls[1][2] == "experiments.charm_structure_audit.learned_audit.run"
    assert "model with spaces" in calls[1]
    assert "audit folder" in calls[1]


@pytest.mark.parametrize("bias,should_change", [(0., False), (-.19, True)])
def test_joint_term_removal_distinguishes_additive_from_interacting_messages(bias, should_change):
    from experiments.charm_structure_audit.learned_audit.messages import remove_head_interaction
    g = graph(layers=1, heads=2)
    m = model(channels=2, layers=1)
    with torch.no_grad():
        for parameter in m.parameters():
            parameter.zero_()
        m.mp_layers[0].msg_mlp[0].weight[0, 8:10] = 1
        m.mp_layers[0].msg_mlp[0].bias[0] = bias
        m.mp_layers[0].msg_mlp[2].weight[0, 0] = 1
        m.mp_layers[0].up_mlp[0].weight[0, 8] = 1
        m.mp_layers[0].up_mlp[2].weight[0, 0] = 1
        m.pred[0].weight[0, 0] = 1
        m.pred[3].weight[0, 0] = 1
    before = predict(m, g)
    after, details = remove_head_interaction(m, g, 0, 0, 1)
    assert details["edges"] == g["edge_index"].shape[1]
    assert not m.mp_layers[0].msg_mlp._forward_hooks
    if should_change:
        assert np.max(abs(before - after)) > 1e-4
        assert details["mean_message_interaction_norm"] > 0
    else:
        np.testing.assert_allclose(before, after, atol=1e-6)
        assert details["mean_message_interaction_norm"] < 1e-6


def test_head_pair_screening_uses_only_fit_logits():
    from experiments.charm_structure_audit.learned_audit.explain import screen_head_pairs
    rows = []
    for index in range(3):
        features, _ = graph_features(graph(seed=index, heads=4))
        rows.append(dict(id=str(index), features=features, logits=features[:, 10] + features[:, 11]))
    parts = dict(fit=["0", "1"], test=["2"])
    before = screen_head_pairs(rows, parts, 2, 4)
    rows[2]["logits"][:] = -1e9
    rows[2]["features"][:] = 0
    assert screen_head_pairs(rows, parts, 2, 4) == before
