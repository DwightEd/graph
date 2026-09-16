"""Synthetic scientific-contract tests, not natural-data performance evidence."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from experiments.charm_structure_audit.graph import degree, prefix_graph
from experiments.charm_structure_audit.model import CHARM
from experiments.charm_structure_audit.deep_audit.common import (
    load_predictions, membership, merged_spans, metric, roles, validate_sample, win_credit,
)
from experiments.charm_structure_audit.deep_audit.controls import (
    degree_preserving_rewire, edge_groups, model_graph, oracle_span_cut,
    permute_head_identity, shuffle_endpoints, subset_edges,
)
from experiments.charm_structure_audit.deep_audit.representations import (
    features, head_overlap, matched_relation_contrasts, relation_rows, retained_entropy, trace_model,
)
from experiments.charm_structure_audit.deep_audit.scores import (
    auc_accounting, control_comparison, population, span_comparisons, token_ledger, within_answers,
)


def sample(identity="s", count=14, spans=((2, 5), (8, 11)), scores=None):
    words = [f"w{index}" for index in range(count)]
    response = " ".join(words)
    starts = np.cumsum([0] + [len(word) + 1 for word in words[:-1]])
    offsets = np.column_stack((starts, starts + [len(word) for word in words]))
    gold = np.zeros(count, bool)
    onset = np.zeros(count, bool)
    for start, end in spans:
        gold[start:end] = True
        onset[start] = True
    score = np.linspace(.05, .95, count) if scores is None else np.asarray(scores, float)
    result = dict(id=str(identity), source_id="source_" + str(identity), task="QA", generator="synthetic",
                  split="test", response=response, offsets=offsets, gold=gold, onset=onset,
                  spans=np.asarray(spans, int).reshape(-1, 2), score=score)
    validate_sample(result)
    return result


def graph_fixture(seed=4, count=14, prompt=3):
    rng = np.random.default_rng(seed)
    count_all = prompt + count
    edges = [(source, target) for target in range(prompt, count_all) for source in range(target)
             if source < prompt or source == target - 1 or rng.random() < .40]
    source, target = np.asarray(edges, dtype=np.int64).T
    values = rng.uniform(.05, .2, (len(source), 8)).astype(np.float32)
    values[rng.random(values.shape) < .45] = 0.
    values[:, 0] = .06 + rng.random(len(source)) * .1  # Every union edge really exists.
    return dict(x=rng.uniform(.05, .25, (count_all, 8)).astype(np.float32),
                edge_index=np.stack((source, target)), edge_attr=values,
                edge_mark=np.stack((source < prompt, source >= prompt), axis=1).astype(np.float32),
                prompt_length=np.asarray(prompt), layers=np.asarray(2), heads=np.asarray(4))


def model_fixture(layers=2, normalization="in"):
    torch.manual_seed(9)
    return CHARM(8, 8, dict(hidden_dim=12, gnn_layers=layers, residual_mp=True),
                 normalization=normalization, edge_chunk=17).eval()


def snapshot(paths):
    return {str(path): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths}


def saved_experiment(root, training=False):
    """Real CHARM forward/checkpoint with synthetic graphs in the repository's actual schema."""
    prepared = root / "prepared"
    model_dir = root / "QA" / "seed_0" / "charm_in"
    predictions = model_dir / "test"
    (predictions / "samples").mkdir(parents=True)
    model = model_fixture()
    checkpoint = model_dir / "checkpoint.pt"
    torch.save(dict(model_state=model.state_dict(), hp=dict(hidden_dim=12, gnn_layers=2, residual_mp=True)), checkpoint)
    parts = dict(fit=["0", "1", "2"], select=["3"], calibration=["4"], test=["5", "6"])
    records = []
    for index in range(7):
        record = sample(str(index))
        record["split"] = "test" if index >= 5 else "train"
        graph = graph_fixture(seed=index)
        destination = prepared / "graphs" / record["split"] / f"{index}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {key: record[key] for key in ("id", "source_id", "task", "generator", "split")}
        metadata.update(graph=str(destination), response_tokens=len(record["gold"]), positives=int(record["gold"].sum()))
        labels = {key: record[key] for key in ("gold", "onset", "spans", "offsets", "response")}
        np.savez_compressed(destination, **graph, **labels, token_ids=np.arange(len(graph["x"])),
                            record_json=np.asarray(json.dumps(metadata)))
        records.append(metadata)
        if index >= 5:
            with torch.no_grad():
                score = torch.sigmoid(model(graph)[int(graph["prompt_length"]):]).numpy()
            np.savez_compressed(predictions / "samples" / f"{index}.npz", **labels, score=score,
                                record_json=np.asarray(json.dumps(metadata)),
                                embedding=np.array([{"must_not_load": True}], dtype=object))
    (prepared / "index.json").write_text(json.dumps(records))
    stamp = [str(checkpoint), checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns]
    settings = dict(checkpoint=stamp, records=parts["test"], variant="charm_in", seed=0,
                    threshold=dict(value=.5, rule="score > threshold", origin="synthetic_fixture"))
    (predictions / "prediction_settings.json").write_text(json.dumps(settings))
    (predictions / "predictions.json").write_text(json.dumps([f"/old/server/samples/{index}.npz" for index in parts["test"]]))
    if training:
        (model_dir / "training.json").write_text(json.dumps(dict(variant="charm_in", seed=0, partitions=parts)))
    return predictions, prepared, checkpoint, model


def test_population_counts_and_overlap_vs_adjacency():
    row = sample(count=9, spans=((1, 4), (3, 5), (5, 7)))
    assert merged_spans(row) == [[1, 5], [5, 7]]
    assert membership(row).tolist() == [-1, 0, 0, 0, 0, 1, 1, -1, -1]
    counts = population([row, sample("clean", count=9, spans=())])
    assert counts["tokens"] == 18 and counts["error_tokens"] == 6
    assert counts["raw_annotation_spans"] == 3 and counts["overlap_merged_spans"] == 2
    assert counts["contiguous_error_runs"] == 1
    assert [counts["roles"][key]["tokens"] for key in ("first_error", "later_onset", "continuation")] == [1, 2, 3]
    assert sum(mask.sum() for mask in roles(row).values()) == 9


def test_strict_threshold_and_one_class_metrics():
    measured = metric([1, 1, 0, 0], [.5, .8, .5, .8], .5)
    assert [measured[key] for key in ("tp", "fn", "fp", "tn")] == [1, 1, 1, 1]
    assert metric([1, 1], [.2, .9], .5)["auroc"] is None
    assert metric([0, 0], [.2, .9], .5)["auroc"] is None
    assert metric([], [], .5)["ap"] is None


def test_exact_auc_cell_and_token_accounting_with_ties():
    rows = [sample(count=8, spans=((1, 4), (6, 8)), scores=[.1, .2, .2, .8, .8, .5, .5, .9]),
            sample("b", count=6, spans=((3, 5),), scores=[.2, .4, .4, .4, .7, .7]),
            sample("c", count=4, spans=(), scores=[.1, .2, .3, .4])]
    gold = np.concatenate([row["gold"] for row in rows])
    score = np.concatenate([row["score"] for row in rows])
    auc = roc_auc_score(gold, score)
    accounting = auc_accounting(rows)
    assert accounting["reconstructed_auroc"] == pytest.approx(auc)
    assert sum(cell["pairs"] for cell in accounting["cells"]) == int(gold.sum()) * int((~gold).sum())
    assert sum(cell["contribution_above_chance"] for cell in accounting["cells"]) == pytest.approx(auc - .5)
    tokens = list(token_ledger(rows, .5))
    positives = [token for token in tokens if token["gold"]]
    assert np.mean([token["auroc_credit_vs_all_normal"] for token in positives]) == pytest.approx(auc)
    assert sum(token["contribution_above_chance"] for token in positives) == pytest.approx(auc - .5)
    assert win_credit(np.array([.5]), np.array([.2, .5, .8]))[0] == .5


def test_high_pooled_score_can_have_chance_within_answers():
    high = sample("high", count=10, spans=((0, 9),), scores=np.full(10, .9))
    low = sample("low", count=10, spans=((0, 1),), scores=np.full(10, .1))
    assert auc_accounting([high, low])["reconstructed_auroc"] == pytest.approx(.9)
    within, _ = within_answers([high, low], .5)
    row = next(r for r in within if r["role"] == "all_error" and r["negative_context"] == "all_normal")
    assert row["macro_auroc"] == .5 and row["pair_weighted_auroc"] == .5


def test_span_coverage_normal_gaps_and_no_within_span_auc():
    row = sample(count=10, spans=((1, 4), (6, 9)), scores=[.1, .2, .9, .9, .8, .9, .8, .9, .9, .1])
    spans, gaps = span_comparisons(row, .5, 2)
    assert spans[0]["within_span_auroc"] is None
    assert spans[0]["delay"] == 1 and not spans[0]["onset_hit"]
    assert spans[0]["coverage"] == pytest.approx(2 / 3)
    assert gaps[0]["tokens"] == 2 and gaps[0]["fpr"] == 1
    assert spans[0]["nearby_normal_tokens"] == 3
    all_error = sample(count=4, spans=((0, 4),))
    assert span_comparisons(all_error, .5, 2)[0][0]["same_answer_auroc"] is None


def test_prefix_comparison_uses_common_coordinates_and_source_draws():
    rows = [sample("a"), sample("b")]
    for row in rows:
        row["score_prefix"] = np.full(14, np.nan)
        row["score_prefix"][[0, 2, 5, 8]] = row["score"][[0, 2, 5, 8]]
    result = control_comparison(rows, "score_prefix", .5, bootstrap=12)
    assert result["common_tokens"] == 8 and result["total_tokens"] == 28
    assert result["baseline"] == result["control"]
    assert result["delta_ci95"] == [[0., 0.], [0., 0.]]
    for row in rows:
        row["score_prefix"][:] = np.nan
    empty = control_comparison(rows, "score_prefix", .5, bootstrap=3)
    assert empty["common_tokens"] == 0 and empty["control"]["auroc"] is None
    assert empty["delta_ci95"] is None


def test_shuffles_keep_per_head_marginals_but_break_co_location():
    graph = graph_fixture(count=36)
    original = copy.deepcopy(graph)
    coupled, coupled_info = shuffle_endpoints(model_graph(graph), 17, independent=False)
    independent, info = shuffle_endpoints(model_graph(graph), 17, independent=True)
    for group in edge_groups(graph):
        expected = np.sort(graph["edge_attr"][group], axis=0)
        np.testing.assert_array_equal(np.sort(coupled["edge_attr"][group], axis=0), expected)
        np.testing.assert_array_equal(np.sort(independent["edge_attr"][group], axis=0), expected)
    for altered in (coupled, independent):
        np.testing.assert_array_equal(altered["edge_index"], graph["edge_index"])
        np.testing.assert_allclose(retained_entropy(altered)[0], retained_entropy(graph)[0], atol=2e-6)
        np.testing.assert_allclose(retained_entropy(altered)[1], retained_entropy(graph)[1], atol=2e-6)
    np.testing.assert_allclose(head_overlap(coupled), head_overlap(graph), atol=1e-6)
    assert np.max(abs(head_overlap(independent) - head_overlap(graph))) > 1e-5
    assert info["changed_cells"] > 0 and coupled_info["eligible_edges"] > 0
    for key in graph:
        np.testing.assert_array_equal(graph[key], original[key])
    assert "gold" not in model_graph(dict(graph, gold=np.ones(36)))


def test_head_identity_is_not_the_co_location_control():
    graph = graph_fixture()
    changed, info = permute_head_identity(graph, 3)
    assert sorted(info["channel_order"]) == list(range(8))
    np.testing.assert_allclose(head_overlap(changed), head_overlap(graph), atol=1e-6)
    edge_only, _ = permute_head_identity(graph, 3, edge_only=True)
    np.testing.assert_array_equal(edge_only["x"], graph["x"])
    assert not np.array_equal(changed["x"], graph["x"])


def test_rewiring_preserves_degrees_attributes_causality_and_lag_bands():
    graph = graph_fixture(count=55)
    changed, info = degree_preserving_rewire(graph, seed=12, attempts_per_edge=12)
    source, target = graph["edge_index"]
    new_source, new_target = changed["edge_index"]
    assert info["accepted_swaps"] > 0 and info["changed_rr_edges"] > 0
    assert len(set(zip(new_source, new_target))) == len(source)
    assert np.all(new_source < new_target)
    np.testing.assert_array_equal(new_source[source < 3], source[source < 3])
    for normalization in ("in", "out"):
        np.testing.assert_array_equal(degree(graph, normalization), degree(changed, normalization))
    np.testing.assert_array_equal(graph["edge_attr"], changed["edge_attr"])
    np.testing.assert_array_equal(graph["edge_mark"], changed["edge_mark"])
    np.testing.assert_array_equal(np.floor(np.log2(target - source)), np.floor(np.log2(new_target - new_source)))
    empty = subset_edges(graph, source < 3)
    assert degree_preserving_rewire(empty, 0)[1]["changed_rr_fraction"] is None


def test_oracle_cut_and_matched_null_remove_equal_counts_per_group():
    graph = graph_fixture(count=24)
    row = sample(count=24, spans=((2, 10), (14, 22)))
    oracle, random, info = oracle_span_cut(model_graph(graph), row, 15)
    original_keys = graph["edge_index"][1] * len(graph["x"]) + graph["edge_index"][0]
    removed = []
    for view in (oracle, random):
        kept = view["edge_index"][1] * len(graph["x"]) + view["edge_index"][0]
        removed.append(~np.isin(original_keys, kept))
    for group in edge_groups(graph):
        assert removed[0][group].sum() == removed[1][group].sum()
    assert info["diagnostic_uses_gold"]
    assert info["removed_edges"] == removed[0].sum()
    assert 0 <= info["null_overlap_fraction"] <= 1
    assert info["exchangeable_removed_edges"] > 0


@pytest.mark.parametrize("normalization", ["in", "out"])
def test_trace_exactly_reproduces_original_model(normalization):
    graph = graph_fixture()
    model = model_fixture(normalization=normalization)
    with torch.no_grad():
        logits, hidden = model(graph, return_hidden=True)
    scores, states = trace_model(model, graph)
    np.testing.assert_allclose(scores, torch.sigmoid(logits[3:]).numpy(), atol=1e-7)
    np.testing.assert_allclose(states[-1], hidden[3:].numpy(), atol=1e-7)


def test_no_relay_preserves_one_hop_and_self_only_depth_updates():
    graph = graph_fixture()
    model = model_fixture(layers=1)
    np.testing.assert_allclose(trace_model(model, graph)[0], trace_model(model, graph, True)[0], atol=1e-7)
    model = model_fixture(layers=3)
    empty = subset_edges(graph, np.zeros(graph["edge_index"].shape[1], bool))
    np.testing.assert_allclose(trace_model(model, empty)[0], trace_model(model, empty, True)[0], atol=1e-7)
    assert np.max(abs(trace_model(model, graph)[0] - trace_model(model, graph, True)[0])) > 1e-8


def test_in_degree_prefix_invariant_but_out_degree_can_use_suffix():
    graph = graph_fixture(count=24)
    prefix = prefix_graph(graph, 15)
    assert degree(graph, "out")[14] != degree(prefix, "out")[14]
    model = model_fixture(normalization="in")
    np.testing.assert_allclose(trace_model(model, graph)[0][:12], trace_model(model, prefix)[0], atol=1e-7)
    model.normalization = "out"
    assert np.max(abs(trace_model(model, graph)[0][:12] - trace_model(model, prefix)[0])) > 1e-8


def test_same_span_contrast_never_substitutes_other_distances_or_answers():
    graph = graph_fixture(count=22)
    row = sample(count=22, spans=((1, 7), (8, 14), (16, 21)))
    _, states = trace_model(model_fixture(), graph)
    pairs = list(relation_rows(row, graph, features(graph, states), np.arange(len(graph["x"])), max_lag=10))
    contrasts = matched_relation_contrasts(pairs)
    assert contrasts
    for contrast in contrasts:
        matches = [pair for pair in pairs if all(pair[key] == contrast[key] for key in ("id", "lag", "connected", "same_token_id"))]
        kinds = {pair["relation"] for pair in matches}
        assert {"same_span", "different_error_spans"} <= kinds
    assert matched_relation_contrasts([pair for pair in pairs if pair["relation"] == "same_span"]) == []


def test_manifest_relocation_partial_handling_and_lazy_embedding(tmp_path):
    root, prepared, checkpoint, _ = saved_experiment(tmp_path)
    (root / "samples" / "unfinished.partial").write_bytes(b"not an npz")
    rows, settings = load_predictions(root)
    assert [row["id"] for row in rows] == ["5", "6"]
    assert all("embedding" not in row for row in rows)
    (root / "predictions.json").write_text(json.dumps(["/old/5.npz"]))
    with pytest.raises(ValueError, match="does not cover"):
        load_predictions(root)
    assert len(load_predictions(root, completed_only=True)[0]) == 2
    settings["records"] = ["5"]
    (root / "prediction_settings.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="outside"):
        load_predictions(root, completed_only=True)


def test_saved_score_cli_never_imports_torch_or_changes_inputs(tmp_path):
    root, prepared, _, _ = saved_experiment(tmp_path)
    paths = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = snapshot(paths)
    code = "from experiments.charm_structure_audit.deep_audit.run import main; import sys; main(sys.argv[1:]); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code, "--root", str(root), "--stage", "scores", "--bootstrap", "0"],
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(paths) == before
    report = json.loads((root / "deep_audit" / "saved_scores" / "scores.json").read_text())
    assert report["population"]["tokens"] == 28
    assert (root / "deep_audit" / "saved_scores" / "tokens.html").exists()
    inventory = json.loads((root / "deep_audit" / "dataset_inventory.json").read_text())
    assert sum(group["tokens"] for group in inventory["groups"].values()) == 98


def test_replay_gate_and_resume_keep_original_inputs_unchanged(tmp_path):
    from experiments.charm_structure_audit.deep_audit.capture import capture_samples

    root, prepared, checkpoint, _ = saved_experiment(tmp_path)
    samples, settings = load_predictions(root)
    paths = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = snapshot(paths)
    output = tmp_path / "replay"
    enriched, _ = capture_samples(samples, settings, output, prepared, repeats=1, max_lag=8)
    assert "score_prefix" in enriched[0] and np.isfinite(enriched[0]["score_prefix"]).sum() < 14
    assert "score_independent_0" in enriched[0]
    assert snapshot(paths) == before
    repeated, _ = capture_samples(samples, settings, output, prepared, repeats=1, max_lag=8)
    np.testing.assert_array_equal(repeated[0]["score_no_relay"], enriched[0]["score_no_relay"])
    wrong = [dict(samples[0], score=samples[0]["score"] + .1)]
    with pytest.raises(ValueError, match="replay mismatch"):
        capture_samples(wrong, settings, tmp_path / "wrong", prepared, repeats=1)
    assert not (tmp_path / "wrong" / "captures" / "5.npz").exists()


def test_all_cli_frozen_controls_and_fit_only_probes(tmp_path):
    root, prepared, checkpoint, _ = saved_experiment(tmp_path, training=True)
    paths = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = snapshot(paths)
    result = subprocess.run([sys.executable, "-m", "experiments.charm_structure_audit.deep_audit", "--root", str(root),
                             "--stage", "all", "--device", "cpu", "--bootstrap", "4", "--repeats", "1", "--probe-epochs", "2"],
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(paths) == before
    output = root / "deep_audit"
    probes = json.loads((output / "probes.json").read_text())
    assert probes["status"] == "completed" and probes["fit_answers"] == 3
    assert probes["threshold"] is None
    assert {"node", "retained_entropy", "projected", "layer_1", "layer_2", "head_overlap"} <= set(probes["probes"])
    with np.load(output / "linear_readouts.npz") as readouts:
        fitted = []
        for path in (output / "fit_features").glob("*.npz"):
            with np.load(path) as saved:
                fitted.append(saved["feature_node"])
        np.testing.assert_allclose(readouts["feature_node_mean"], np.concatenate(fitted).mean(axis=0), atol=1e-7)
    paired = json.loads((output / "paired_controls.json").read_text())
    assert set(paired) == {"independent_minus_coupled_0", "oracle_minus_matched_random_0"}
    assert (output / "summary.md").exists()
    for path in output.rglob("*.json"):
        json.loads(path.read_text(), parse_constant=lambda value: pytest.fail("Nonfinite JSON: " + value))


def test_probe_membership_is_required_and_source_overlap_rejected(tmp_path):
    from experiments.charm_structure_audit.deep_audit.probes import probe_records

    root, prepared, checkpoint, _ = saved_experiment(tmp_path)
    samples, _ = load_predictions(root)
    assert probe_records(samples, checkpoint, prepared)[0] is None
    parts = dict(fit=["0"], select=["1"], calibration=["2"], test=["5", "6"])
    (checkpoint.parent / "training.json").write_text(json.dumps(dict(partitions=parts)))
    records = json.loads((prepared / "index.json").read_text())
    records[0]["source_id"] = records[5]["source_id"]
    (prepared / "index.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Source overlap"):
        probe_records(samples, checkpoint, prepared)


def test_alternative_alignment_checks_offsets_not_only_labels():
    from experiments.charm_structure_audit.deep_audit.run import align_alternative

    row = sample()
    bad = copy.deepcopy(row)
    bad["offsets"][0, 0] += 1
    with pytest.raises(ValueError, match="offsets"):
        align_alternative([row], [bad], "alternative")


def test_edge_only_messages_and_head_marginal_features():
    graph = graph_fixture(count=22)
    model = model_fixture()
    base, states = trace_model(model, graph)
    altered, _ = trace_model(model, graph, block_sender=True)
    assert np.max(abs(base - altered)) > 1e-8
    coupled, _ = shuffle_endpoints(graph, 2)
    independent, _ = shuffle_endpoints(graph, 2, independent=True)
    for view in (coupled, independent):
        np.testing.assert_allclose(features(graph, states)["head_marginals"],
                                   features(view, states)["head_marginals"], atol=2e-6)


def test_previous_learned_audit_joins_only_identical_original_tokens(tmp_path):
    from experiments.charm_structure_audit.deep_audit.learned import attach_learned_controls
    from scipy.special import logit

    row = sample()
    directory = tmp_path / "learned_audit"
    (directory / "samples").mkdir(parents=True)
    arrays = {key: row[key] for key in ("score", "gold", "onset", "offsets", "response")}
    identity = np.asarray(json.dumps({key: row[key] for key in ("id", "source_id")}))
    path = directory / "samples" / "s.npz"
    np.savez_compressed(path, **arrays, identity=identity, control_head_s0=logit(row["score"]),
                        features=np.array([{"must_not_load": True}], dtype=object))
    joined, report = attach_learned_controls([row, sample("missing")], directory)
    assert report["status"] == "partial" and report["loaded_answers"] == 1
    np.testing.assert_allclose(joined[0]["score_learned_head_s0"], row["score"])
    assert "score_learned_head_s0" not in joined[1]
    wrong = dict(row, score=row["score"] + .01)
    with pytest.raises(ValueError, match="original token data differs"):
        attach_learned_controls([wrong], directory)
