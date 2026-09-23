"""Audit invariants: exact AP/ties, graph alignment, cluster weights and read-only CLI."""

from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from state_audit.storage import read_json, write_arrays, write_json

from experiments.native_support.comparison_evaluation import evaluate_comparison
from experiments.native_support.dynamics import score_rows
from experiments.native_support.dynamics_audit_rank import rank_ledger
from experiments.native_support.transport import METHODS, main, offline_mean
from experiments.native_support.transport_audit_bootstrap import (
    paired_bootstrap,
    weighted_ranks,
)
from experiments.native_support.transport_audit_rank import (
    ranking_tables,
    selection_weight,
)
from experiments.native_support.transport_audit_state import (
    edge_rows,
    state_measurements,
)
from experiments.native_support.transport_state import budget_risk, infer_budget


def token_fixture():
    rows = []
    for index, label in enumerate([0, 1, 0, 1, 1, 0, 0, 1]):
        rows.append({"response_id": "a" if index < 4 else "b", "source_id": "source-a" if index < 4 else "source-b",
            "target": index % 4, "label": label, "is_span_onset": bool(label),
            "is_answer_first_error": index in (1, 4), "phase": "error" if label else "normal",
            "half": "front" if index % 4 < 2 else "back",
            "transport_route": [1., .8, .8, .4, .3, .3, .1, .1][index],
            "raw_route": [.8, .8, .4, .5, .3, .3, 0., .4][index],
            "route_offline_mean": [.5, .5, .5, .5, .1, .1, .1, .1][index]})
    return rows


def test_ap_credits_and_budget_changes_respect_complete_ties():
    tokens = token_fixture()
    methods = ("transport_route", "raw_route", "route_offline_mean")
    tables = ranking_tables(tokens, methods)
    metrics = {(row["scope"], row["method"]): row for row in tables["metrics"] if row["response_id"] == "ALL"}
    for scope in ("all_error", "front_half", "back_half"):
        for control in methods[1:]:
            parts = [row["delta_ap_credit"] for row in tables["ap_attribution"]
                     if row["scope"] == scope and row["control"] == control and row["partition"] == "response_id"]
            expected = metrics[scope, methods[0]]["ap"] - metrics[scope, control]["ap"]
            assert sum(parts) == pytest.approx(expected)
    labels = np.asarray([row["label"] for row in tokens])
    scores = np.asarray([row["transport_route"] for row in tokens])
    ledger = rank_ledger(labels, scores)
    assert ledger["ap_credit"].sum() == pytest.approx(average_precision_score(labels, scores))
    np.testing.assert_equal(ledger["rank_first"][[1, 2]], [2, 2])
    weights = selection_weight(scores, 2)
    np.testing.assert_equal(weights[:3], [1., .5, .5])
    assert weights.sum() == 2
    changes = [row for row in tables["budget_changes"] if row["scope"] == "all_error" and row["budget"] == 1
               and row["control"] == "raw_route"]
    assert sum(row["selection_change"] for row in changes) == pytest.approx(0.)


def test_source_multiplicity_is_equivalent_to_resampling_whole_clusters():
    labels = np.asarray([0, 1, 0, 1, 1, 0])
    scores = np.asarray([.2, .2, .7, .8, .8, .9])
    # The first source occurs twice, the second is omitted, the third once.
    weights = np.asarray([2, 2, 0, 0, 1, 1])
    bins = np.unique(scores, return_inverse=True)[1]
    actual = weighted_ranks(labels, bins, weights)
    repeated = np.repeat(np.arange(len(labels)), weights)
    expected = [roc_auc_score(labels[repeated], scores[repeated]),
                average_precision_score(labels[repeated], scores[repeated])]
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.isnan(weighted_ranks(labels, bins, (labels == 0).astype(int))).all()
    all_positive = weighted_ranks(labels, bins, labels)
    assert np.isnan(all_positive[0]) and all_positive[1] == 1


def test_bootstrap_is_paired_reproducible_and_handles_one_source():
    tokens = token_fixture()
    for row in tokens:
        row["raw_route"] = row["transport_route"]
    rows, draws = paired_bootstrap(tokens, 12, 7)
    _, repeated = paired_bootstrap(tokens, 12, 7)
    np.testing.assert_equal(draws["deltas"], repeated["deltas"])
    finite = draws["deltas"][:, :, 0][np.isfinite(draws["deltas"][:, :, 0])]
    np.testing.assert_equal(finite, 0.)
    assert all(row["source_clusters"] == 2 for row in rows)
    for row in tokens:
        row["source_id"] = "shared-source"
    rows, _ = paired_bootstrap(tokens, 4, 7)
    assert all(row["ci_low"] is None and row["ci_high"] is None for row in rows)


def cached_state():
    budget = np.asarray([[[[1., 1., 0.]]], [[[2., 1., 1.]]], [[[1., 3., 1.]]], [[[4., 1., 0.]]]])
    edges = np.zeros((4, 4))
    edges[2, 1], edges[3, 2] = .2, .4
    return {"observed_budget": budget, "edge_weight": edges, "reuse_weight": edges * 2,
            "degree": (edges + edges.T).sum(axis=1), **infer_budget(budget, edges)}


def test_saved_role_equation_and_key_labels_use_distinct_alignment():
    state = cached_state()
    labels, valid = np.asarray([0, 1, 1, 0]), np.asarray([True, True, True, False])
    measured = state_measurements(state, 1, 1., labels, valid)
    assert measured["relative_role_residual"].max() < 1e-14
    np.testing.assert_equal(measured["relative_budget_change"][0], 0.)
    assert np.isnan(measured["error_neighbor_fraction"][0])
    assert measured["unknown_neighbor_mass"][2] == pytest.approx(.4)
    rows = edge_rows(state, labels, valid)
    query = next(row for row in rows if row["donor_label_alignment"] == "query_target"
                 and row["receiver_label"] == 1 and row["donor_label"] == 1)
    key = next(row for row in rows if row["donor_label_alignment"] == "answer_key"
               and row["receiver_label"] == 1 and row["donor_label"] == 0)
    assert query["conditioned_mass"] == pytest.approx(.2)
    assert key["conditioned_mass"] == pytest.approx(.2)
    for alignment in ("query_target", "answer_key"):
        assert sum(row["conditioned_mass"] for row in rows if row["donor_label_alignment"] == alignment) == pytest.approx(.6)


def write_saved_audit_input(output):
    directory = output / "source_transport"
    responses, annotations, token_rows = [], {}, []
    for index, identity in enumerate(("a", "b")):
        item = {"id": identity, "source_id": identity, "prompt_length": 2,
                "token_ids": [7, 8, 1, 2, 3, 4], "token_text": ["p", "q", "a", "b", "c", "d"]}
        responses.append(item)
        annotations[identity] = {"token_ids": [1, 2, 3, 4], "labels": [0, 1, 1, 0], "valid_tokens": [1, 1, 1, 0]}
        state = cached_state()
        raw = budget_risk(state["observed_budget"], 1)
        transport = budget_risk(state["inferred_budget"], 1)
        scores = {"target": np.arange(4), "query": np.arange(4) + 1, "token_id": np.arange(1, 5),
                  "raw_route": raw, "transport_route": transport, "risk": transport,
                  "route_offline_mean": offline_mean(raw), "entropy": np.arange(4) / 10, "raw_attention": raw}
        response_dir = directory / "responses" / f"{index:04d}"
        write_arrays(response_dir / "scores.npz", **scores)
        write_arrays(response_dir / "state.npz", **state)
        write_json(response_dir / "sources.json", {"blocks": []})
        token_rows.extend(score_rows(item, scores, METHODS))
    write_json(output / "settings.json", {"responses": responses})
    write_json(output / "annotations.json", annotations)
    methods = {name: name for name in METHODS}
    write_json(directory / "scoring_protocol.json", {"methods": methods, "strength": 1.})
    evaluate_comparison(output, directory, output / "annotations.json", methods, token_rows)


def test_audit_cli_uses_only_saved_outputs_and_never_changes_scores(tmp_path):
    write_saved_audit_input(tmp_path)
    source = tmp_path / "source_transport"
    frozen = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with patch("experiments.native_support.transport.build_graph", side_effect=AssertionError("rebuilt graph")), \
         patch("experiments.native_support.transport.infer_budget", side_effect=AssertionError("solved state")), \
         patch("state_audit.model.load_model", side_effect=AssertionError("loaded LLM")), \
         patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("loaded tokenizer")):
        main(["--stage", "audit", "--output", str(tmp_path), "--bootstrap-replicates", "10"])
    summary = read_json(source / "audit/summary.json")
    assert summary["tokens"] == 6 and summary["positives"] == 4
    assert summary["state_checks_passed"] and summary["saved_evaluation_matches"]
    assert all(path.read_bytes() == content for path, content in frozen.items())
    with ZipFile(source / "audit_data.zip") as archive:
        assert "audit/tokens.csv" in archive.namelist()
        assert "audit/bootstrap.npz" in archive.namelist()
        assert "audit/responses/0000.npz" in archive.namelist()
    arrays = np.load(source / "audit/responses/0000.npz", allow_pickle=False)
    assert arrays["valid"].tolist() == [True, True, True, False]
    assert arrays["query"].tolist() == [1, 2, 3, 4]


def test_audit_refuses_misaligned_target_ids(tmp_path):
    write_saved_audit_input(tmp_path)
    path = tmp_path / "annotations.json"
    annotations = read_json(path)
    annotations["a"]["token_ids"][0] = 99
    write_json(path, annotations)
    with pytest.raises(ValueError, match="annotation token IDs"):
        main(["--stage", "audit", "--output", str(tmp_path), "--bootstrap-replicates", "0"])
