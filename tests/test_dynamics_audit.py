"""Ranking identities and cache-only audit contracts, not natural-data results."""

import csv
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from state_audit.storage import read_arrays, read_json, write_arrays

from experiments.native_support.dynamics import main
from experiments.native_support.dynamics_audit_rank import (
    budget_rows,
    half_pair_rows,
    rank_ledger,
)
from experiments.native_support.dynamics_core import mode_posteriors


def test_ap_credits_match_sklearn_with_mixed_ties_and_token_permutations():
    labels = np.array([0, 1, 1, 0, 0, 1, 0])
    scores = np.array([1., 1., .8, .8, .2, .2, .1])
    ledger = rank_ledger(labels, scores)
    assert ledger["ap_credit"].sum() == pytest.approx(average_precision_score(labels, scores), abs=1e-15)
    np.testing.assert_array_equal(ledger["rank_first"], [1, 1, 3, 3, 5, 5, 7])
    np.testing.assert_array_equal(ledger["normals_above"], [0, 0, 1, 1, 2, 2, 3])
    permutation = np.array([4, 3, 2, 6, 1, 5, 0])
    permuted = rank_ledger(labels[permutation], scores[permutation])
    np.testing.assert_array_equal(permuted["ap_credit"], ledger["ap_credit"][permutation])


def test_budget_ties_do_not_use_arbitrary_token_order():
    labels = np.array([1, 0, 1, 0, 0])
    rows = budget_rows(labels, np.ones(5), "tied")
    one = next(row for row in rows if row["budget"] == 1)
    assert one["precision_expected"] == pytest.approx(.4)
    assert one["precision_min"] == 0 and one["precision_max"] == 1
    all_tokens = next(row for row in rows if row["budget"] == 5)
    assert all_tokens["expected_errors"] == 2
    assert all_tokens["precision_min"] == all_tokens["precision_max"] == .4


def test_log_posterior_recovers_order_lost_to_probability_rounding():
    emission = np.array([[0., 60.], [0., 80.], [0., 45.]])
    posterior = mode_posteriors(emission, np.full((2, 2), .5), np.full(2, .5))
    probability = posterior["smoothed"][:, 1]
    odds = posterior["log_smoothed"][:, 1] - posterior["log_smoothed"][:, 0]
    np.testing.assert_array_equal(probability, np.ones(3))
    np.testing.assert_allclose(odds, emission[:, 1], atol=1e-12)
    assert average_precision_score([0, 1, 0], probability) == pytest.approx(1 / 3)
    assert average_precision_score([0, 1, 0], odds) == 1


def test_half_pair_auc_credits_recover_global_auc():
    tokens = [{"label": label, "half": half} for label, half in
              ((1, "front"), (0, "front"), (1, "back"), (0, "back"), (0, "back"))]
    scores = np.array([.9, .4, .4, .2, .8])
    rows = half_pair_rows(tokens, scores, "example")
    assert sum(row["global_auc_credit"] for row in rows) == pytest.approx(
        roc_auc_score([row["label"] for row in tokens], scores))


def verify_cached_audit(target):
    """Called by the existing tiny-Llama integration after its cache is built."""
    score_path = target / "state_dynamics/responses/0000/scores.npz"
    original_bytes = score_path.read_bytes()
    with patch("state_audit.model.load_model", side_effect=AssertionError("audit loaded LLM")), \
         patch("experiments.native_support.dynamics.fit_model", side_effect=AssertionError("audit fitted model")), \
         patch("experiments.native_support.dynamics.build_sequences", side_effect=AssertionError("audit read raw token traces")):
        main(["--stage", "audit", "--output", str(target), "--cpu-threads", "1"])
    root = target / "state_dynamics/audit"
    summary = read_json(root / "summary.json")
    assert summary["status"] == "verified" and summary["evaluated_tokens"] == 20
    assert score_path.read_bytes() == original_bytes
    schema = read_json(root / "representation_schema.json")
    assert schema["vectors"]["observation"]["width"] == 7
    vectors = read_arrays(root / "representations/0000.npz")
    assert vectors["z"].shape == (10, 3) and vectors["h"].shape == (10, 6)
    with (root / "ap_parts.csv").open() as stream:
        parts = list(csv.DictReader(stream))
    for measured in summary["all_error"]:
        credits = [float(row["global_ap_credit"]) for row in parts
                   if row["method"] == measured["method"] and row["partition"] == "answer"]
        assert sum(credits) == pytest.approx(measured["ap"], abs=1e-12)
    with ZipFile(target / "state_dynamics/audit_data.zip") as archive:
        assert "audit/representation_schema.json" in archive.namelist()
    damaged = read_arrays(score_path)
    damaged["query"][0] += 1
    write_arrays(score_path, **damaged)
    with pytest.raises(ValueError, match="alignment/numeric audit failed"):
        main(["--stage", "audit", "--output", str(target), "--cpu-threads", "1"])
    score_path.write_bytes(original_bytes)
