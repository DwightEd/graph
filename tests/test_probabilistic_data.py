"""Scientific data boundaries and ranking semantics for conditional detection."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "teaching/state_audit/src"))

from experiments.probabilistic_detection.data import answer_features, select_records, source_weights
from experiments.probabilistic_detection.evaluation import pairwise_within, threshold_at_fpr, weighted_auc_order, weighted_auc


def test_cluster_bootstrap_auc_matches_sklearn_with_ties_and_zero_weights():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(326)
    labels = rng.integers(0, 2, 1000)
    scores = rng.integers(0, 9, 1000).astype(float)
    prepared = weighted_auc_order(labels, scores)
    for _ in range(5):
        weights = rng.integers(0, 5, 1000)
        assert weighted_auc(prepared, weights) == pytest.approx(roc_auc_score(labels, scores, sample_weight=weights))


def test_source_split_keeps_all_generators_together():
    records = [dict(task="QA", split="train", source_id=str(s), id=f"{s}:{g}")
               for s in range(20) for g in range(3)]
    records += [dict(task="QA", split="test", source_id="held-out", id="test")]
    selected = select_records(dict(records=records), "QA", "train", 5, 2)
    fit = {r["source_id"] for r in selected if r["partition"] == "fit"}
    dev = {r["source_id"] for r in selected if r["partition"] == "dev"}
    assert len(fit) == 5 and len(dev) == 2 and not fit & dev
    assert len(selected) == 21
    assert "held-out" not in fit | dev


def test_source_balance_is_equal_per_source_not_per_answer():
    groups = np.array([0, 0, 1, 1, 1, 1])
    weights = source_weights(groups)
    assert weights[groups == 0].sum() == pytest.approx(weights[groups == 1].sum())
    assert weights.sum() == pytest.approx(len(groups))


def fixture_trace():
    values = np.arange(6, dtype=float)
    observed = dict(token_id=np.arange(6), source_local=values, source_full=values * 2,
        raw_route=values, raw_attention=values / 6, entropy=values + 1)
    native = dict(token_id=np.arange(6), full=-values, local=-values - 1)
    response = dict(answer_ids=list(range(6)), units=[dict(start=0, stop=3), dict(start=3, stop=6)])
    return observed, native, response


def test_feature_values_and_unit_alignment():
    observed, native, response = fixture_trace()
    context, features, units = answer_features(observed, native, response, 100)
    np.testing.assert_allclose(context[:, 0], [1, 1, 1, 4, 4, 4])
    np.testing.assert_allclose(features[:, 0], [-1, 0, 1, -1, 0, 1])
    np.testing.assert_array_equal(units, [0, 0, 0, 1, 1, 1])
    assert context.shape == (6, 6) and features.shape == (6, 11)


def test_incomplete_units_and_token_misalignment_are_rejected():
    observed, native, response = fixture_trace()
    response["units"][1]["start"] = 4
    with pytest.raises(ValueError, match="partition"):
        answer_features(observed, native, response, 100)
    observed, native, response = fixture_trace()
    native["token_id"] = np.arange(6) + 1
    with pytest.raises(ValueError, match="token IDs"):
        answer_features(observed, native, response, 100)


def test_within_answer_pairs_exclude_between_answer_differences():
    labels = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0., 1., 101., 100., 999., 999.])
    groups = np.array([0, 0, 1, 1, 2, 2])
    assert pairwise_within(labels, scores, groups) == pytest.approx(.5)


def test_threshold_ties_do_not_manufacture_five_percent_false_alarms():
    labels = np.zeros(100, dtype=int)
    scores = np.r_[np.zeros(95), np.ones(5)]
    threshold = threshold_at_fpr(labels, scores)
    assert (scores > threshold).mean() <= .05
    assert threshold_at_fpr(np.ones(4, dtype=int), np.arange(4)) is None
