from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.reanchor_flow.detection_metrics import RankedScores, detection_report


def test_ties_and_weighted_metrics_match_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(10)
    labels = rng.integers(0, 2, size=90)
    scores = rng.integers(0, 5, size=90).astype(float)
    sources = np.arange(90) % 9
    ranked = RankedScores(labels, scores, sources)
    for _ in range(8):
        weights = rng.integers(0, 4, size=9)
        token_weights = weights[sources]
        auc, ap = ranked.evaluate(weights)
        assert auc == pytest.approx(sklearn.roc_auc_score(labels, scores, sample_weight=token_weights))
        assert ap == pytest.approx(sklearn.average_precision_score(labels, scores, sample_weight=token_weights))


def test_fixed_direction_unknown_labels_and_paired_identical_scores():
    labels = np.array([0, 1, 0, 1, -1])
    scores = np.array([0.1, 0.8, 0.2, 0.9, 9.0])
    report = detection_report(labels, {"routing_joint": scores, "same": scores, "reverse": -scores},
                              np.array(["a", "a", "b", "b", "c"]), np.array(["QA"] * 5),
                              bootstrap=30)
    group = report["groups"]["ALL"]
    assert group["tokens"] == 5
    assert group["known_tokens"] == 4
    assert group["unknown_tokens"] == 1
    assert group["sources"] == 2
    assert group["prevalence"] == 0.5
    assert group["scores"]["routing_joint"]["auroc"] == 1
    assert group["scores"]["reverse"]["auroc"] == 0
    assert group["paired_differences"]["same"]["auroc_ci95"] == [0, 0]
    assert group["paired_differences"]["same"]["auprc_ci95"] == [0, 0]
    json.dumps(report, allow_nan=False)


def test_source_cluster_bootstrap_is_invariant_to_replicating_all_tokens():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.4, 0.5, 0.6])
    sources = np.array(["a", "a", "b", "b", "c", "c"])
    tasks = np.array(["QA"] * 6)
    report = detection_report(labels, {"routing_joint": scores}, sources, tasks, bootstrap=80)
    duplicated = detection_report(np.repeat(labels, 3), {"routing_joint": np.repeat(scores, 3)},
                                  np.repeat(sources, 3), np.repeat(tasks, 3), bootstrap=80)
    first = report["groups"]["ALL"]["scores"]["routing_joint"]
    second = duplicated["groups"]["ALL"]["scores"]["routing_joint"]
    for field in ("auroc", "auprc", "auroc_ci95", "auprc_ci95"):
        np.testing.assert_allclose(first[field], second[field], atol=1e-14)


def test_cluster_weights_match_explicit_source_resampling():
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.6, 0.6, 0.3, 0.9, 0.2])
    sources = np.array([0, 0, 1, 1, 2, 2])
    weight = np.array([2, 0, 1])
    weighted = RankedScores(labels, scores, sources).evaluate(weight)
    copied = np.repeat(np.arange(len(labels)), weight[sources])
    explicit = RankedScores(labels[copied], scores[copied], np.zeros(len(copied), dtype=int))
    np.testing.assert_allclose(weighted, explicit.evaluate(np.ones(1)))


def test_one_class_and_empty_groups_are_json_safe():
    report = detection_report(np.array([0, 0, -1]), {"routing_joint": np.zeros(3)},
                              np.array(["a", "a", "b"]), np.array(["QA", "QA", "Summary"]),
                              bootstrap=8)
    assert report["groups"]["QA"]["scores"]["routing_joint"]["auroc"] is None
    assert report["groups"]["QA"]["scores"]["routing_joint"]["auprc"] is None
    assert report["groups"]["Summary"]["known_tokens"] == 0
    assert report["groups"]["Summary"]["prevalence"] is None
    json.dumps(report, allow_nan=False)


def test_invalid_score_cannot_silently_reduce_coverage():
    with pytest.raises(ValueError, match="finite"):
        detection_report(np.array([0, 1]), {"routing_joint": np.array([0, np.nan])},
                         np.array(["a", "b"]), np.array(["QA", "QA"]))
