"""Synthetic CPU contracts for frozen grounded-graph ranking evaluation."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from next_iteration.grounded_graph_evaluate import evaluate_members, ranking
from next_iteration.grounded_graph_predict import PROTOCOL as PREDICTION_PROTOCOL


def test_source_balanced_ranking_uses_inverse_per_source_word_counts_without_direction_flip():
    labels = np.array([0, 1, 0, 1, 0])
    # Source A has three words; source B has two.  The score is used as supplied.
    scores = np.array([.1, .8, .3, .7, .2])
    sources = np.array(["A", "A", "A", "B", "B"])
    result = ranking(labels, scores, sources)
    weights = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2])

    assert result["auroc"] == roc_auc_score(labels, scores, sample_weight=weights)
    assert result["auprc"] == average_precision_score(labels, scores, sample_weight=weights)
    assert result["error_prevalence"] == np.average(labels, weights=weights)
    # Reversing after labels would be a different, prohibited metric.
    assert result["auroc"] != roc_auc_score(labels, -scores, sample_weight=weights)


def test_all_first_inclusive_and_strict_postfirst_keep_unavailable_words_in_denominator():
    member = {
        "y": np.array([False, True, True, False, False]),
        # Includes the first error token; strict post-first begins at index two.
        "first": np.array([True, True, False, False, False]),
        "source": np.array(["s"] * 5),
        "available": np.array([True, False, True, False, True]),
        "scores": {name: np.linspace(0.1, 0.5, 5) for name in PREDICTION_PROTOCOL["scores"]},
    }

    result = evaluate_members([member])

    assert result["all_words"]["words"] == 5
    assert result["all_words"]["error_words"] == 2
    assert result["all_words"]["unavailable_words"] == 2
    assert result["through_first_error"]["words"] == 2
    assert result["through_first_error"]["error_words"] == 1
    assert result["through_first_error"]["unavailable_words"] == 1
    assert set(result["through_first_error"]["metrics"]) == set(PREDICTION_PROTOCOL["scores"])
    assert result["post_first_error"]["words"] == 3
    assert result["post_first_error"]["error_words"] == 1
    assert result["post_first_error"]["unavailable_words"] == 1
