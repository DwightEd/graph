"""CPU checks for evaluation denominators and abstention accounting."""

import numpy as np

from experiments.evaluate_native_audit import metrics


def test_metrics_keeps_abstentions_in_the_full_word_error_denominator():
    result = metrics(
        [
            {
                "source_id": "source-a",
                "labels": np.array([True, False]),
                "scores": {"semantic_A": np.array([0.95, 0.99])},
                "abstain": {"semantic_A": np.array([False, True])},
            },
            {
                "source_id": "source-b",
                "labels": np.array([False]),
                "scores": {"semantic_A": np.array([0.1])},
                "abstain": {"semantic_A": np.array([False])},
            },
        ],
        "semantic_A",
    )
    assert result["all_words"] == 3
    assert result["scored_words"] == 2
    assert result["abstained_words"] == 1
    assert result["annotated_error_words"] == 1
    assert result["recall_including_abstentions"] == 1.0
    assert result["coverage"] == 2 / 3
