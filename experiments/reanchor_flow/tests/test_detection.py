import numpy as np

from experiments.reanchor_flow.detection import _MetricOrder, _metric, _onset_labels


def test_onset_labels_mark_each_positive_run_once_per_sample():
    label = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int8)
    sample = np.array(["a"] * 5 + ["b"] * 3)
    token = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    onset = _onset_labels(label, sample, token)
    assert np.array_equal(onset, [False, True, False, False, True, False, True, False])


def test_metric_reports_perfect_ranking_and_prevalence_lift():
    result = _metric(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["auprc_lift"] == 2.0


def test_weighted_bootstrap_metric_matches_duplicated_source_rows():
    label = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.8, 0.2, 0.8, 0.7, 0.4])
    source = np.array([0, 0, 1, 1, 2, 2])
    weight = np.array([2, 0, 1])
    weighted = _MetricOrder.build(label, score, source).metric(weight)
    selected = np.repeat(np.arange(len(label)), weight[source])
    duplicated = _metric(label[selected], score[selected])
    assert np.allclose(weighted, [duplicated["auroc"], duplicated["auprc"]])
