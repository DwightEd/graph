from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

from experiments.evidence_route_state import evaluate


def scored_record() -> dict[str, object]:
    count = 3
    record: dict[str, object] = {
        "sample_id": "sample",
        "source_id": "source",
        "split_root": "/unused/test",
        "response_token_ids": np.array([10, 11, 12]),
        "query_position": np.array([2, 3, 4]),
        "prediction_position": np.array([3, 4, 5]),
        "valid": np.ones(count, dtype=bool),
    }
    for index, name in enumerate(evaluate.SCORE_NAMES):
        record[name] = np.linspace(0.1 + index, 0.3 + index, count)
    return record


class LabelPoisonRecord(Mapping[str, object]):
    """A score record that fails if score freezing requests a label."""

    def __init__(self, values: Mapping[str, object]):
        self.values = dict(values)

    def __getitem__(self, key: str) -> object:
        if "label" in key:
            raise AssertionError("freeze_scores opened labels")
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


def test_score_freeze_is_label_free_and_contains_every_fixed_score(tmp_path):
    output = tmp_path / "scores.npz"
    record = LabelPoisonRecord(scored_record())

    frozen = evaluate.freeze_scores([record], output)

    assert output.is_file()
    assert evaluate.SCORE_NAMES == (
        "conditional_graph_energy",
        "independent_graph_energy",
        "functional_route_collapse",
        "attention_route_collapse",
        "confidence",
    )
    for name in evaluate.SCORE_NAMES:
        np.testing.assert_array_equal(frozen[name], scored_record()[name])


def test_evaluation_opens_labels_only_after_frozen_scores_exist(tmp_path, monkeypatch):
    record = scored_record()
    frozen_path = tmp_path / "frozen_scores.npz"
    expected = evaluate.freeze_scores([record], frozen_path)
    label_events = []

    def load_labels(_records, frozen):
        assert frozen_path.is_file()
        for name in evaluate.SCORE_NAMES:
            np.testing.assert_array_equal(frozen[name], expected[name])
        label_events.append("opened")
        return np.zeros(3, dtype=bool)

    monkeypatch.setattr(evaluate, "load_labels", load_labels)
    report = evaluate.evaluate_scores(
        [record],
        frozen_path,
        tmp_path / "report.json",
        task_type="QA",
        bootstrap=0,
    )

    assert label_events == ["opened"]
    assert report["primary_score"] == "conditional_graph_energy"
    assert report["labels_used_during"] == "posthoc evaluation after score freeze"
    assert all(result["auroc"] is None for result in report["detection"].values())


def test_primary_and_each_control_use_the_same_paired_token_subset(
    tmp_path, monkeypatch
):
    record = scored_record()
    record["independent_graph_energy"] = np.array([0.1, np.nan, 0.3])
    frozen_path = tmp_path / "frozen_scores.npz"
    evaluate.freeze_scores([record], frozen_path)
    monkeypatch.setattr(
        evaluate,
        "load_labels",
        lambda _records, _frozen: np.array([False, True, True]),
    )

    report = evaluate.evaluate_scores(
        [record],
        frozen_path,
        tmp_path / "report.json",
        task_type="QA",
        bootstrap=0,
    )

    comparison = report["paired_primary_minus_control"]["independent_graph_energy"]
    assert comparison["evaluated_tokens"] == 2
    assert comparison["auroc_difference"] == 0.0
