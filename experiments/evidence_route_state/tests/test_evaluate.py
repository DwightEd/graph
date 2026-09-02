import numpy as np

from experiments.evidence_route_state import evaluate, run


def scored_record():
    count = 3
    record = {
        "sample_id": "sample",
        "source_id": "source",
        "split_root": "/unused/test",
        "response_token_ids": np.array([10, 11, 12]),
        "prediction_position": np.array([3, 4, 5]),
        "valid": np.ones(count, dtype=bool),
        "contraction": np.array([0.2, 0.5, 0.8]),
        "takeover": np.array([0.1, 0.4, 0.9]),
    }
    for index, name in enumerate(evaluate.SCORE_NAMES):
        record[name] = np.linspace(0.1 + index, 0.3 + index, count)
    return record


def test_frozen_outputs_include_primary_and_both_locked_route_controls(tmp_path):
    assert evaluate.SCORE_NAMES[0] == "captured_posterior"
    assert {
        "independent_token_posterior",
        "one_hop_posterior",
        "endpoint_rewire_posterior",
        "weight_shuffle_posterior",
        "functional_route_collapse",
        "attention_route_collapse",
    }.issubset(evaluate.SCORE_NAMES)

    frozen = evaluate.freeze_scores([scored_record()], tmp_path / "scores.npz")

    for name in evaluate.SCORE_NAMES:
        np.testing.assert_array_equal(
            frozen[name],
            np.asarray(scored_record()[name], dtype=np.float32),
        )


def test_single_class_smoke_opens_labels_only_after_freeze_and_reports_na(
    tmp_path, monkeypatch, capsys
):
    record = scored_record()
    frozen_path = tmp_path / "frozen_scores.npz"
    evaluate.freeze_scores([record], frozen_path)
    label_events = []

    def load_labels(_records, frozen):
        assert frozen_path.is_file()
        assert "captured_posterior" in frozen
        label_events.append("labels opened")
        return np.zeros(3, dtype=bool)

    monkeypatch.setattr(evaluate, "load_labels", load_labels)
    report = evaluate.evaluate_scores(
        [record],
        frozen_path,
        tmp_path / "report.json",
        task_type="QA",
        bootstrap=10,
    )
    run.print_report(report)
    printed = capsys.readouterr().out

    assert label_events == ["labels opened"]
    assert report["primary_score"] == "captured_posterior"
    assert all(result["auroc"] is None for result in report["detection"].values())
    assert printed.count("AUROC=n/a AP=n/a") == len(evaluate.SCORE_NAMES)


def test_primary_and_topology_control_are_compared_on_the_same_tokens(
    tmp_path, monkeypatch
):
    record = scored_record()
    record["endpoint_rewire_posterior"] = np.array([0.1, np.nan, 0.3])
    frozen = tmp_path / "frozen.npz"
    evaluate.freeze_scores([record], frozen)
    monkeypatch.setattr(
        evaluate,
        "load_labels",
        lambda _records, _frozen: np.array([False, True, True]),
    )

    report = evaluate.evaluate_scores(
        [record],
        frozen,
        tmp_path / "report.json",
        task_type="QA",
        bootstrap=0,
    )

    comparison = report["paired_primary_minus_control"]["endpoint_rewire_posterior"]
    assert comparison["evaluated_tokens"] == 2
    focus = report["legitimate_narrow_focus"]
    assert focus["threshold_fixed_before_labels"] == 0.9
    assert focus["tokens"] == 1
