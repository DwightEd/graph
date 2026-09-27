"""Synthetic regression rendering and freeze/partition integrity; no natural labels."""

import json

import numpy as np
import pytest

from experiments.probabilistic_detection import cases


def example():
    record = dict(id="9022", source_id="s", task="QA", generator="synthetic",
                  split="test", partition="test", packed_start=0, packed_stop=5)
    text = "a <b> c"
    response = dict(answer_ids=[10, 11, 12, 13, 14], text=text,
                    offsets=[[0, 1], [2, 3], [3, 4], [4, 5], [6, 7]])
    annotation = dict(response_text=text, token_ids=response["answer_ids"],
        labels=[0, 1, 1, 1, 0], valid_tokens=[True] * 5,
        character_spans=[dict(start=2, end=5)], annotation_origin="synthetic")
    return record, response, annotation


def test_span_status_uses_strict_frozen_threshold_and_nearby_normal_tokens():
    record, response, annotation = example()
    scores = dict(entire=np.array([0., 2., 3., 4., 2.]),
                  partial=np.array([0., 1., 2., 0., 0.]),
                  missed=np.array([2., 0., 1., 0., 0.]))
    result = cases.build_case(record, response, annotation, np.arange(5),
                             response["answer_ids"], scores, dict.fromkeys(scores, 1.), 1)
    measured = result["spans"][0]["methods"]
    assert measured["entire"]["status"] == "entire_span"
    assert measured["entire"]["neighboring_normal_fpr"] == .5
    assert measured["partial"]["status"] == "partial_span"
    assert measured["partial"]["detected_tokens"] == 1
    assert measured["missed"]["status"] == "missed"
    html = cases.render_html(dict(cases=[result]))
    assert "&lt;b&gt;" in html and "<b>" not in html
    assert "探索性复验" in html


def test_development_scores_align_after_excluding_interleaved_fit_rows(tmp_path):
    pack = dict(development=np.array([False, False, True, True, False, True]), target=np.arange(6))
    record = dict(partition="dev", packed_start=2, packed_stop=4)
    actual = cases.response_scores(pack, record, tmp_path, ["score"],
                                   dict(score=np.array([.1, .2, .3])), False)
    np.testing.assert_array_equal(actual["score"], [.1, .2])
    record.update(partition="fit", packed_start=0, packed_stop=2)
    assert cases.response_scores(pack, record, tmp_path, [], {}, False) is None


def test_incomplete_freeze_stops_before_any_annotation_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Labels must remain unopened")

    monkeypatch.setattr(cases, "annotations", forbidden)
    with pytest.raises(ValueError, match="freeze models"):
        cases.main(["--run", str(tmp_path), "--tasks", "QA"])


def test_token_identity_mismatch_cannot_silently_render_scores():
    record, response, annotation = example()
    with pytest.raises(ValueError, match="token IDs"):
        cases.build_case(record, response, annotation, np.arange(5), [0] * 5,
                         dict(score=np.zeros(5)), dict(score=1), 2)


def test_fit_reports_are_explicit_and_missing_threshold_is_not_a_negative():
    record, response, annotation = example()
    record.update(partition="fit", split="train")
    result = cases.build_case(record, response, annotation, np.arange(5), response["answer_ids"],
                             dict(score=np.zeros(5)), dict(score=None), 0)
    assert result["in_sample"]
    assert "in-sample" in result["partition_note"]
    assert result["spans"][0]["methods"]["score"]["status"] == "threshold_unavailable"


def test_cli_reports_absent_ids_without_creating_scores(tmp_path, monkeypatch):
    run = tmp_path / "run"
    packs = run / "packs"
    packs.mkdir(parents=True)
    for split in ["train", "test"]:
        (packs / f"QA_{split}.json").write_text(json.dumps(dict(records=[])))
    monkeypatch.setattr(cases, "require_frozen", lambda *args: None)
    cases.main(["--run", str(run), "--ids", "missing", "--tasks", "QA"])
    report = json.loads((run / "cases/report.json").read_text())
    assert report["cases"][0]["status"] == "not_in_packs"
    assert "spans" not in report["cases"][0]
    assert (run / "cases/index.html").is_file()


def test_normal_answer_preserves_false_alarm_rows():
    record, response, annotation = example()
    annotation.update(labels=[0] * 5, character_spans=[])
    result = cases.build_case(record, response, annotation, np.arange(5), response["answer_ids"],
                             dict(score=np.array([0., 0., 0., 2., 0.])), dict(score=1.), 2)
    assert result["spans"] == []
    assert len(result["normal_tokens"]) == 5
    assert sum(row["methods"]["score"]["alarm"] for row in result["normal_tokens"]) == 1


def test_synthetic_pack_and_frozen_readout_generate_aligned_report(tmp_path, monkeypatch):
    run = tmp_path / "run"
    directory = run / "QA"
    packs = run / "packs"
    cache = tmp_path / "cache"
    for path in [directory, packs, cache / "responses/0"]:
        path.mkdir(parents=True)
    record, response, annotation = example()
    record["directory"] = "responses/0"
    record["tokens"] = 5
    (cache / "responses/0/response.json").write_text(json.dumps(response))
    (cache / "manifest.json").write_text("{}")
    for split, records in [("train", []), ("test", [record])]:
        (packs / f"QA_{split}.json").write_text(json.dumps(dict(records=records, source_cache=str(cache))))
        np.savez(packs / f"QA_{split}.npz", target=np.arange(5), token_id=response["answer_ids"],
                 development=np.zeros(5, bool))
    (directory / "models.joblib").write_bytes(b"synthetic unused model marker")
    (directory / "selection.json").write_text(json.dumps(dict(thresholds=dict(source_first=1.))))
    (directory / "test_frozen.json").write_text(json.dumps(dict(status="all_predictions_frozen")))
    np.savez(directory / "development_scores.npz", source_first=np.zeros(0))
    np.savez(directory / "test_scores.npz", source_first=np.array([0., 1., 2., 3., 0.]))
    calls = []

    def synthetic_annotations(root, manifest, records):
        calls.append([row["id"] for row in records])
        return {"9022": annotation}

    monkeypatch.setattr(cases, "annotations", synthetic_annotations)
    cases.main(["--run", str(run), "--ids", "9022", "--methods", "source_first"])
    report = json.loads((run / "cases/report.json").read_text())
    measured = report["cases"][0]["spans"][0]["methods"]["source_first"]
    assert calls == [["9022"]]
    assert measured["status"] == "partial_span"
    assert measured["detected_tokens"] == 2
    assert measured["neighboring_normal_fpr"] == 0


@pytest.mark.parametrize("selected", ["trees", "selected_readout"])
def test_deployment_alias_uses_frozen_dev_choice_and_threshold(tmp_path, selected):
    values = np.array([-.5, 1.2, 2.1])
    (tmp_path / "selection.json").write_text(json.dumps(dict(thresholds=dict(trees=10.))))
    (tmp_path / "detector_selection.json").write_text(json.dumps(dict(selected=selected, threshold=1.2)))
    np.savez(tmp_path / "development_scores.npz", trees=values)
    np.savez(tmp_path / "readout_development_scores.npz", selected_readout=values + 2)
    result = cases.partition_predictions(tmp_path, "train", ["selected_detector"])
    expected = values if selected == "trees" else values + 2
    assert list(result) == ["selected_detector"]
    np.testing.assert_array_equal(result["selected_detector"], expected)
    assert cases.task_thresholds(tmp_path)["selected_detector"] == 1.2


def test_test_deployment_alias_uses_frozen_predictions_without_reselection(tmp_path):
    (tmp_path / "detector_selection.json").write_text(json.dumps(dict(selected="trees", threshold=1.)))
    np.savez(tmp_path / "test_scores.npz", selected_detector=np.array([2., 3.]), trees=np.array([9., 9.]))
    result = cases.partition_predictions(tmp_path, "test", ["selected_detector"])
    np.testing.assert_array_equal(result["selected_detector"], [2., 3.])
