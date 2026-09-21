import csv
import hashlib
import json
import tarfile

import numpy as np
import pytest

from experiments.unsupervised_token_graph.evaluate import label_views
from experiments.unsupervised_token_graph.head_geometry.continuity import (
    main as audit_main,
)
from experiments.unsupervised_token_graph.head_geometry.continuity import (
    normalize_annotations,
    rank_view,
)
from experiments.unsupervised_token_graph.head_geometry.run import main
from experiments.unsupervised_token_graph.head_geometry.tests.test_pipeline import (
    fixture_arguments,
)


def prediction_digests(output):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (output / "predictions").glob("*") if path.is_file()}


def test_three_task_audit_reuses_scores_matches_controls_and_archives(tmp_path, monkeypatch):
    output = tmp_path / "cross"
    tasks = ("QA", "Data2txt", "Summary")
    argv = fixture_arguments(tmp_path, output, tasks)
    annotation = tmp_path / "dataset/response.jsonl"
    rows = [json.loads(line) for line in annotation.read_text().splitlines()]
    for row in rows:
        row["labels"] = [{"start": 9, "end": 12}]
    annotation.write_text("\n".join(map(json.dumps, rows)))
    main([*argv, "--tasks", *tasks, "--suite", "cross_terms", "--no-layer-bands"])
    before = prediction_digests(output)
    for name in ("train", "test"):
        (tmp_path / name).rename(tmp_path / (name + "_unavailable"))

    def forbidden(*args, **kwargs):
        pytest.fail("Frozen audit must not refit, score or load a tokenizer with verified saved offsets")

    monkeypatch.setattr("experiments.unsupervised_token_graph.head_geometry.pipeline.fit", forbidden)
    monkeypatch.setattr("experiments.unsupervised_token_graph.head_geometry.pipeline.score", forbidden)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", forbidden)
    report = audit_main(["--output", str(output), "--bootstrap", "4", "--permutations", "3"])
    assert set(report["groups"]) == {task + "|fixture" for task in tasks}
    for result in report["groups"].values():
        assert result["annotation"]["continuation_fraction"] == pytest.approx(2 / 3)
        comparison = result["comparisons"]["pair_state_smooth_minus_pair_state"]
        assert comparison["decomposition"]["residual"] == pytest.approx(0, abs=1e-12)
        assert result["methods"]["all__pair_state"]["matched"]["normal"]["spans"] == 2
        null = result["order_null"]
        assert all(row["raw"] == null["observed"]["raw"] for row in null["draws"])
    assert prediction_digests(output) == before
    with tarfile.open(tmp_path / "cross_continuity_review.tar.gz") as archive:
        names = archive.getnames()
        assert "cross/continuity/tokens.npz" in names
        assert "cross/continuity/decomposition.csv" in names
        assert "cross/continuity/matching.json" in names
        assert not any("bank.npz" in name for name in names)
    with np.load(output / "continuity/tokens.npz") as saved:
        assert saved["label"].sum() == 18
        assert len(saved["answer_index"]) == 72
    assert "同答等长匹配：6 对" in (output / "continuity/summary.md").read_text()
    with (output / "continuity/profiles.csv").open() as stream:
        profiles = list(csv.DictReader(stream))
    assert {row["selection"] for row in profiles} == {"all_error", "matched_error", "matched_normal"}


def test_overlap_merge_keeps_adjacent_annotation_onsets():
    offsets = np.column_stack((np.arange(8), np.arange(1, 9)))
    spans = [{"start": 2, "end": 5}, {"start": 4, "end": 6}, {"start": 6, "end": 7}]
    block = {"gold": np.array([[2, 5], [4, 6], [6, 7]]), "views": label_views(offsets, spans)}
    normalized = normalize_annotations(block)
    np.testing.assert_array_equal(normalized["gold"], [[2, 6], [6, 7]])
    assert np.flatnonzero(normalized["views"]["span_onset_vs_normal"][0]).tolist() == [2, 6]
    assert np.flatnonzero(block["views"]["span_onset_vs_normal"][0]).tolist() == [2, 4, 6]


def test_cross_answer_auc_is_only_cross_answer_pairs():
    blocks = []
    for identity, values in enumerate(([0., .5, 1.], [2., 2., 3.])):
        labels = np.array([False, True, False])
        blocks.append({"record": {"id": str(identity), "source_id": str(identity)}, "tokens": 3,
                           "scores": {"raw": np.array(values)}, "alarms": {"raw": np.zeros(3, bool)},
                           "views": {"all_error": (labels, np.ones(3, bool))}})
    measured, _ = rank_view(blocks, "raw", "all_error")
    expected = [0., 0., 1., 1.]
    assert measured["cross_answer_pairs"] == 4
    assert measured["cross_answer_auroc"] == np.mean(expected)


def test_negative_neighborhood_cannot_admit_error_tokens_as_normal():
    with pytest.raises(SystemExit) as error:
        audit_main(["--neighborhood", "-1"])
    assert error.value.code == 2
