"""A partial evaluation reads finalized scores; it never runs or completes a scorer."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from experiments.unsupervised_token_graph.reanchor_evaluate import evaluate, main, prediction_records


VERSION = "source-carrier-information-v1"


def publish_sample(root, rid, split="train", erroneous=True):
    """Use the runner's saved NPZ contract, without invoking a model or graph."""
    response = "abcd"
    record = dict(id=str(rid), source_id="source_" + str(rid), split=split,
                  task="QA", generator="fixture", file=f"samples/attention/{rid}.npz",
                  response_sha256=hashlib.sha256(response.encode()).hexdigest())
    path = root / record["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.array([[np.nan, .8, .2, .1], [np.nan, .7, .1, .05]])
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, record_json=json.dumps(record), cache_stamp=[123, 456],
                            source_mismatch_bits=values, permuted_source_mismatch_bits=values / 2,
                            event_strength=np.nan_to_num(values), prompt_reach=np.full((2, 4), .8),
                            unknown_mass=np.zeros((2, 4)), layer_ids=[0, 1], head_ids=[0, 0],
                            query_positions=[2, 3, 4, 5], prediction_positions=[3, 4, 5, 6],
                            offsets=[[0, 1], [1, 2], [2, 3], [3, 4]], prompt_length=2, total_tokens=6,
                            cache_format="canonical_csr", token_ids=np.arange(6))
    temporary.replace(path)
    gold = dict(id=str(rid), source_id=record["source_id"], split=split, response=response,
                labels=[dict(start=2, end=3)] if erroneous else [])
    return record, gold


@pytest.fixture
def interrupted(tmp_path):
    root = tmp_path / "result"
    root.mkdir()
    records, labels = [], []
    for rid in range(3):
        record, gold = publish_sample(root, rid, erroneous=rid != 1)
        records.append(record); labels.append(gold)
    annotations = tmp_path / "response.jsonl"
    annotations.write_text("\n".join(map(json.dumps, labels)), encoding="utf-8")
    (root / "settings.json").write_text(json.dumps(dict(version=VERSION, labels_read=False,
                                                         annotations=str(annotations))))
    # A killed writer may leave an unreadable temporary file. It is not a sample.
    (root / "samples/attention/unfinished.partial").write_bytes(b"interrupted archive")
    return root, annotations, records


def test_interrupted_run_without_summary_or_complete_can_be_previewed(interrupted):
    root, labels, records = interrupted
    report = evaluate(root, labels, root / "evaluation_partial.json", split="train",
                      bootstrap=10, completed_only=True)
    assert report["evaluation_scope"] == "completed_samples_preview"
    assert report["completed_samples_found"] == 3
    assert report["evaluated_responses"] == 3
    assert report["evaluated_records"] == [{k: r[k] for k in ("id", "source_id", "split", "file")} for r in records]
    views = report["groups"]["ALL"]["views"]
    assert len(views) == 7
    metric = views["all_error"]["event_strength"]
    assert metric["eligible_tokens"] == 12
    assert metric["evaluated_tokens"] == 9
    assert metric["coverage"] == pytest.approx(.75)
    assert metric["pooled"]["auroc"] is not None
    assert "common_coverage_real_vs_permuted" in report["groups"]["ALL"]


def test_preview_keeps_checkpoint_files_and_existing_full_report_unchanged(interrupted):
    root, labels, _ = interrupted
    (root / "evaluation.json").write_text("old full report")
    before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    evaluate(root, labels, root / "evaluation_partial.json", split="train", bootstrap=0, completed_only=True)
    for relative, (data, modified) in before.items():
        assert (root / relative).read_bytes() == data
        assert (root / relative).stat().st_mtime_ns == modified
    assert not (root / "summary.json").exists()
    assert not (root / "complete.json").exists()


def test_full_evaluation_still_requires_completed_run(interrupted):
    root, labels, _ = interrupted
    with pytest.raises(ValueError, match="--completed-only"):
        evaluate(root, labels, split="train", bootstrap=0)


def test_full_and_preview_metrics_agree_on_identical_cohort(interrupted):
    root, labels, records = interrupted
    (root / "complete.json").write_text(json.dumps(dict(complete=True, responses=3)))
    (root / "summary.json").write_text(json.dumps(dict(version=VERSION, labels_read=False, responses=records)))
    full = evaluate(root, labels, split="train", bootstrap=0)
    preview = evaluate(root, labels, split="train", bootstrap=0, completed_only=True)
    assert full["evaluation_scope"] == "full_run"
    assert full["groups"] == preview["groups"]


def test_snapshot_precedes_annotation_read(interrupted, monkeypatch):
    root, labels, _ = interrupted
    original = Path.open
    def open_with_new_sample(path, *args, **kwargs):
        if path == labels:
            publish_sample(root, 99)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_with_new_sample)
    report = evaluate(root, labels, split="train", bootstrap=0, completed_only=True)
    assert report["completed_samples_found"] == 3
    assert (root / "samples/attention/99.npz").exists()
    assert report["evaluated_responses"] == 3


def test_preview_ignores_stale_summary(interrupted):
    root, labels, _ = interrupted
    (root / "summary.json").write_text(json.dumps(dict(version=VERSION, labels_read=False, responses=[])))
    report = evaluate(root, labels, split="train", bootstrap=0, completed_only=True)
    assert report["completed_samples_found"] == 3


def test_no_completed_npz_is_an_explicit_error(interrupted):
    root, labels, _ = interrupted
    for path in (root / "samples").rglob("*.npz"):
        path.unlink()
    with pytest.raises(ValueError, match="no completed sample"):
        evaluate(root, labels, split="train", bootstrap=0, completed_only=True)


def test_wrong_split_is_reported_not_relabelled(interrupted):
    root, labels, _ = interrupted
    with pytest.raises(ValueError, match="saved splits=.*train"):
        evaluate(root, labels, split="test", bootstrap=0, completed_only=True)


def test_preview_refuses_full_report_path(interrupted):
    root, labels, _ = interrupted
    with pytest.raises(ValueError, match="do not overwrite"):
        evaluate(root, labels, root / "evaluation.json", split="train", bootstrap=0, completed_only=True)


def test_preview_does_not_bypass_label_free_or_identity_checks(interrupted):
    root, labels, _ = interrupted
    settings = root / "settings.json"
    value = json.loads(settings.read_text()); value["labels_read"] = True
    settings.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="label-free"):
        prediction_records(root, completed_only=True)
    value["labels_read"] = False
    settings.write_text(json.dumps(value))
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    rows[0]["response"] = "abce"
    labels.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluate(root, labels, split="train", bootstrap=0, completed_only=True)


def rewrite_npz(path, update):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    update(arrays)
    np.savez_compressed(path, **arrays)


def test_missing_offsets_are_not_fabricated(interrupted):
    root, labels, _ = interrupted
    rewrite_npz(root / "samples/attention/0.npz", lambda a: a.pop("offsets"))
    with pytest.raises(ValueError, match="offsets are required"):
        evaluate(root, labels, split="train", bootstrap=0, completed_only=True)


def test_record_path_must_match_saved_file(interrupted):
    root, _, _ = interrupted
    def change(arrays):
        row = json.loads(str(arrays["record_json"])); row["file"] = "samples/other.npz"
        arrays["record_json"] = np.asarray(json.dumps(row))
    rewrite_npz(root / "samples/attention/0.npz", change)
    with pytest.raises(ValueError, match="path and record disagree"):
        prediction_records(root, completed_only=True)


def test_duplicate_ids_are_not_counted_twice(interrupted):
    root, labels, _ = interrupted
    def change(arrays):
        row = json.loads(str(arrays["record_json"])); row["id"] = "0"
        arrays["record_json"] = np.asarray(json.dumps(row))
    rewrite_npz(root / "samples/attention/1.npz", change)
    with pytest.raises(ValueError, match="unique response IDs"):
        evaluate(root, labels, split="train", bootstrap=0, completed_only=True)


def test_one_class_reports_null_auroc(interrupted):
    root, labels, _ = interrupted
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    for row in rows:
        row["labels"] = []
    labels.write_text("\n".join(map(json.dumps, rows)))
    report = evaluate(root, labels, split="train", bootstrap=5, completed_only=True)
    assert report["groups"]["ALL"]["views"]["all_error"]["event_strength"]["pooled"]["auroc"] is None


def test_cli_uses_saved_annotations_and_prints_preview_scope(interrupted, capsys):
    root, _, _ = interrupted
    main(["--predictions", str(root), "--output", str(root / "evaluation_partial.json"),
          "--split", "train", "--bootstrap", "0", "--completed-only"])
    assert '"evaluation_scope": "completed_samples_preview"' in capsys.readouterr().out
    assert (root / "evaluation_partial.json").is_file()


@pytest.mark.parametrize("via_environment", [False, True])
def test_shell_preview_uses_separate_report_and_never_calls_analysis(interrupted, via_environment):
    root, labels, _ = interrupted
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(SPLIT="train", OUTPUT=str(root), ANNOTATIONS=str(labels), BOOTSTRAP="0")
    env.pop("COMPLETED_ONLY", None)
    if via_environment:
        env["COMPLETED_ONLY"] = "1"
    command = ["bash", str(repo / "experiments/unsupervised_token_graph/evaluate.sh")]
    if not via_environment:
        command.append("--completed-only")
    result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (root / "evaluation_partial.json").is_file()
    assert not (root / "evaluation.json").exists()
    assert not (root / "complete.json").exists()
