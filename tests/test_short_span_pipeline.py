"""Frozen archive integration: unchanged scores, review files, and capture positions."""

import csv
import json
import tarfile

import numpy as np
import pytest

from experiments.short_span_audit.__main__ import main
from experiments.short_span_audit.reanchor import reanchor_rows


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "continuity"
    root.mkdir()
    record = dict(id="a", source_id="s", task="QA", generator="test", split="test",
                  prompt_length=5, cache="/original/a.npz")
    answer = dict(record=record, tokens=12, gold=[[3, 5]], text="abcdefghijkl",
                  offsets=[[i, i + 1] for i in range(12)])
    pair = dict(pair_id="a:0", span_index=0, error_start=3, error_end=5,
                normal_start=8, normal_end=10, length=2)
    documents = {
        "answers.json": [answer],
        "matching.json": [dict(record=record, pairs=[pair], unmatched=[])],
        "input_settings.json": dict(dataset="/data", test_cache="/cache"),
        "audit.json": dict(methods=["all__raw"], groups={"QA|test": {"original_scored_tokens": {"all__raw": 12}}}),
    }
    for name, value in documents.items():
        (root / name).write_text(json.dumps(value))
    scores = np.array([0, 1, 0, 5, 4, 0, 0, 1, 2, 0, 1, 0], dtype=np.float32)
    labels = np.zeros(12, bool)
    labels[3:5] = True
    np.savez(root / "tokens.npz", answer_index=np.zeros(12, int), label=labels,
             common_finite=np.ones(12, bool), all__raw=scores, all__raw__alarm=scores >= 4)
    return root


def test_cpu_pipeline_exports_frozen_metrics_and_exact_capture_cohort(snapshot, tmp_path):
    before = (snapshot / "tokens.npz").read_bytes()
    output = tmp_path / "review"
    report = main(["--input", str(snapshot), "--output", str(output), "--bootstrap", "0"])
    group = report["groups"]["RAGTruth|QA|test"]["methods"]["all__raw"]["frozen"]
    assert group["bins"]["1-2"]["error"]["before_end_detected"] == 1
    assert group["bins"]["1-2"]["matched"]["recovery"]["normal"]["spans"] == 1
    cohort = json.loads((output / "cohort.json").read_text())
    assert [(row["target"], row["side"]) for row in cohort[0]["targets"]] == [
        (3, "error"), (4, "error"), (8, "normal"), (9, "normal")]
    assert (snapshot / "tokens.npz").read_bytes() == before
    with tarfile.open(tmp_path / "review_review.tar.gz") as bundle:
        assert {"review/audit.json", "review/cohort.json", "review/metrics.csv"} <= set(bundle.getnames())


def test_unconfirmed_reanchor_is_not_called_absent(snapshot, tmp_path):
    output = tmp_path / "review"
    main(["--input", str(snapshot), "--output", str(output), "--bootstrap", "0"])
    cohort = json.loads((output / "cohort.json").read_text())
    events = tmp_path / "events"
    events.mkdir()
    fields = ["id", "source_id", "task", "generator", "threshold", "target", "query"]
    with (events / "nodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(id="a", source_id="s", task="QA", generator="test",
                             threshold=.1, target=3, query=7))
    rows = reanchor_rows(events, cohort)
    assert not rows[0]["prior_confirmed_event"]
    assert rows[0]["decision_confirmed_event"]
    assert rows[0]["interpretation"].endswith("or_unconfirmed")
    cohort[0]["record"]["prompt_length"] += 1
    with pytest.raises(ValueError, match="identity/query mismatch"):
        reanchor_rows(events, cohort)
