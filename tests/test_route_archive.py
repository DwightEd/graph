import json

import numpy as np
import pytest

from route_graph.archive import FeatureArchive
from route_graph.data import text_digest, write_jsonl


def fixture_files(tmp_path):
    records, rows = [], []
    for i in range(3):
        record = {
            "schema": "route-graph/response@1",
            "id": str(i),
            "source_id": str(i),
            "split": "train" if i == 0 else "test",
            "task": "QA",
            "generator": "fixture",
            "prompt": "Evidence",
            "source_span": [0, 8],
            "response": "AB",
            "response_sha256": text_digest("AB"),
        }
        records.append(record)
        for t in range(2):
            rows.append(
                dict(
                    schema="route-graph/features@1",
                    response_id=str(i),
                    **{
                        k: record[k]
                        for k in (
                            "source_id",
                            "split",
                            "task",
                            "generator",
                            "response_sha256",
                        )
                    },
                    token_index=t,
                    token_count=2,
                    prompt_tokens=4,
                    predictor_index=3 + t,
                    char_span=[t, t + 1],
                    token_id=1 + t,
                    candidate_ids=[1, 2],
                    entropy=0.3,
                    negative_margin=-0.4,
                    observed=[0.5, 0.75],
                    null=[0.25, 0.5],
                    residual=[0.25, 0.25],
                    signal=[0.2, 0.3],
                )
            )
    inputs, features = tmp_path / "input.jsonl", tmp_path / "features.jsonl"
    write_jsonl(inputs, records)
    write_jsonl(features, rows[:-1])
    return inputs, features, rows


def test_recovery_preserves_exact_values_and_reuses_complete_files(tmp_path):
    inputs, features, original = fixture_files(tmp_path)
    archive = FeatureArchive(tmp_path / "archive")
    before = features.read_bytes()
    report = archive.recover(features, inputs)
    assert report["complete_responses"] == 2
    assert report["complete_tokens"] == 4
    assert report["rejected"][0]["response_id"] == "2"
    assert report["original_capture_complete"] is False
    assert not (archive.directory / "manifest.json").exists()
    recovered = [r for group in archive.responses() for r in group]
    assert len(recovered) == 4
    for expected, actual in zip(original[:4], recovered, strict=True):
        for view in ("observed", "null", "residual", "signal"):
            np.testing.assert_array_equal(actual[view], expected[view])
    mtimes = {p.name: p.stat().st_mtime_ns for p in archive.directory.glob("*.npz")}
    assert archive.recover(features, inputs) == report
    assert mtimes == {
        p.name: p.stat().st_mtime_ns for p in archive.directory.glob("*.npz")
    }
    assert features.read_bytes() == before


def test_recovery_rejects_changed_source_identity(tmp_path):
    inputs, features, rows = fixture_files(tmp_path)
    rows[0]["source_id"] = "different"
    features.write_text("\n".join(map(json.dumps, rows)))
    report = FeatureArchive(tmp_path / "archive").recover(features, inputs)
    assert report["complete_responses"] == 2
    assert report["rejected"][0]["reason"] == "input identity mismatch"


def test_torn_final_line_is_reported_but_interior_corruption_fails(tmp_path):
    inputs, features, _ = fixture_files(tmp_path)
    with features.open("a") as f:
        f.write('{"response_id":')
    report = FeatureArchive(tmp_path / "archive").recover(features, inputs)
    assert report["complete_responses"] == 2
    assert any(r["reason"] == "truncated final JSON line" for r in report["rejected"])
    with features.open("a") as f:
        f.write("\n{}\n")
    with pytest.raises(ValueError, match="inside capture"):
        FeatureArchive(tmp_path / "archive2").recover(features, inputs)


def test_changed_archive_and_changed_input_are_not_silently_reused(tmp_path):
    inputs, features, _ = fixture_files(tmp_path)
    archive = FeatureArchive(tmp_path / "archive")
    archive.recover(features, inputs)
    next(archive.directory.glob("*.npz")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(archive.responses())
    with features.open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="inputs changed"):
        archive.recover(features, inputs)


@pytest.mark.parametrize("change", ["label", "missing", "bad_candidates", "bad_token"])
def test_archive_rejects_label_fields_and_malformed_identities(tmp_path, change):
    inputs, features, rows = fixture_files(tmp_path)
    if change == "label":
        rows[0]["unexpected_label"] = 1
    elif change == "missing":
        rows[0].pop("token_id")
    elif change == "bad_candidates":
        rows[0]["candidate_ids"] = [1, 1]
    else:
        rows[0]["token_id"] = True
    features.write_text("\n".join(map(json.dumps, rows)))
    result = FeatureArchive(tmp_path / "archive").recover(features, inputs)
    assert result["complete_responses"] == 2
    assert result["rejected"][0]["response_id"] == "0"


def test_width_mismatch_does_not_pass_as_complete(tmp_path):
    inputs, features, rows = fixture_files(tmp_path)
    for row in rows[2:4]:
        for view in ("observed", "null", "residual", "signal"):
            row[view] = row[view][:1]
    features.write_text("\n".join(map(json.dumps, rows)))
    archive = FeatureArchive(tmp_path / "archive")
    with pytest.raises(ValueError, match="across responses"):
        archive.recover(features, inputs)
    assert not (archive.directory / "index.json").exists()


def test_interrupted_metadata_temporary_files_preserve_completed_responses(tmp_path):
    inputs, features, _ = fixture_files(tmp_path)
    archive = FeatureArchive(tmp_path / "archive")
    report = archive.recover(features, inputs)
    (archive.directory / "settings.json.partial").write_text('{"broken":')
    for path in archive.directory.glob("*.npz"):
        path.with_suffix(".json.partial").write_text('{"broken":')
    assert archive.recover(features, inputs) == report
