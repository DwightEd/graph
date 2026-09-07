import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reanchor_flow import discover
from experiments.reanchor_flow.artifacts import save_json, save_result
from experiments.reanchor_flow.tests.test_native_report import _sample
from experiments.reanchor_flow.tests.test_scan_dataset import _capture


def _old_scans(root):
    root.mkdir()
    train = _capture(root, records={"a": ("source-a", "QA"), "b": ("source-a", "QA"),
                                    "c": ("source-c", "Summary"), "overlap": ("shared", "QA")})
    train.rename(root / "train")
    _capture(root, records={"test-a": ("shared", "QA"), "test-c": ("source-test-c", "Summary")})
    for split in ("train", "test"):
        path = root / split / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["config"].update(split=split, model="absent-local-model", model_dtype="float32")
        save_json(path, manifest)


def _fake_trace(model, ids, start, args):
    rows = len(ids) - start
    trace, _ = _sample(np.arange(rows) % 2)
    rng = np.random.default_rng(len(ids))
    trace.update(
        native_trace_schema=1, token_ids=ids, response_start=start,
        row_position=np.arange(start - 1, len(ids) - 1), labels_used_for_capture=False,
        head_sketch=rng.normal(size=(2, 3, rows, 2)).astype(np.float32),
        mlp_sketch=rng.normal(size=(2, rows, 2)).astype(np.float32),
    )
    return trace


def _arguments(scans, output, *extra):
    return ["--scans", str(scans), "--output", str(output), "--bootstrap", "8",
            "--fit-rows", "6", "--components", "2", "--patterns", "2", "--no-plot", *extra]


@pytest.fixture(autouse=True)
def _tokenizer(monkeypatch):
    monkeypatch.setattr(discover, "_load_tokenizer", lambda path: SimpleNamespace(
        decode=lambda ids: f"token-{ids[0]}"))


def test_all_cli_freezes_every_prediction_before_labels_and_preserves_full_label_horizon(tmp_path, monkeypatch):
    scans, output = tmp_path / "scans", tmp_path / "out"
    _old_scans(scans)
    loads = []
    monkeypatch.setattr(discover, "_load_model", lambda *a: loads.append(a) or object())
    monkeypatch.setattr(discover, "_capture_trace", _fake_trace)
    joined = []

    class LabelStore:
        def __init__(self, dataset, dataset_root=None):
            assert (output / "native_patterns.npz").exists()
            assert (output / "source_partition.json").exists()
            assert len(list((output / "train" / "predictions").rglob("*.npz"))) == 4
            assert len(list((output / "test" / "predictions").rglob("*.npz"))) == 2
            assert len(list((output / "train" / "examples").rglob("*.json"))) == 4
            assert len(list((output / "test" / "examples").rglob("*.json"))) == 2
            example = json.loads((output / "test" / "examples" / "QA" / "test-a.json").read_text())
            assert example["labels_used"] is False
            assert "largest_head_sketch_change" in example["transition_examples"][0]
            assert "largest_mlp_sketch_change" in example["transition_examples"][0]

        def load(self, scan):
            assert scan.metadata["full_response_tokens"] == 4
            assert scan.rows == 4
            joined.append(scan.sample_id)
            return np.array([0, 0, 1, -1])

    monkeypatch.setattr(discover, "ScanLabelStore", LabelStore)
    report, metrics = discover.main(_arguments(scans, output, "--max-response-tokens", "3"))
    assert len(loads) == 1
    assert set(joined) == {"test-a", "test-c"}
    assert metrics["groups"]["ALL"]["tokens"] == 6
    assert metrics["groups"]["ALL"]["positives"] == 2
    assert report["discovery"]["labels_used_for_fit"] is False
    assert report["discovery"]["head_shape"] == (2, 3, 2)
    partition = json.loads((output / "source_partition.json").read_text())
    assert partition["excluded_overlap_sources"] == ["shared"]
    assert partition["train_sources"] == ["source-a", "source-c"]
    assert partition["rows_used"] == 6
    assert (output / "mechanism_report.json").exists()
    assert (output / "detection_report.json").exists()
    assert (output / "summary.md").exists()
    saved = discover._load(output / "test" / "traces" / "QA" / "test-a.npz")
    np.testing.assert_array_equal(saved["token_text"], [f"token-{i}" for i in range(10, 16)])


def test_capture_resume_never_loads_model_or_labels_and_checks_tokens(tmp_path, monkeypatch):
    scans, output = tmp_path / "scans", tmp_path / "out"
    _old_scans(scans)
    monkeypatch.setattr(discover, "_load_model", lambda *a: object())
    monkeypatch.setattr(discover, "_capture_trace", _fake_trace)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed capture must not load model or labels")

    monkeypatch.setattr(discover, "ScanLabelStore", forbidden)
    argv = _arguments(scans, output, "--phase", "capture")
    discover.main(argv)
    monkeypatch.setattr(discover, "_load_model", forbidden)
    discover.main(argv)
    path = scans / "train" / "a.npz"
    changed = discover._load(path)
    changed["token_ids"][0] += 1
    save_result(path, changed)
    with pytest.raises(ValueError, match="source tokens"):
        discover.main(argv)


def test_capture_resume_rejects_projection_and_horizon_changes(tmp_path, monkeypatch):
    scans, output = tmp_path / "scans", tmp_path / "out"
    _old_scans(scans)
    monkeypatch.setattr(discover, "_load_model", lambda *a: object())
    monkeypatch.setattr(discover, "_capture_trace", _fake_trace)
    argv = _arguments(scans, output, "--phase", "capture")
    discover.main(argv)
    for change in (("--seed", "9"), ("--max-response-tokens", "2"), ("--sketch-dim", "8")):
        with pytest.raises(ValueError, match="configuration changed"):
            discover.main([*argv, *change])


def test_fit_bank_source_and_sample_balance_excludes_all_test_sources(tmp_path):
    entries = []
    for index, (sample, source) in enumerate((("a", "one"), ("b", "one"),
                                            ("c", "two"), ("overlap", "test"))):
        entry = {"dataset_sample_id": sample, "source_id": source, "task_type": "QA",
                 "path": f"traces/QA/{sample}.npz", "rows": 4}
        entries.append(entry)
        save_result(tmp_path / "train" / entry["path"], {
            "head_sketch": np.full((1, 2, 4, 2), index, dtype=float),
            "mlp_sketch": np.full((1, 4, 2), index, dtype=float),
        })
    manifests = {"train": {"entries": entries}, "test": {"entries": [{"source_id": "test"}]}}
    bank, head, mlp, partition = discover.fit_row_bank(tmp_path, manifests, 6, 9)
    assert head == (1, 2, 2) and mlp == (1, 2)
    assert bank.shape == (6, 6)
    assert not (bank == 3).any()
    counts = Counter()
    for sample in partition["samples"]:
        counts[sample["source_id"]] += sample["fit_rows"]
    assert counts == {"one": 3, "two": 3}
    sample_counts = [s["fit_rows"] for s in partition["samples"] if s["source_id"] == "one"]
    assert sorted(sample_counts) == [1, 2]
    again = discover.fit_row_bank(tmp_path, manifests, 6, 9)[0]
    np.testing.assert_array_equal(again, bank)
    minimum = discover.fit_row_bank(tmp_path, manifests, 2, 9)[3]
    assert len(minimum["samples"]) == 2
    assert any(sample["row_indices"][0] > 0 for sample in minimum["samples"])
    assert "uniform random" in minimum["sampling"]
    with pytest.raises(ValueError, match="all 2"):
        discover.fit_row_bank(tmp_path, manifests, 1, 9)


def test_analyze_only_reuses_native_traces_without_model_capture(tmp_path, monkeypatch):
    scans, output = tmp_path / "scans", tmp_path / "out"
    _old_scans(scans)
    monkeypatch.setattr(discover, "_load_model", lambda *a: object())
    monkeypatch.setattr(discover, "_capture_trace", _fake_trace)
    discover.main(_arguments(scans, output, "--phase", "capture"))

    def forbidden(*args, **kwargs):
        raise AssertionError("analyze must not recapture native vectors")

    monkeypatch.setattr(discover, "_load_model", forbidden)
    monkeypatch.setattr(discover, "_capture_trace", forbidden)

    class LabelStore:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, scan):
            return np.array([0, 0, 1, -1])

    monkeypatch.setattr(discover, "ScanLabelStore", LabelStore)
    _, metrics = discover.main(_arguments(scans, output, "--phase", "analyze"))
    assert metrics["groups"]["ALL"]["tokens"] == 8
    assert metrics["groups"]["ALL"]["unknown_tokens"] == 2


def test_default_plot_branch_preserves_vector_changes_and_uses_saved_token_text(tmp_path, monkeypatch):
    scans, output = tmp_path / "scans", tmp_path / "out"
    _old_scans(scans)
    monkeypatch.setattr(discover, "_load_model", lambda *a: object())
    monkeypatch.setattr(discover, "_capture_trace", _fake_trace)

    class LabelStore:
        def __init__(self, *args, **kwargs):
            assert len(list((output / "test" / "examples").rglob("*.json"))) == 2

        def load(self, scan):
            return np.array([0, 0, 1, -1])

    monkeypatch.setattr(discover, "ScanLabelStore", LabelStore)
    discover.main([arg for arg in _arguments(scans, output) if arg != "--no-plot"])
    metadata = json.loads((output / "figures" / "QA" / "test-a.json").read_text())
    assert metadata["focus_token_text"].startswith("token-")
    assert metadata["transition_examples"]
    for example in metadata["transition_examples"]:
        assert "largest_head_sketch_change" in example
        assert "largest_mlp_sketch_change" in example
    assert (output / "mode_writes_ALL.png").exists()
    assert (output / "figures" / "QA" / "test-a.png").exists()
