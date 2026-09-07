from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.reanchor_flow import scan_analyze
from experiments.reanchor_flow.routing_transition import (
    RoutingSequence,
    RoutingTransitionModel,
)
from experiments.reanchor_flow.scan_dataset import BUCKET_NAMES, ScanDataset


def _write_scan(root, sample_id, source_id, seed):
    rng = np.random.default_rng(seed)
    rows, start, local_window = 64, 10, 4
    labels = np.repeat([0, 1, 0, 1], 16).astype(np.int8)
    labels[0] = -1
    transport = rng.lognormal(0, 0.12, size=(1, 2, rows, 4))
    transport *= np.array([1.0, 2.0, 1.0, 5.0])
    transport[0, 0, labels == 1, 0] *= 0.5
    transport[..., : local_window + 2, 2] = 0
    transport[..., 0, 3] = 0
    score = np.zeros((1, 2, rows))
    events = np.array([8, 20, 26, 40, 52, 58])
    score[0, 0, events] = 0.9
    transport[0, 0, events] = [4.0, 1.0, 2.0, 1.0]
    q = np.arange(start - 1, start + rows - 1)
    source = np.broadcast_to(np.array([0, 1, start, -1]), (1, 2, rows, 4)).copy()
    source[..., : local_window + 2, 2] = -1
    source[..., 1:, 3] = q[1:]
    unit = np.where(source >= 0, np.broadcast_to(np.arange(4), source.shape), -1)
    path = root / "scans" / "QA" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_scan_schema=1,
        dataset_sample_id=sample_id,
        source_id=source_id,
        task_type="QA",
        labels_used_for_capture=False,
        response_start=start,
        sequence_length=start + rows,
        full_response_tokens=rows,
        processed_response_tokens=rows,
        local_window=local_window,
        route_row_position=q,
        reanchor_bucket_name=np.asarray(BUCKET_NAMES),
        token_ids=np.arange(start + rows),
        reanchor_bucket_transport=transport,
        reanchor_bucket_attention=transport / transport.sum(axis=-1, keepdims=True),
        reanchor_bucket_source_position=source,
        reanchor_bucket_source_unit_id=unit,
        reanchor_score=score,
    )
    return {
        "source_id": source_id,
        "task_type": "QA",
        "scan": str(path.relative_to(root)),
    }, labels


def _capture(tmp_path):
    capture = tmp_path / "capture"
    annotations = {}
    for split in ("train", "test"):
        root = capture / split
        sources = (
            [f"train-source-{i}" for i in range(14)]
            if split == "train"
            else [f"test-source-{i}" for i in range(6)]
        )
        if split == "train":
            sources.append("test-source-0")
        samples = {}
        for source_index, source in enumerate(sources):
            for answer in range(2):
                sample_id = f"{split}-{source_index}-{answer}"
                sample, labels = _write_scan(
                    root, sample_id, source, 100 * source_index + answer
                )
                samples[sample_id] = sample
                annotations[split, sample_id] = labels
        manifest = {
            "subset_manifest_schema": 3,
            "analysis_complete": True,
            "labels_used_for_capture": False,
            "config": {
                "split": split,
                "dataset_root": str(tmp_path / "unused-cache" / split),
                "model": "tiny-synthetic",
                "model_dtype": "float32",
                "tokenizer": "synthetic",
                "flow_signal": "message",
                "local_window": 4,
            },
            "samples": samples,
            "selection": [
                {"sample_id": key, "source_id": value["source_id"], "task_type": "QA"}
                for key, value in samples.items()
            ],
        }
        (root / "run_manifest.json").write_text(json.dumps(manifest))
    return capture, annotations


def test_complete_offline_audit_scores_all_tokens_before_any_labels(
    tmp_path, monkeypatch
):
    capture, annotations = _capture(tmp_path)
    output = tmp_path / "analysis"
    label_calls = []

    class Labels:
        def __init__(self, scans, *, dataset_root=None):
            assert (output / "frozen_detector.json").is_file()
            frozen = json.loads((output / "frozen_detector.json").read_text())
            assert frozen["scored_tokens"] == 12 * 64
            assert frozen["density_uses_labels"] is False
            for prediction in frozen["predictions"]:
                assert (output / prediction["path"]).is_file()
                if not label_calls:
                    with np.load(
                        output / prediction["path"], allow_pickle=False
                    ) as saved:
                        assert "labels" not in saved.files
                        assert np.isfinite(saved["routing_joint"]).all()
            self.scans = scans
            label_calls.append((scans.split, "construct", scans.sample_ids))

        def load(self, scan):
            assert (output / "frozen_detector.json").is_file()
            label_calls.append((self.scans.split, "load", scan.sample_id))
            return annotations[self.scans.split, scan.sample_id].copy()

    monkeypatch.setattr(scan_analyze, "ScanLabelStore", Labels)
    report = scan_analyze.run_analysis(
        capture,
        output,
        bootstrap=10,
        supervised_probe=True,
        plot=True,
        events=True,
    )
    partition = json.loads((output / "source_partition.json").read_text())
    fit = set(partition["fit_source_ids"])
    calibration = set(partition["calibration_source_ids"])
    test = set(partition["test_source_ids"])
    assert fit and calibration and test
    assert not fit & calibration
    assert not (fit | calibration) & test
    assert len(fit | calibration) == 14
    assert set(partition["excluded_train_sample_ids"]) == {"train-14-0", "train-14-1"}
    assert label_calls[0][0] == "train"  # optional supervised probe starts after freeze
    train = ScanDataset(capture / "train")
    assert set(label_calls[0][2]) == {
        record.sample_id for record in train.records if record.source_id in fit
    }

    group = report["groups"]["QA"]
    assert group["tokens"] == 768
    assert group["known_tokens"] == 756
    assert group["unknown_tokens"] == 12
    assert group["sources"] == 6
    assert report["coverage"]["first_token_fallbacks"] == 12
    assert (
        report["coverage"]["scored_tokens"]
        == report["coverage"]["full_response_tokens"]
    )
    for item in group["scores"].values():
        assert 0 <= item["auroc"] <= 1
        assert 0 <= item["auprc"] <= 1
        assert item["auroc_ci95"] is not None
    assert set(report["supervised_diagnostic"]["groups"]["QA"]["scores"]) == {
        "supervised_routing",
        "supervised_position",
    }
    assert (output / "detection_report.json").is_file()
    assert (output / "detection_curves.png").stat().st_size > 1000
    events = json.loads((output / "events" / "event_audit.json").read_text())
    assert events["selection"]["QA"]
    assert events["tasks"]["QA"][0]["groups"]["hallucinated"]["matched_pair_count"] > 0
    assert list((output / "events").glob("*.png"))

    frozen = json.loads((output / "frozen_detector.json").read_text())
    record = frozen["predictions"][0]
    scans = ScanDataset(capture / "test")
    scan = scans.load(record["sample_id"], fields=("reanchor_bucket_transport",))
    model = RoutingTransitionModel.load(output / "models" / "QA_routing.npz")
    normalizer = scan_analyze.ScoreCalibration.load(
        output / "models" / "QA_calibration.npz"
    )
    rescored = normalizer.transform(model.score(RoutingSequence.from_scan(scan)))
    with np.load(output / record["path"], allow_pickle=False) as prediction:
        for name in scan_analyze.SCORES:
            assert np.isfinite(prediction[name]).all()
            np.testing.assert_allclose(prediction[name], rescored[name])
        assert prediction["labels"][0] == -1
        assert prediction["response_index"][0] == 0
        assert not prediction["has_previous"][0]
        np.testing.assert_array_equal(
            prediction["prediction_position"], prediction["query_position"] + 1
        )
        assert prediction["supervised_routing"].shape == (64,)
        assert "reanchor_bucket_transport" not in prediction.files


def test_capture_configuration_mismatch_stops_before_labels_and_outputs(tmp_path):
    capture, _ = _capture(tmp_path)
    path = capture / "test" / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["local_window"] = 8
    path.write_text(json.dumps(manifest))
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="local_window"):
        scan_analyze.run_analysis(capture, output)
    assert not output.exists()


def test_cli_help_imports_no_torch_or_transformers():
    source = """
import importlib.abc
import runpy
import sys
class NoModelImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('offline CLI attempted model import: ' + fullname)
sys.meta_path.insert(0, NoModelImport())
sys.argv = ['scan_analyze', '--help']
runpy.run_module('experiments.reanchor_flow.scan_analyze', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[3],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--supervised-probe" in result.stdout
    assert "--scans" in result.stdout
