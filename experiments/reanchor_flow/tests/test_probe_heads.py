import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experiments.reanchor_flow.probe_heads import (
    HeadReadout,
    SourceGaps,
    _collect,
    build_report,
    freeze_selection,
    run_audit,
)
from experiments.reanchor_flow.routing_probe import SupervisedRoutingProbe
from experiments.reanchor_flow.routing_transition import RoutingSequence
from experiments.reanchor_flow.scan_dataset import ScanDataset
from experiments.reanchor_flow.tests.test_scan_dataset import _capture


def _probe(shape=(2, 3, 4)):
    rng = np.random.default_rng(17)
    probe = SupervisedRoutingProbe()
    probe.state_shape = shape
    probe.fit_metadata = {"max_rows_per_sample": 128}
    for field in ("mean", "scale", "coefficient", "intercept"):
        setattr(probe, field, {})
    for name, width in zip(probe.NAMES, (2 * int(np.prod(shape)) + 7, 7), strict=True):
        probe.mean[name] = rng.normal(size=width)
        probe.scale[name] = np.exp(rng.normal(size=width))
        probe.coefficient[name] = rng.normal(size=width)
        probe.intercept[name] = float(rng.normal())
    return probe


def _sequence(rows=601):
    rng = np.random.default_rng(42)
    return RoutingSequence(
        rng.normal(size=(rows, 2, 3, 4)), rng.normal(size=(rows, 8)),
        np.arange(rows),
    )


def test_exact_decomposition_matches_numeric_saved_probe_across_batches(tmp_path):
    original = _probe()
    path = tmp_path / "probe.npz"
    original.save(path)
    probe = SupervisedRoutingProbe.load(path)
    sequence = _sequence()
    result = HeadReadout(probe).decompose(sequence)
    np.testing.assert_allclose(result.score, probe.score(sequence)["supervised_routing"], atol=1e-12)
    assert result.current.shape == (601, 2, 3)
    # This is the context term in the routing model, not the separately fitted
    # supervised_position model, whose coefficient and intercept are different.
    width = 2 * int(np.prod(probe.state_shape))
    expected = ((sequence.context[:, 1:] - probe.mean["supervised_routing"][width:])
                / probe.scale["supervised_routing"][width:]
                @ probe.coefficient["supervised_routing"][width:])
    np.testing.assert_allclose(result.context, expected)


def test_prefix_invariance_and_gap_difference_respect_original_readout():
    model = HeadReadout(_probe())
    sequence = _sequence(30)
    # A missing predecessor row must produce the trained zero-difference input.
    index = sequence.response_index.copy()
    index[10:] += 3
    sequence = replace(sequence, response_index=index)
    prefix = replace(sequence, state=sequence.state[:19], context=sequence.context[:19],
                     response_index=sequence.response_index[:19])
    complete, short = model.decompose(sequence), model.decompose(prefix)
    for field in ("current", "difference", "context"):
        np.testing.assert_array_equal(getattr(complete, field)[:19], getattr(short, field))
    np.testing.assert_allclose(complete.score, model.probe.score(sequence)["supervised_routing"])
    np.testing.assert_allclose(complete.difference[0], complete.difference[10])


def test_source_gap_averages_samples_then_sources_and_excludes_unpaired():
    gaps = SourceGaps((1, 2))
    gaps.add("one", np.array([[2, -1]]))
    gaps.add("one", np.array([[4, 3]]))
    gaps.add("two", np.array([[-1, -2]]))
    gaps.add("three", None)
    summary = gaps.summary()
    assert summary["paired_sources"] == 2
    assert summary["paired_samples"] == 3
    np.testing.assert_allclose(summary["mean_gap"], [[1, -0.5]])
    np.testing.assert_allclose(summary["positive_source_fraction"], [[0.5, 0.5]])


def test_selection_uses_calibration_only_and_has_no_empty_test_fallback():
    calibration = SourceGaps((3, 2, 3))
    value = np.zeros(calibration.shape)
    value[0] = [[2, 1, -1], [0, 4, -3]]
    calibration.add("train-source", value)
    selection = freeze_selection(calibration, limit=2)
    assert [(h["layer"], h["head"]) for h in selection] == [(1, 1), (0, 0)]
    test = SourceGaps(calibration.shape)
    test.add("test-source", -1000 * value)
    reports = build_report({"QA": HeadReadout(_probe())}, {"QA": calibration},
                           {"QA": calibration}, {"QA": test}, {"QA": selection})
    assert reports["QA"]["selected_heads"] == selection
    assert reports["QA"]["heads"][4]["selected_rank"] == 1
    assert reports["QA"]["heads"][4]["contribution_gap"]["test"]["total"]["mean_gap"] < 0
    assert freeze_selection(SourceGaps(calibration.shape)) == []


def test_parameter_norm_and_source_contrast_are_correct_without_head_averaging():
    probe = _probe()
    summaries = HeadReadout(probe).parameter_summary()
    assert len(summaries) == 6
    assert sorted(row["parameter_rank"] for row in summaries) == list(range(1, 7))
    for row in summaries:
        for name in ("current", "difference"):
            assert abs(sum(row[name]["log_fraction_contrast"].values())) < 1e-12
    standard = probe.coefficient["supervised_routing"][:48].reshape(2, 2, 3, 4)
    np.testing.assert_allclose(
        [entry["standardized_coefficient_norm"] for entry in summaries],
        np.linalg.norm(standard.transpose(1, 2, 0, 3).reshape(6, 8), axis=-1),
    )


def test_streamed_scan_contributions_keep_tasks_separate_and_read_only_transport(tmp_path, monkeypatch):
    root = _capture(tmp_path, records={
        "one": ("shared-source", "QA"), "two": ("shared-source", "QA"),
        "three": ("shared-source", "Summary"), "four": ("excluded", "QA"),
    })
    scans = ScanDataset(root)
    loaded = []
    original = scans.load

    def load(sample_id, *, fields):
        loaded.append((sample_id, fields))
        return original(sample_id, fields=fields)

    monkeypatch.setattr(scans, "load", load)
    labels = SimpleNamespace(load=lambda scan: np.array([0, 0, 1, -1]))
    readouts = {task: HeadReadout(_probe()) for task in ("QA", "Summary")}
    result = _collect(scans, labels, readouts, {"shared-source": "calibration"})
    assert [item[0] for item in loaded] == ["one", "two", "three"]
    assert all(item[1] == ("reanchor_bucket_transport",) for item in loaded)
    qa, summary = (result["calibration"][task].summary() for task in ("QA", "Summary"))
    assert qa["paired_sources"] == summary["paired_sources"] == 1
    assert qa["paired_samples"] == 2 and summary["paired_samples"] == 1
    np.testing.assert_allclose(qa["mean_gap"][0], qa["mean_gap"][1:].sum(0))


def test_unknown_labels_cannot_create_paired_source(tmp_path):
    scans = ScanDataset(_capture(tmp_path))
    labels = SimpleNamespace(load=lambda scan: np.array([0, 0, -1, -1]))
    result = _collect(scans, labels, {"QA": HeadReadout(_probe())}, {"source-a": "calibration"})
    gaps = result["calibration"]["QA"]
    assert gaps.summary()["paired_sources"] == 0
    assert freeze_selection(gaps) == []


def test_full_saved_scan_audit_freezes_selection_before_test_labels(tmp_path, monkeypatch):
    capture, analysis = tmp_path / "capture", tmp_path / "analysis"
    capture.mkdir()
    _capture(capture, records={"test": ("test-source", "QA")})
    staging = tmp_path / "staging"
    staging.mkdir()
    train_root = _capture(staging, records={
        "fit": ("fit-source", "QA"), "calibration": ("calibration-source", "QA"),
    })
    train_root.rename(capture / "train")
    train_root = capture / "train"
    manifest = json.loads((train_root / "run_manifest.json").read_text())
    manifest["config"]["split"] = "train"
    (train_root / "run_manifest.json").write_text(json.dumps(manifest))
    (analysis / "models").mkdir(parents=True)
    _probe().save(analysis / "models" / "QA_supervised_probe.npz")
    (analysis / "frozen_detector.json").write_text(json.dumps({"capture_root": str(capture)}))
    (analysis / "source_partition.json").write_text(json.dumps({
        "fit_source_ids": ["fit-source"], "calibration_source_ids": ["calibration-source"],
    }))
    constructed = []

    class Labels:
        def __init__(self, scans, **kwargs):
            constructed.append(scans.split)
            if scans.split == "test":
                assert (analysis / "head_audit" / "frozen_head_selection.json").is_file()

        def load(self, scan):
            # No calibration pairs: test must not supply replacement choices.
            return np.full(4, -1) if scan.source_id == "calibration-source" else np.array([0, 0, 1, 1])

    monkeypatch.setattr("experiments.reanchor_flow.probe_heads.ScanLabelStore", Labels)
    report = run_audit(analysis)
    assert constructed == ["train", "test"]
    assert report["tasks"]["QA"]["selected_heads"] == []
    saved = json.loads((analysis / "head_audit" / "head_report.json").read_text())
    assert saved == report
    assert saved["tasks"]["QA"]["heads"][0]["contribution_gap"]["test"]["paired_sources"] == 1
