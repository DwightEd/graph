"""Audit positions, source grouping, and interrupted target capture contracts."""

import argparse
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from state_audit.storage import read_json, write_json

from experiments.short_span_audit import capture
from experiments.short_span_audit.capture_reporting import (
    load_targets,
    measured_pairs,
    normal_history,
    source_means,
    write_report,
)


def answer_and_entry():
    answer = SimpleNamespace(
        response_id="7",
        source_id="s",
        task="QA",
        generator="generator",
        split="test",
        prompt_length=2,
        token_ids=np.arange(6),
        response_ids=np.arange(2, 6),
        offsets=np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]),
        text="abcd",
    )
    record = {
        "id": "7",
        "source_id": "s",
        "task": "QA",
        "generator": "generator",
        "split": "test",
        "prompt_length": 2,
    }
    targets = [
        {
            "target": i,
            "side": "error" if i < 2 else "normal",
            "pair_id": "p",
            "span_start": 0 if i < 2 else 2,
            "span_end": 2 if i < 2 else 4,
            "offset": i % 2,
        }
        for i in range(4)
    ]
    entry = {
        "record": record,
        "tokens": 4,
        "offsets": answer.offsets.tolist(),
        "text": "abcd",
        "targets": targets,
        "gold": [[0, 2]],
    }
    return answer, entry


def fake_attribution(model, token_ids, prompt_length, target, **kwargs):
    keys = prompt_length + target
    attention = np.full((1, 2, keys), 1 / keys)
    contribution = attention * np.where(np.arange(keys) % 2, -1, 1)
    return {
        "attention": attention,
        "contribution": contribution,
        "value_energy": attention**2,
        "ordinary_keys": np.arange(keys) != 0,
        "layers": np.array([8]),
        "query": np.array(keys - 1),
        "target": np.array(target),
        "target_id": np.array(token_ids[keys]),
        "key_positions": np.arange(keys),
        "logit_entropy": np.array(0.4 + target),
        "target_logp": np.array(-0.5),
        "margin": np.array(0.2),
    }


def test_source_groups_exclude_unseen_target_and_keep_self():
    assert capture.source_groups(3, 0, 2).tolist() == [0, 0, 0]
    assert capture.source_groups(3, 4, 2).tolist() == [0, 0, 0, 2, 2, 1, 1]
    assert capture.source_groups(3, 1, 10).tolist() == [0, 0, 0, 1]


def test_target_plan_deduplicates_computation_not_memberships():
    _, entry = answer_and_entry()
    entry["targets"].append(dict(entry["targets"][0], pair_id="another"))
    plan = capture.target_plan(entry)
    assert list(plan) == [0, 1, 2, 3]
    assert len(plan[0]) == 2


def test_cohort_alignment_rejects_changed_offsets_and_prompt():
    answer, entry = answer_and_entry()
    capture.verify_cohort_answer(answer, entry)
    entry["offsets"][0] = [0, 2]
    with pytest.raises(ValueError, match="frozen short-span cohort"):
        capture.verify_cohort_answer(answer, entry)


def test_observer_tokenizer_is_checked_even_with_saved_offsets(monkeypatch):
    answer, _ = answer_and_entry()
    wrong_offsets = answer.offsets.copy()
    wrong_offsets[0] = [0, 2]
    monkeypatch.setattr(capture, "verified_offsets", lambda *args: wrong_offsets)
    with pytest.raises(ValueError, match="observer tokenizer differs"):
        capture.verify_model_alignment(answer, object())


def test_capture_arguments_reject_empty_or_duplicate_layers():
    for value in ("8:8", "8,8", "-1,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            capture.parse_layers(value)
    assert capture.parse_layers("8:10") == [8, 9]
    assert capture.nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        capture.positive_int("0")


def test_raw_edges_and_excluded_specials_survive_aggregation(monkeypatch):
    answer, _ = answer_and_entry()
    monkeypatch.setattr(capture, "capture_target_attribution", fake_attribution)
    arrays = capture.target_arrays(None, answer, 2, [8], [0], 1)
    assert arrays["value_energy"].shape == (1, 2, 4)
    assert arrays["source_value_energy"].shape == (1, 2, 3)
    np.testing.assert_allclose(arrays["source_route_mass"].sum(-1), 0.75)
    np.testing.assert_allclose(arrays["excluded_special_route_mass"], 0.25)
    np.testing.assert_allclose(arrays["source_contribution_negative"].sum(-1), 0.5)


def test_nonfinite_derivative_cannot_be_saved_as_completed_target(
    tmp_path, monkeypatch
):
    answer, entry = answer_and_entry()

    def nonfinite(*args, **kwargs):
        arrays = fake_attribution(*args, **kwargs)
        arrays["contribution"][0, 0, 0] = np.nan
        return arrays

    monkeypatch.setattr(capture, "capture_target_attribution", nonfinite)
    settings = {"layers": [8], "local_window": 1, "model": "observer"}
    with pytest.raises(FloatingPointError, match="7 target 0: nonfinite"):
        capture.capture_answer(None, answer, entry, tmp_path, settings, [0])
    directory = capture.answer_directory(tmp_path, entry)
    assert not (directory / "000000.npz").exists()
    assert not read_json(directory / "answer.json")["complete"]


def test_interrupted_capture_resumes_targets_and_preserves_identity(
    tmp_path, monkeypatch
):
    answer, entry = answer_and_entry()
    calls = []

    def interrupted(*args, **kwargs):
        target = args[3]
        calls.append(target)
        if target == 1:
            raise RuntimeError("interrupted")
        return fake_attribution(*args, **kwargs)

    settings = {"layers": [8], "local_window": 1, "model": "observer"}
    monkeypatch.setattr(capture, "capture_target_attribution", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        capture.capture_answer(None, answer, entry, tmp_path, settings, [0])
    directory = capture.answer_directory(tmp_path, entry)
    assert not read_json(directory / "answer.json")["complete"]
    assert (directory / "000000.npz").exists()
    monkeypatch.setattr(capture, "capture_target_attribution", fake_attribution)
    capture.capture_answer(None, answer, entry, tmp_path, settings, [0])
    assert read_json(directory / "answer.json")["complete"]
    assert calls == [0, 1]


def test_partial_pair_does_not_become_full_span_result():
    _, entry = answer_and_entry()
    values = {
        target: {
            "sources": np.full((4, 1, 2, 3), target),
            "readouts": np.full(3, target),
        }
        for target in (0, 2, 3)
    }
    pairs = measured_pairs(entry, values)
    assert len(pairs) == 1
    assert pairs[0]["phase"] == "onset"
    assert pairs[0]["normal_history"] == "recovery"
    np.testing.assert_allclose(pairs[0]["error"]["readouts"], 0)
    np.testing.assert_allclose(pairs[0]["normal"]["readouts"], 2)


def test_normal_history_is_the_same_fifteen_step_half_open_window_as_cpu_audit():
    assert normal_history([[4, 5]], 20) == "clean_history"
    assert normal_history([[4, 6]], 20) == "recovery"
    assert normal_history([[20, 22]], 20) == "clean_history"
    assert normal_history([[0, 1]], 0) == "clean_history"


@pytest.mark.parametrize("field", ["source_route_mass", "logit_entropy"])
def test_report_rejects_nonfinite_saved_targets(tmp_path, monkeypatch, field):
    answer, entry = answer_and_entry()
    settings = {"layers": [8], "local_window": 1, "model": "observer"}
    monkeypatch.setattr(capture, "capture_target_attribution", fake_attribution)
    capture.capture_answer(None, answer, entry, tmp_path, settings, [0])
    directory = capture.answer_directory(tmp_path, entry)
    path = directory / "000000.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays[field][...] = np.nan
    np.savez_compressed(path, **arrays)
    with pytest.raises(FloatingPointError, match="7 target 0: nonfinite saved"):
        load_targets(directory, entry)


def test_source_average_does_not_overweight_repeated_answers():
    pairs = [
        {
            "source_id": source,
            "error": {"readouts": np.array([value])},
            "normal": {"readouts": np.array([0.0])},
        }
        for source, value in (("s1", 0), ("s1", 2), ("s2", 9))
    ]
    means, count = source_means(pairs, "readouts")
    assert count == 2
    np.testing.assert_allclose(means["error"], [5.0])


def test_report_distinguishes_pending_targets_and_mechanism_audit(
    tmp_path, monkeypatch
):
    answer, entry = answer_and_entry()
    settings = {
        "layers": [8],
        "local_window": 1,
        "model": "observer",
        "source_names": capture.SOURCE_NAMES,
        "purpose": "label_assisted_short_span_audit_not_detector_evaluation",
    }
    write_json(tmp_path / "settings.json", settings)
    pending = write_report(tmp_path, [entry])
    assert pending["completed_targets"] == 0
    assert pending["expected_targets"] == 4
    monkeypatch.setattr(capture, "capture_target_attribution", fake_attribution)
    capture.capture_answer(None, answer, entry, tmp_path, settings, [0])
    summary = write_report(tmp_path, [entry])
    assert summary["completed_answers"] == 1
    assert summary["completed_pair_phases"] == 2
    assert summary["detector_evaluation"] is False
    records = read_json(tmp_path / "paired_records.json")
    assert {row["normal_history"] for row in records} == {"recovery"}
    with np.load(tmp_path / "paired_measurements.npz", allow_pickle=False) as saved:
        assert saved["error_sources"].shape == (2, 4, 1, 2, 3)
    with tarfile.open(summary["archive"], "r:gz") as archive:
        names = archive.getnames()
    assert "answers/test/QA/7/answer.json" in names
    assert "paired_measurements.npz" in names
    assert not any(name.endswith("000000.npz") for name in names)
