"""Resolve existing labels for saved results, without rerunning graph analysis."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.unsupervised_token_graph.reanchor_evaluate import (
    evaluate, main, resolve_annotations,
)


@pytest.fixture
def interrupted(tmp_path):
    dataset = tmp_path / "RAGTruth"
    cache = dataset / "attention/llama31_8b/train"
    cache.mkdir(parents=True)
    output = tmp_path / "saved output"
    output.mkdir()
    (output / "settings.json").write_text(json.dumps(dict(
        version="source-carrier-information-v1", labels_read=False,
        cache=str(cache), annotations=None,
    )))
    labels = []
    for i in range(2):
        record = dict(id=str(i), source_id=f"s{i}", split="train",
                      task="QA", generator="fixture", file=f"samples/attention_{i}.npz",
                      response_sha256=hashlib.sha256(b"abc").hexdigest())
        path = output / record["file"]
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            path, record_json=json.dumps(record), cache_stamp=[1, 2],
            source_mismatch_bits=[[.8, .2, .1]],
            permuted_source_mismatch_bits=[[.2, .8, .1]],
            event_strength=[[.8, .2, .1]], prompt_reach=[[.5, .7, .8]],
            unknown_mass=[[.1, .1, .1]], query_positions=[2, 3, 4],
            prediction_positions=[3, 4, 5], prompt_length=2, total_tokens=5,
            offsets=[[0, 1], [1, 2], [2, 3]], cache_format="canonical_csr",
        )
        labels.append(dict(id=str(i), source_id=f"s{i}", split="train",
                           response="abc", labels=[dict(start=1, end=2)] if i == 0 else []))
    annotations = dataset / "response.jsonl"
    annotations.write_text("\n".join(map(json.dumps, labels)))
    (output / "samples/unfinished.partial").write_bytes(b"unfinished")
    return output, cache, annotations


def arguments(output):
    return ["--predictions", str(output), "--output", str(output / "evaluation_partial.json"),
            "--split", "train", "--bootstrap", "0", "--completed-only"]


def update_settings(output, **values):
    path = output / "settings.json"
    saved = json.loads(path.read_text())
    saved.update(values)
    path.write_text(json.dumps(saved))


@pytest.mark.parametrize("single_file", [False, True])
def test_cache_ancestor_discovery(interrupted, single_file):
    output, cache, annotations = interrupted
    if single_file:
        path = cache / "attention_0.npz"
        path.touch()
        update_settings(output, cache=str(path))
    assert resolve_annotations(output) == str(annotations)


def test_explicit_path_precedes_saved_and_discovered_paths(interrupted, tmp_path):
    output, _, _ = interrupted
    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text("")
    update_settings(output, annotations=str(tmp_path / "missing_saved.jsonl"))
    assert resolve_annotations(output, explicit) == str(explicit)


def test_saved_path_precedes_discovery(interrupted, tmp_path):
    output, _, _ = interrupted
    saved = tmp_path / "saved.jsonl"
    saved.write_text("")
    update_settings(output, annotations=str(saved))
    assert resolve_annotations(output) == str(saved)


@pytest.mark.parametrize("explicit", [False, True])
def test_configured_missing_file_never_silently_switches_dataset(interrupted, tmp_path, explicit):
    output, _, _ = interrupted
    missing = tmp_path / "absent.jsonl"
    if not explicit:
        update_settings(output, annotations=str(missing))
    with pytest.raises(FileNotFoundError, match="annotation file not found"):
        resolve_annotations(output, missing if explicit else None)


def test_discovery_does_not_read_labels_or_modify_settings(interrupted, monkeypatch):
    output, _, annotations = interrupted
    before = (output / "settings.json").read_bytes()
    original = Path.open

    def forbid_annotations(path, *args, **kwargs):
        if path == annotations:
            raise AssertionError("path discovery must not read labels")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_annotations)
    assert resolve_annotations(output) == str(annotations)
    assert (output / "settings.json").read_bytes() == before


def test_missing_annotations_still_requires_a_real_file(interrupted, capsys):
    output, _, annotations = interrupted
    annotations.unlink()
    assert resolve_annotations(output) is None
    with pytest.raises(SystemExit) as error:
        main(arguments(output))
    assert error.value.code == 2
    assert "Do not rerun analysis" in capsys.readouterr().err
    assert not (output / "evaluation_partial.json").exists()


def test_if_available_can_skip_without_claiming_success(interrupted, capsys):
    output, _, annotations = interrupted
    annotations.unlink()
    main(arguments(output) + ["--if-available"])
    assert "Evaluation skipped" in capsys.readouterr().out
    assert not (output / "evaluation_partial.json").exists()


@pytest.mark.parametrize("shell", [False, True])
def test_preview_without_population_reuses_saved_npz_only(interrupted, shell):
    output, _, annotations = interrupted
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in output.rglob("*") if p.is_file()}
    expected = evaluate(output, annotations, split="train", bootstrap=0, completed_only=True)
    repo = Path(__file__).resolve().parents[1]
    if shell:
        env = os.environ.copy()
        env.update(PY=sys.executable, OUTPUT=str(output), SPLIT="train",
                   ANNOTATIONS="", BOOTSTRAP="0", COMPLETED_ONLY="0")
        command = ["bash", str(repo / "experiments/unsupervised_token_graph/evaluate.sh"),
                   "--completed-only"]
    else:
        env = os.environ.copy()
        command = [sys.executable, "-m", "experiments.unsupervised_token_graph.run",
                   "evaluate", *arguments(output)]
    result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert f"Annotations: {annotations}" in result.stdout
    report = json.loads((output / "evaluation_partial.json").read_text())
    assert report["groups"] == expected["groups"]
    assert report["evaluation_scope"] == "completed_samples_preview"
    assert report["evaluated_responses"] == 2
    assert not (output / "complete.json").exists()
    assert not (output / "summary.json").exists()
    assert not (output / "evaluation.json").exists()
    for path, state in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == state


def test_discovery_does_not_bypass_identity_checks(interrupted):
    output, _, annotations = interrupted
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    rows[0]["response"] = "abd"
    annotations.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="identity mismatch"):
        main(arguments(output))
