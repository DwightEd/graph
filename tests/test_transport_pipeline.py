"""Small native-model integration; no claims about natural detection quality."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from state_audit.storage import read_arrays, read_json, write_json
from test_dynamics_pipeline import write_cohort

from experiments.native_support.transport import main

pytest_plugins = ("test_dynamics_pipeline",)


def options(output):
    return ["--output", str(output), "--device", "cpu", "--dtype", "float32",
            "--rank", "3", "--choices", "3", "--block-tokens", "3",
            "--gradient-batch", "3", "--head-batch", "2", "--cpu-threads", "1"]


def test_native_capture_to_transport_needs_no_reference_or_evidence_labels(tmp_path, tiny_model):
    write_cohort(tmp_path, ["a", "b"])
    tokenizer = SimpleNamespace(all_special_ids=[0], decode=lambda ids: str(ids[0]))
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("state_audit.model.load_model", return_value=(tiny_model, tokenizer)):
        main(["--stage", "run", *options(tmp_path)])

    directory = tmp_path / "source_transport"
    summary = read_json(directory / "summary.json")
    assert summary["scored_tokens"] == 20
    assert not summary["manual_evidence_annotations_required"]
    assert not summary["svd"] and not summary["parameter_fitting"]
    assert summary["evaluation"]["status"] == "evaluated"
    assert not (tmp_path / "state_dynamics/capture/0000/observations.npz").exists()
    scores = read_arrays(directory / "responses/0000/scores.npz")
    state = read_arrays(directory / "responses/0000/state.npz")
    assert np.isfinite(scores["risk"]).all()
    assert np.all(np.abs(scores["risk"]) <= 1.)
    assert state["inferred_budget"].shape[1:3] == (3, 4)
    np.testing.assert_array_equal(state["future_observed"][-2:], False)
    sources = read_json(directory / "responses/0000/sources.json")
    assert all(block["semantic_type"] == "unassigned" for block in sources["blocks"])

    labels = read_json(tmp_path / "annotations.json")
    labels["a"]["labels"] = [1 - label for label in labels["a"]["labels"]]
    write_json(tmp_path / "annotations.json", labels)
    with patch("state_audit.model.load_model", side_effect=AssertionError("score loaded LLM")), \
         patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("score loaded tokenizer")):
        main(["--stage", "score", *options(tmp_path)])
    changed = read_arrays(directory / "responses/0000/scores.npz")
    for name in scores:
        np.testing.assert_array_equal(changed[name], scores[name])

    with patch("state_audit.model.load_model", side_effect=AssertionError("resume loaded LLM")), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        main(["--stage", "run", "--resume", *options(tmp_path)])
    resumed = read_arrays(directory / "responses/0000/scores.npz")
    np.testing.assert_array_equal(resumed["risk"], scores["risk"])
    with patch("experiments.native_support.transport.build_graph", side_effect=AssertionError("evaluate built graph")):
        main(["--stage", "evaluate", *options(tmp_path)])


def test_label_free_scoring_and_zero_strength_recover_raw_route(tmp_path, tiny_model):
    write_cohort(tmp_path, ["a"])
    (tmp_path / "annotations.json").unlink()
    tokenizer = SimpleNamespace(all_special_ids=[0], decode=lambda ids: str(ids[0]))
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("state_audit.model.load_model", return_value=(tiny_model, tokenizer)):
        main(["--stage", "run", "--strength", "0", *options(tmp_path)])
    directory = tmp_path / "source_transport"
    scores = read_arrays(directory / "responses/0000/scores.npz")
    # Regrouping native float32 edge norms changes summation roundoff only.
    np.testing.assert_allclose(scores["transport_route"], scores["raw_route"], atol=1e-7)
    summary = read_json(directory / "summary.json")
    assert summary["evaluation"]["status"] == "unavailable"
    assert (directory / "report.html").is_file()
