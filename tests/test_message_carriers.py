"""Native gate semantics, source-world alignment, representation axes and label isolation."""

from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from state_audit.intervention import intervene
from state_audit.model.replay import attention_backend
from state_audit.operations import Delete, Target
from state_audit.storage import read_arrays, read_json, write_arrays, write_json
from experiments.native_support.choice_cache import CaptureReader
from experiments.native_support.evidence_contrast.views import prepare_views
from experiments.native_support.message_carriers.capture import capture_unit, deleted_logp
from experiments.native_support.message_carriers.data import prepare
from experiments.native_support.message_carriers.representation import (
    GLOBAL_CHANNELS, HEAD_CHANNELS, METHODS, feature_schema, unit_vectors)
from experiments.native_support.message_carriers.run import arguments, main
from experiments.native_support.message_carriers.selection import measure_candidates, select_carriers
from test_evidence_contrast import make_input, tiny_model


def dense_deleted_logp(model, prompt, answer, unit, edges):
    queries = tuple(range(len(prompt) + unit["start"] - 1, len(prompt) + unit["stop"] - 1))
    operations = [Delete(Target("attention", (layer,), positions=queries,
                                heads=(head,), keys=(len(prompt) + key,)))
                  for layer, head, key in edges]
    with torch.no_grad(), attention_backend(model, "eager"), intervene(model, operations):
        hidden = model.forward(prompt + answer[:unit["stop"] - 1])
        targets = torch.tensor(answer[unit["start"]:unit["stop"]])
        return model.score(hidden[list(queries)], targets)[0].numpy()


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_sparse_cuts_equal_native_dense_edge_deletions(family):
    model = tiny_model(family)
    prompt, answer = [1, 3, 5, 7, 9], [11, 13, 15, 17, 19, 21, 23]
    unit = dict(start=3, stop=6)
    edges = [(0, 0, 1), (1, 1, 2)]
    actual = deleted_logp(model, prompt, answer, unit, edges)
    np.testing.assert_allclose(actual, dense_deleted_logp(model, prompt, answer, unit, edges), atol=8e-7)
    np.testing.assert_allclose(deleted_logp(model, prompt, answer, unit, edges, strength=0),
                               deleted_logp(model, prompt, answer, unit), atol=1e-7)
    # Restored backend/hooks and unchanged parameter gradient state.
    assert model.native.config._attn_implementation == "eager"
    assert all(parameter.grad is None for parameter in model.native.parameters())
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.native.modules())


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_unit_gate_gradient_matches_centered_finite_difference(family):
    model = tiny_model(family)
    answer, unit = [11, 13, 15, 17, 19, 21], dict(start=3, stop=6)
    for prompt in ([1, 3, 5, 7, 9], [1, 7, 9]):
        observed = measure_candidates(model, prompt, answer, unit)
        edge = np.unravel_index(np.abs(observed["gradient"]).argmax(), observed["gradient"].shape)
        epsilon = .02
        plus = deleted_logp(model, prompt, answer, unit, [edge], strength=-epsilon)
        minus = deleted_logp(model, prompt, answer, unit, [edge], strength=epsilon)
        finite = np.mean(plus.astype(float) - minus) / (2 * epsilon)
        assert finite == pytest.approx(observed["gradient"][edge], abs=2e-5, rel=.03)
        np.testing.assert_allclose(observed["logp"], deleted_logp(model, prompt, answer, unit), atol=8e-7)
        assert observed["reconstruction_error"].max() < 1e-6


def test_selection_uses_both_worlds_keeps_signs_and_matches_random_heads():
    first = np.array([[[1., -3., 0.], [0., 0., 0.]], [[2., 0., 0.], [0., 0., 0.]]])
    second = first.copy()
    second[1, 1, 2] = -5
    observations = [dict(gradient=value, eligible=np.ones_like(value, dtype=bool)) for value in (first, second)]
    selected = select_carriers(observations, 2, 37)
    np.testing.assert_array_equal(selected["edges"], [[1, 1, 2], [0, 0, 1]])
    np.testing.assert_array_equal(selected["approximation"], [[0, -5], [-3, -3]])
    np.testing.assert_array_equal(selected["edges"][:, :2], selected["sham_edges"][:, :2])
    assert (selected["edges"][:, 2] != selected["sham_edges"][:, 2]).all()
    np.testing.assert_array_equal(selected["sham_edges"], select_carriers(observations, 2, 37)["sham_edges"])


def example_views():
    return dict(prompt_with_source=[1, 3, 5, 7, 9], prompt_without_source=[1, 7, 9],
                answer_ids=[11, 13, 15, 17, 19, 21, 23],
                units=[dict(start=0, stop=3), dict(start=3, stop=7)])


def test_vectors_preserve_physical_heads_shared_keys_and_conditional_identity():
    model, views = tiny_model(), example_views()
    saved = capture_unit(model, views, views["units"][1], 2, 37)
    vector, history = unit_vectors(saved)
    schema = feature_schema(2, 2)
    assert vector.shape == (4, schema["dimension"])
    heads = vector[:, len(GLOBAL_CHANNELS):].reshape(4, 2, 2, len(HEAD_CHANNELS))
    assert (saved["edges"][:, 2] < 3).all()
    for index, (layer, head, key) in enumerate(saved["edges"]):
        np.testing.assert_array_equal(history[:, layer, head], key)
        expected = saved["keep"].astype(float) - saved["single"][:, index]
        np.testing.assert_allclose(heads[:, layer, head, :2], expected.T, atol=1e-7)
        np.testing.assert_allclose(heads[:, layer, head, 2], expected[0] - expected[1], atol=1e-7)
    assert (heads[..., 3].sum((1, 2)) == 2).all()
    np.testing.assert_array_equal(history == -1, heads[..., 3] == 0)
    columns = {name: vector[:, i] for i, name in enumerate(GLOBAL_CHANNELS)}
    np.testing.assert_allclose(columns["source_gain_after_joint_delete"],
        columns["source_gain"] - columns["joint_interaction"], atol=1e-7)
    # Two source worlds use the same answer key identity despite different prompt lengths.
    for condition, name in enumerate(("with_source", "without_source")):
        prompt_length = len(views[f"prompt_{name}"])
        np.testing.assert_array_equal(saved["edge_keys"][condition], prompt_length + saved["edges"][:, 2])
        np.testing.assert_array_equal(saved["queries"][condition], prompt_length + saved["target"] - 1)
        expected = dense_deleted_logp(model, views[f"prompt_{name}"], views["answer_ids"],
                                     views["units"][1], saved["edges"])
        np.testing.assert_allclose(saved["joint"][condition], expected, atol=8e-7)


def test_no_history_and_no_source_are_explicit_identity_interventions():
    model, views = tiny_model(), example_views()
    first = capture_unit(model, views, views["units"][0], 2, 37)
    assert first["edges"].shape == (0, 3)
    assert first["single"].shape == (2, 0, 3)
    np.testing.assert_array_equal(first["keep"], first["joint"])
    vector, history = unit_vectors(first)
    assert (history == -1).all()
    assert (vector[:, len(GLOBAL_CHANNELS):] == 0).all()
    views["prompt_without_source"] = views["prompt_with_source"]
    saved = capture_unit(model, views, views["units"][1], 2, 37)
    for name in ("keep", "single", "joint", "sham"):
        np.testing.assert_array_equal(saved[name][0], saved[name][1])


def make_contrast_input(source):
    pieces, annotations = make_input(source)
    settings = read_json(source / "settings.json")
    write_json(source / "protocol.json", dict(version="test-contrast"))
    for index, response in enumerate(settings["responses"]):
        directory = source / "responses" / f"{index:04d}"
        views = prepare_views(response, read_json(directory / "sources.json"), 64)
        write_json(directory / "views.json", views)
        original = read_arrays(directory / "scores.npz")
        extra = {name: np.linspace(0, .1, 5) for name in (
            "source_full", "source_local", "source_pair", "surprisal_full",
            "raw_route_offline_mean", "observable_route_offline_mean")}
        write_arrays(directory / "scores.npz", **original, **extra, risk=original["raw_route"])
    return pieces, annotations


def test_pipeline_zip_resume_label_isolation_and_baseline_regression(tmp_path):
    source, output = tmp_path / "input", tmp_path / "carriers"
    pieces, annotations = make_contrast_input(source)
    archive = tmp_path / "contrast.zip"
    with ZipFile(archive, "w") as bundle:
        for file in source.rglob("*"):
            if file.is_file():
                bundle.write(file, file.relative_to(source))
    class Tokenizer:
        def decode(self, ids):
            return pieces[ids[0]]
    args = ["--input", str(archive), "--output", str(output), "--device", "cpu", "--dtype", "float32", "--top-k", "2"]
    original_json = CaptureReader.json
    def observations_only(reader, name):
        assert name != "annotations.json", "Measurement accessed labels"
        return original_json(reader, name)
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch.object(CaptureReader, "json", observations_only):
        main([*args, "--stage", "capture"])
    assert not (output / "annotations.json").exists()
    main(["--output", str(output), "--stage", "score"])
    nodes = read_arrays(output / "responses/0000/node_features.npz")
    scores = read_arrays(output / "responses/0000/scores.npz")
    assert nodes["node_features"].shape == (5, 27)
    assert read_json(output / "coverage.json")["represented_tokens"] == 10
    assert len(read_json(output / "evaluation.json")["methods"]) == len(METHODS)
    for name, value in read_arrays(output / "responses/0000/baselines.npz").items():
        np.testing.assert_array_equal(scores[name], value)
    with ZipFile(output.with_name("carriers_review.zip")) as bundle:
        assert {"feature_schema.json", "intervention_diagnostics.csv", "within_unit.json",
                "intervened_tokens_evaluation.json", "responses/0000/node_features.npz"} <= set(bundle.namelist())
    for row in annotations.values():
        row["labels"] = [1 - value for value in row["labels"]]
    changed_labels = tmp_path / "other_annotations.json"
    write_json(changed_labels, annotations)
    with patch("state_audit.model.load_model", side_effect=AssertionError("Resume loaded model")):
        main([*args, "--resume", "--annotations", str(changed_labels)])
    repeated = read_arrays(output / "responses/0000/node_features.npz")
    for name in nodes:
        np.testing.assert_array_equal(nodes[name], repeated[name])
    for name, value in read_arrays(output / "responses/0000/scores.npz").items():
        np.testing.assert_array_equal(scores[name], value)
    with pytest.raises(ValueError, match="settings changed"):
        prepare(arguments([*args, "--resume", "--top-k", "3"]))


def test_interrupted_unit_resumes_and_future_suffix_cannot_change_measurement(tmp_path):
    source, output = tmp_path / "input", tmp_path / "carriers"
    pieces, _ = make_contrast_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces[ids[0]]
    args = ["--input", str(source), "--output", str(output), "--device", "cpu", "--dtype", "float32", "--stage", "capture", "--top-k", "1"]
    def interrupted(model, views, unit, top_k, seed):
        if unit["start"] > 0:
            raise RuntimeError("interrupted")
        return capture_unit(model, views, unit, top_k, seed)
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch("experiments.native_support.message_carriers.capture.capture_unit", side_effect=interrupted):
        with pytest.raises(RuntimeError, match="interrupted"):
            main(args)
    completed = output / "responses/0000/unit_000000/measurements.npz"
    original = completed.read_bytes()
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())):
        main([*args, "--resume"])
    assert completed.read_bytes() == original
    model, views = tiny_model(), example_views()
    unit = dict(start=3, stop=5)
    before = capture_unit(model, views, unit, 2, 37)
    views["answer_ids"][5:] = [25, 27]
    after = capture_unit(model, views, unit, 2, 37)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
