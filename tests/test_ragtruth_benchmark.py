"""Compact-native equivalence, full population coverage, and source-disjoint selection."""

import json
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from state_audit.attribution import _projection_gram
from state_audit.model.adapter import ModelAdapter
from state_audit.native_forward import iter_forward_traces
from state_audit.storage import read_arrays, read_json, write_arrays, write_json
from experiments.native_support.evidence_contrast.capture import collect_condition
from experiments.native_support.evidence_contrast.views import prepare_views
from experiments.native_support.ragtruth_benchmark.capture import collect_with_source
from experiments.native_support.ragtruth_benchmark.data import TASKS
from experiments.native_support.ragtruth_benchmark.run import arguments, main
from experiments.native_support.ragtruth_benchmark.selection import development_sources, choose_readout, SELECTION_METHODS
from experiments.native_support.routes import route_scores
from test_evidence_contrast import tiny_model, make_input


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_compact_routing_matches_original_head_message_formula_and_native_likelihood(family):
    model = tiny_model(family)
    prefix, answer = [1, 3, 5, 7], [9, 11, 13, 15, 17]
    mask = np.array([False, True, True, True])  # Includes predictor self: exclude only at t=0.
    response = dict(answer_ids=answer, units=[dict(start=0, stop=2), dict(start=2, stop=5)])
    source = dict(prompt_with_source=prefix, source_mask=mask)
    grams = {layer: _projection_gram(model, layer) for layer in range(len(model.layers))}
    scores, heads = collect_with_source(model, source, response, grams, 2, 2)
    reference = collect_condition(model, prefix, answer, response["units"], 2, 2, "reference")
    for name in ("full", "local"):
        np.testing.assert_allclose(scores[name], reference[name], atol=1e-6)
    original = list(iter_forward_traces(model, prefix + answer, len(prefix), range(len(answer)), [],
                                        prefill_chunk_size=2, query_chunk_size=2))
    for index, arrays in enumerate(original):
        expected = route_scores(arrays["attention"], np.sqrt(arrays["edge_value_energy"]),
                                arrays["group_ids"], len(prefix), mask)
        assert scores["raw_route"][index] == pytest.approx(expected["routing_imbalance"], abs=2e-7)
        assert scores["raw_attention"][index] == pytest.approx(expected["attention_displacement"], abs=2e-7)
    assert heads["norm_total"].shape == (5, 2, 2)
    assert all(parameter.grad is None for parameter in model.native.parameters())
    changed = {**response, "answer_ids": answer[:2] + [19, 21, 23]}
    repeated, _ = collect_with_source(model, source, changed, grams, 2, 2)
    for name in ("raw_route", "raw_attention", "full", "local"):
        np.testing.assert_array_equal(scores[name][:2], repeated[name][:2])


class CharacterTokenizer:
    all_special_ids = [1, 2]
    added_tokens_decoder = {}

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(char) + 3 for char in text],
                    offset_mapping=[(index, index + 1) for index in range(len(text))])

    def decode(self, ids):
        return "".join(chr(token - 3) for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "[" + messages[0]["content"] + "]"


def official_fixture(directory):
    directory.mkdir()
    sources, responses = [], []
    for task in TASKS:
        for number in range(6):
            source_id = f"{task}-{number}"
            source = dict(source_id=source_id, task_type=task, source_info="Facts.", prompt="Summarize: Facts.")
            if task == "QA":
                source.update(prompt="Q\npassages:\nFacts.\noutput:", source_info=dict(question="Q", passages="Facts."))
            if task == "Data2txt":
                source.update(prompt="Describe: {'name': 'X'}", source_info=dict(name="X"))
            sources.append(source)
            for generator in ("generator-a", "generator-b"):
                labels = [dict(start=3, end=4, text="B")] if generator.endswith("b") else []
                responses.append(dict(id=f"{source_id}-{generator}", source_id=source_id,
                    split="test" if number == 5 else "train", model=generator,
                    response="A. B.", labels=labels, quality="good"))
    for name, rows in (("source_info", sources), ("response", responses)):
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return responses


def test_full_official_pipeline_all_tasks_generators_splits_and_train_only_selection(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    rows = official_fixture(dataset)
    model = ModelAdapter(LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=512)))
    args = ["--dataset", str(dataset), "--output", str(output), "--model", "tiny",
            "--device", "cpu", "--dtype", "float32", "--query-chunk-size", "2",
            "--prefill-chunk-size", "8", "--cpu-threads", "1", "--select-on-train"]
    from experiments.native_support.ragtruth_benchmark import report, selection
    original_evaluate = report.annotations
    def check_evaluate(root, manifest, records):
        assert (root / "coverage.json").is_file()
        assert all((root / record["directory"] / "scores.npz").is_file() for record in manifest["records"])
        return original_evaluate(root, manifest, records)
    def check_select(root, manifest, records):
        assert all(record["split"] == "train" for record in records)
        return check_evaluate(root, manifest, records)
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()), \
            patch("state_audit.model.load_model", return_value=(model, CharacterTokenizer())), \
            patch.object(report, "annotations", check_evaluate), patch.object(selection, "annotations", check_select):
        main(args)
    manifest = read_json(output / "manifest.json")
    assert len(manifest["records"]) == 36
    assert len(manifest["groups"]) == 12
    assert read_json(output / "coverage.json")["scored_tokens"] == 180
    summary = read_json(output / "summary.json")
    assert len(summary["test_by_dataset"]) == 6
    assert {row["task"] for row in summary["test_by_dataset"]} == set(TASKS)
    original_choices = read_json(output / "selection.json")
    original_scores = read_arrays(output / manifest["records"][0]["directory"] / "scores.npz")
    for row in rows:
        if row["split"] == "test":
            row["labels"] = [] if row["labels"] else [dict(start=3, end=4, text="B")]
    (dataset / "response.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with patch("state_audit.model.load_model", side_effect=AssertionError("Resume repeated model capture")), \
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("Resume repeated tokenization")):
        main([*args, "--resume"])
    assert read_json(output / "selection.json") == original_choices
    repeated = read_arrays(output / manifest["records"][0]["directory"] / "scores.npz")
    for name, values in original_scores.items():
        np.testing.assert_array_equal(values, repeated[name])
    with ZipFile(output.with_name("output_review_light.zip")) as bundle:
        assert {"metrics_by_dataset.csv", "metrics.csv", "selection.json", "manifest.json", "datasets.png"} <= set(bundle.namelist())
        assert not any(name.startswith("responses/") for name in bundle.namelist())
    with pytest.raises(ValueError, match="settings changed"):
        main([*args, "--resume", "--window", "32"])


def test_official_preparation_never_needs_labels_and_default_has_no_pilot_cap(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    rows = official_fixture(dataset)
    for row in rows:
        del row["labels"]
    (dataset / "response.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = ["--dataset", str(dataset), "--output", str(output), "--stage", "prepare"]
    assert arguments(args).limit is None
    assert arguments(args).generators is None
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()):
        main(args)
    assert len(read_json(output / "manifest.json")["records"]) == 36
    assert not (output / "evaluation.json").exists()


def test_parameter_selection_rejects_overlapping_sources_and_test_only_cache():
    overlap = [dict(task="QA", split=split, source_id="same") for split in ("train", "test")]
    with pytest.raises(ValueError, match="overlap"):
        development_sources(overlap, "QA")
    with pytest.raises(ValueError, match="train sources"):
        development_sources([dict(task="QA", split="test", source_id="x")], "QA")


def test_development_can_select_a_stronger_route_baseline_instead_of_forcing_local_source(tmp_path):
    records = [dict(id=str(index), source_id=f"source{index}", task="Summary", split="train",
                    directory=f"responses/{index}", tokens=2) for index in range(2)]
    for record in records:
        values = {name: np.array([1., 0.]) for name in SELECTION_METHODS}
        values["raw_route"] = np.array([0., 1.])
        write_arrays(tmp_path / record["directory"] / "scores.npz", token_id=np.array([5, 6]), **values)
    truth = {record["id"]: dict(token_ids=[5, 6], labels=[0, 1]) for record in records}
    with patch("experiments.native_support.ragtruth_benchmark.selection.annotations", return_value=truth):
        selected = choose_readout(tmp_path, dict(records=records), "Summary")
    assert selected["method"] == "raw_route"
    assert selected["weight"] is None


def test_cache_scoring_preserves_baseline_and_reads_no_labels_before_coverage(tmp_path):
    source, output = tmp_path / "cache", tmp_path / "output"
    make_input(source)
    settings = read_json(source / "settings.json")
    settings["cohort"] = dict(task="QA", split="test", generator="fixture", labels_used_for_selection=False)
    write_json(source / "settings.json", settings)
    for index, response in enumerate(settings["responses"]):
        directory = source / "responses" / f"{index:04d}"
        views = prepare_views(response, read_json(directory / "sources.json"), 64)
        write_json(directory / "views.json", views)
        values = read_arrays(directory / "scores.npz")
        values.update(source_local=np.array([0., 0., 1., 1., 1.]), source_full=np.arange(5.))
        write_arrays(directory / "scores.npz", **values)
    archive = tmp_path / "cache.zip"
    with ZipFile(archive, "w") as bundle:
        for path in source.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(source))
    original_archive = archive.read_bytes()
    from experiments.native_support.choice_cache import CaptureReader
    original_read = CaptureReader.json
    def check(reader, name):
        if name == "annotations.json":
            assert (output / "coverage.json").is_file()
        return original_read(reader, name)
    with patch.object(CaptureReader, "json", check), \
            patch("state_audit.model.load_model", side_effect=AssertionError("Cache stage loaded model")):
        main(["--cache-input", str(archive), "--output", str(output), "--stage", "score"])
        assert not (output / "evaluation.json").exists()
        main(["--output", str(output), "--stage", "evaluate"])
    metrics = read_json(output / "evaluation.json")["groups"][0]["methods"]
    assert metrics["local_route_0"]["all_error"]["auroc"] == metrics["source_local_unit_mean"]["all_error"]["auroc"]
    assert metrics["local_route_0"]["all_error"]["ap"] == metrics["source_local_unit_mean"]["all_error"]["ap"]
    assert archive.read_bytes() == original_archive


def test_interrupted_native_capture_reuses_completed_source_world(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    official_fixture(dataset)
    model = ModelAdapter(LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=512)))
    args = ["--dataset", str(dataset), "--output", str(output), "--stage", "capture",
            "--limit", "1", "--device", "cpu", "--dtype", "float32", "--save-heads"]
    from experiments.native_support.ragtruth_benchmark import capture
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()), \
            patch("state_audit.model.load_model", return_value=(model, CharacterTokenizer())), \
            patch.object(capture, "collect_condition", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            main(args)
    directory = output / "responses/000000"
    original = (directory / "with_source.npz").read_bytes()
    assert (directory / "routing_heads.npz").exists()
    assert not (directory / "capture_complete.json").exists()
    with patch("state_audit.model.load_model", return_value=(model, CharacterTokenizer())), \
            patch.object(capture, "collect_with_source", side_effect=AssertionError("Completed world rerun")):
        main([*args, "--resume"])
    assert (directory / "with_source.npz").read_bytes() == original
    assert (directory / "capture_complete.json").exists()
