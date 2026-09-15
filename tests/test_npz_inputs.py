"""Existing cache contracts: native fields, population, records and exports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.unsupervised_token_graph.cache_index import CacheIndex, identity_fields
from experiments.unsupervised_token_graph.channels import iter_channels
from experiments.unsupervised_token_graph.data import ResponseCache
from experiments.unsupervised_token_graph.reanchor_run import analyze_record, run
from experiments.unsupervised_token_graph.reanchor_evaluate import evaluate, main as evaluate_main


def canonical(path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(token_ids=np.arange(5), response_idx=2,
                  attention_diagonal=np.array([[[0, 0, .2, .2, .2]]]),
                  response_row_ptr=np.array([0, 1, 3, 5]),
                  response_column_indices=np.array([0, 0, 2, 1, 3]),
                  response_values=np.array([.8, .3, .5, .3, .5]))
    arrays.update(extra)
    np.savez_compressed(path, **arrays)
    return path


def identity(rid="10005"):
    return dict(id=rid, source_id="s1", official_split="test", task="QA", generator="fixture",
                prompt_length=2, token_ids=list(range(5)), offsets=[[0, 1], [1, 2], [2, 3]],
                response_sha256=hashlib.sha256(b"abc").hexdigest())


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("id_key,split_key", [("id", "split"), ("response_id", "official_split"),
                                              ("sample_id", "dataset_split")])
def test_native_npz_identity_without_metadata_and_without_reading_labels(tmp_path, id_key, split_key):
    path = canonical(tmp_path / "attention_10005.npz", **{
        id_key: "10005", split_key: "test", "source_id": "s1", "task_type": "QA",
        "model": "fixture", "response": "abc", "offsets": [[0, 1], [1, 2], [2, 3]],
        "labels": np.array([{"do_not_unpickle": True}], dtype=object)})
    record = ResponseCache().load(path)
    assert record.response_id == "10005" and record.source_id == "s1"
    assert record.metadata["split"] == "test" and record.metadata["generator"] == "fixture"
    assert record.metadata["response_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert "labels" not in record.metadata


def test_embedded_record_json_is_read_without_object_pickle(tmp_path):
    row = identity()
    path = canonical(tmp_path / "x.npz", record_json=json.dumps(row))
    record = ResponseCache().load(path)
    assert record.response_id == row["id"]
    np.testing.assert_array_equal(record.offsets, row["offsets"])


@pytest.mark.parametrize("name", ["inputs.jsonl", "records.jsonl", "records.json"])
def test_existing_index_is_detected_without_cache_or_trace_field(tmp_path, name):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    index = path.parent / name
    index.write_text(json.dumps([identity()]) if name.endswith(".json") else json.dumps(identity()))
    record = ResponseCache(index=CacheIndex(path.parent)).load(path)
    assert record.response_id == "10005" and record.source_id == "s1"
    np.testing.assert_array_equal(record.offsets, identity()["offsets"])


def test_population_can_be_elsewhere_and_resolves_annotations_without_reading_them(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    population = tmp_path / "population"
    jsonl(population / "inputs.jsonl", [identity()])
    dataset = tmp_path / "RAGTruth"; dataset.mkdir()
    (dataset / "response.jsonl").write_text("not JSON: scoring must not read annotations")
    (population / "settings.json").write_text(json.dumps({"dataset": str(dataset)}))
    out = tmp_path / "out"
    rows = run(path.parent, out, population=population, controls=False)
    assert rows[0]["id"] == "10005"
    settings = json.loads((out / "settings.json").read_text())
    assert settings["annotations"] == str(dataset / "response.jsonl")
    assert settings["labels_read"] is False


def test_records_index_reads_identity_arrays_from_existing_companion_npz(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    records = tmp_path / "existing_features"; records.mkdir()
    row = identity(); tokens = row.pop("token_ids"); offsets = row.pop("offsets")
    jsonl(records / "records.jsonl", [row])
    np.savez_compressed(records / "10005.npz", token_ids=tokens, offsets=offsets,
                        values=np.array([{"not_read": 1}], dtype=object),
                        labels=np.array([{"not_read": 1}], dtype=object))
    out = tmp_path / "out"
    rows = run(path.parent, out, index=records, controls=False)
    assert rows[0]["identity_files"][0][0] == str(records / "10005.npz")
    with np.load(out / "samples/attention_10005.npz") as data:
        np.testing.assert_array_equal(data["offsets"], offsets)
    assert run(path.parent, out, index=records, controls=False, resume=True) == rows
    with (records / "10005.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity NPZ changed"):
        run(path.parent, out, index=records, controls=False, resume=True)


@pytest.mark.parametrize("field,value", [("token_ids", [0, 1, 2, 3, 999]), ("prompt_length", 3)])
def test_same_id_and_same_length_cannot_override_different_tokenization(tmp_path, field, value):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    row = identity(); row[field] = value
    index = jsonl(tmp_path / "population" / "inputs.jsonl", [row])
    with pytest.raises(ValueError, match="mismatch"):
        ResponseCache(index=CacheIndex(path.parent, index)).load(path)


def test_feature_only_offsets_without_token_ids_do_not_fake_alignment(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    row = identity(); del row["token_ids"]
    index = jsonl(tmp_path / "features" / "records.jsonl", [row])
    with pytest.raises(ValueError, match="matching token_ids"):
        ResponseCache(index=CacheIndex(path.parent, index)).load(path)


def test_native_offsets_can_be_cross_checked_against_structural_records(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz", offsets=identity()["offsets"])
    row = identity(); del row["token_ids"]
    index = jsonl(tmp_path / "features" / "records.jsonl", [row])
    record = ResponseCache(index=CacheIndex(path.parent, index)).load(path)
    assert record.response_id == "10005"


def test_conflicting_native_and_index_identity_is_not_silently_overwritten(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz", source_id="different")
    index = jsonl(tmp_path / "inputs.jsonl", [identity()])
    with pytest.raises(ValueError, match="source_id"):
        ResponseCache(index=CacheIndex(path.parent, index)).load(path)


def test_index_changes_are_not_silently_resumed(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    index = jsonl(tmp_path / "inputs.jsonl", [identity()])
    out = tmp_path / "out"
    run(path.parent, out, controls=False)
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="input index"):
        run(path.parent, out, controls=False, resume=True)


def test_six_field_canonical_still_runs_without_any_index(tmp_path):
    path = canonical(tmp_path / "only.npz")
    out = tmp_path / "out"
    rows = run(path, out, controls=False)
    assert rows[0]["source_id"] == "" and rows[0]["split"] == ""
    assert (out / "samples/only.npz").exists()


def test_exported_adjacency_keeps_absolute_queries_and_physical_channel(tmp_path):
    path = canonical(tmp_path / "canonical.npz")
    channel = next(iter_channels(ResponseCache().load(path)))
    export = tmp_path / "layer_head.npz"
    np.savez_compressed(export, adjacency=channel.attention.toarray(),
                        A_RP=channel.attention[:, :2].toarray(), A_RR=channel.attention[:, 2:].toarray(),
                        token_ids=np.arange(5), response_idx=2, layer=10, head=7)
    record = ResponseCache().load(export)
    exported = next(iter_channels(record, layers=[10], heads=[7]))
    assert exported.layer == 10 and exported.head == 7
    np.testing.assert_array_equal(exported.queries, channel.queries)
    np.testing.assert_allclose(exported.attention.toarray(), channel.attention.toarray())
    assert not list(iter_channels(record, heads=[0]))


@pytest.mark.parametrize("key", ["local_attention", "values"])
def test_non_graph_npz_is_not_misread_as_attention(tmp_path, key):
    path = tmp_path / "other.npz"
    np.savez(path, **{key: np.zeros((3, 4)), "prompt_length": 2})
    with pytest.raises(ValueError, match="not full attention graphs"):
        ResponseCache().load(path)


def test_aliases_cannot_hide_conflicting_boundary_or_hash():
    with pytest.raises(ValueError, match="conflicting"):
        identity_fields({"prompt_length": 2, "response_idx": 3})
    with pytest.raises(ValueError, match="response text"):
        identity_fields({"response": "abc", "response_sha256": "wrong"})


def test_native_and_index_loading_produce_identical_scores(tmp_path):
    bare = canonical(tmp_path / "bare" / "attention_10005.npz")
    native = canonical(tmp_path / "native" / "attention_10005.npz", **identity())
    index = jsonl(tmp_path / "population" / "inputs.jsonl", [identity()])
    a = analyze_record(ResponseCache().load(native))
    b = analyze_record(ResponseCache(index=CacheIndex(bare.parent, index)).load(bare))
    for key in a:
        if key != "cache_format":
            np.testing.assert_allclose(a[key], b[key], equal_nan=True)


def test_npz_to_evaluation_without_handmade_metadata(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz", **identity())
    gold = jsonl(tmp_path / "RAGTruth" / "response.jsonl", [dict(
        id="10005", source_id="s1", split="test", response="abc", labels=[{"start": 1, "end": 2}])])
    out = tmp_path / "out"
    run(path.parent, out, controls=False)
    report = evaluate(out, gold, bootstrap=0)
    all_error = report["groups"]["ALL"]["views"]["all_error"]["event_strength"]
    assert all_error["eligible_tokens"] == 3
    assert all_error["evaluated_tokens"] == 2
    assert all_error["evaluated_positives"] == 1


def test_evaluation_cli_can_use_saved_population_dataset_path(tmp_path):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    population = tmp_path / "population"
    jsonl(population / "inputs.jsonl", [identity()])
    dataset = tmp_path / "RAGTruth"
    jsonl(dataset / "response.jsonl", [dict(id="10005", source_id="s1", split="test",
                                           response="abc", labels=[{"start": 1, "end": 2}])])
    (population / "settings.json").write_text(json.dumps({"dataset": str(dataset)}))
    out = tmp_path / "out"
    run(path.parent, out, population=population, controls=False)
    evaluate_main(["--predictions", str(out), "--output", str(out / "evaluation.json"), "--bootstrap", "0"])
    assert (out / "evaluation.json").exists()


def test_analysis_only_evaluation_skip_is_explicit(tmp_path, capsys):
    path = canonical(tmp_path / "cache" / "attention_10005.npz")
    out = tmp_path / "out"
    run(path.parent, out, controls=False)
    evaluate_main(["--predictions", str(out), "--output", str(out / "evaluation.json"), "--if-available"])
    assert "Evaluation skipped" in capsys.readouterr().out
    assert not (out / "evaluation.json").exists()
