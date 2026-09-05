from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from experiments.reanchor_flow import subset_report
from experiments.reanchor_flow.subset_artifacts import (
    canonical_capture_config,
    capture_config_sha256,
)
from experiments.reanchor_flow.subset_data import file_sha256


def complete_capture(tmp_path):
    dataset_root = tmp_path / "cache"
    dataset_root.mkdir()
    dataset_manifest = dataset_root / "manifest.json"
    dataset_manifest.write_text('{"split":"test"}\n', encoding="utf-8")

    output = tmp_path / "output"
    world = output / "worlds" / "QA" / "sample-1.npz"
    world.parent.mkdir(parents=True)
    np.savez_compressed(world, native_world_schema=1)
    world_sha256 = file_sha256(world)

    config = {
        "model": "/model/tiny-llama",
        "model_dtype": "float32",
        "tokenizer": "tiny-llama",
        "dataset_root": str(dataset_root.resolve()),
        "dataset_manifest_sha256": file_sha256(dataset_manifest),
        "source_info_sha256": "0" * 64,
        "split": "test",
        "target_policy": "evenly-spaced",
        "flow_signal": "message",
        "carrier_scope": "response",
        "edge_coverage": 1.0,
        "query_chunk": 2,
        "root_screen_limit": 1,
        "carrier_limit": 1,
        "saved_edges": 4,
    }
    config_sha256 = capture_config_sha256(config)
    target = {
        "query_position": 4,
        "positive_token_id": 9,
        "negative_token_id": 8,
        "contrast_origin": "label_free_evenly-spaced_observed_token_vs_native_runner",
    }
    target_key = "q4_a9_b8_message"
    result_relative = f"audits/QA/sample-1/{target_key}.npz"
    result = output / result_relative
    result.parent.mkdir(parents=True)
    np.savez_compressed(
        result,
        subset_audit_schema=1,
        artifact_complete=1,
        capture_config_json=canonical_capture_config(config),
        capture_config_sha256=config_sha256,
        world_sha256=world_sha256,
        dataset_manifest_sha256=config["dataset_manifest_sha256"],
        source_info_sha256=config["source_info_sha256"],
        dataset_sample_id="sample-1",
        sample_id="sample-1",
        source_id="source-1",
        split="test",
        task_type="QA",
        generator_model="generator",
        tokenizer_id="tiny-llama",
        model_id="/model/tiny-llama",
        model_dtype="float32",
        target_selection_policy="evenly-spaced",
        target_selection_rank=0,
        response_start=4,
        prediction_position=5,
        query_position=4,
        positive_token_id=9,
        negative_token_id=8,
        contrast_origin=target["contrast_origin"],
        flow_signal="message",
        edge_coverage=1.0,
        carrier_scope="response",
        query_chunk=2,
        root_screen_limit=1,
        carrier_limit=1,
        edge_save_limit=4,
        labels_used_for_capture=0,
        token_ids=np.asarray([1, 2, 3, 4, 5, 9], dtype=np.int64),
        selected_root_confirmed=True,
        corridor_confirmed=True,
        carrier_any_confirmed=False,
        carrier_value_mediated=False,
        full_chain_confirmed=False,
        corridor_restoration_valid=True,
        root_value_effect=1.0,
        corridor_necessity=0.8,
        corridor_conditional_rescue=0.7,
        corridor_mediated_rescue=0.6,
    )
    audit_key = f"sample-1:{target_key}"
    manifest = {
        "subset_manifest_schema": 1,
        "config": config,
        "config_sha256": config_sha256,
        "selection": [
            {
                "sample_id": "sample-1",
                "source_id": "source-1",
                "task_type": "QA",
                "generator_model": "generator",
            }
        ],
        "analysis_complete": True,
        "labels_used_for_capture": False,
        "samples": {
            "sample-1": {
                "source_id": "source-1",
                "task_type": "QA",
                "world": "worlds/QA/sample-1.npz",
                "world_sha256": world_sha256,
                "targets": [target],
            }
        },
        "audits": {
            audit_key: {
                "result": result_relative,
                "complete": True,
                "dataset_sample_id": "sample-1",
                "sample_id": "sample-1",
                "source_id": "source-1",
                "task_type": "QA",
                "generator_model": "generator",
                "split": "test",
                **target,
                "flow_signal": "message",
                "target_rank": 0,
                "world_sha256": world_sha256,
                "config_sha256": config_sha256,
                "sha256": file_sha256(result),
            }
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_root, output, result


def test_subset_evaluation_opens_labels_only_after_complete_capture(
    tmp_path, monkeypatch
) -> None:
    dataset_root, output, _result = complete_capture(tmp_path)

    class Sample:
        def release_attention(self):
            pass

    class Labels:
        @staticmethod
        def response_labels(_sample):
            return torch.tensor([0, 1])

    class Dataset:
        def __getitem__(self, sample_id):
            assert sample_id == "sample-1"
            return Sample()

        @staticmethod
        def prepare_evaluation_labels(sample_ids):
            assert sample_ids == ["sample-1"]
            return Labels()

    def open_dataset(*_args, **kwargs):
        assert kwargs["retain_embedded_labels"] is True
        assert kwargs["verify_hashes"] is True
        return Dataset()

    monkeypatch.setattr(subset_report, "open_research_dataset", open_dataset)
    report = subset_report.evaluate_subset_split(dataset_root, output)
    hallucinated = report["groups"]["QA"]["hallucinated"]
    assert hallucinated["targets"] == 1
    assert hallucinated["corridor_confirmed_rate"] == 1.0
    assert report["labels_accessed_after_capture"]

    text = (output / "mechanism_evaluation.json").read_text(encoding="utf-8")
    assert "NaN" not in text

    def reject_constant(value):
        raise AssertionError(f"non-standard JSON constant: {value}")

    stored = json.loads(text, parse_constant=reject_constant)
    assert stored["groups"]["QA"]["clean"]["root_confirmed_rate"] is None


@pytest.mark.parametrize("failure", ("missing", "hash-mismatch"))
def test_subset_evaluation_preflights_artifacts_before_labels(
    tmp_path, monkeypatch, failure
) -> None:
    dataset_root, output, result = complete_capture(tmp_path)
    if failure == "missing":
        result.unlink()
    else:
        with result.open("ab") as stream:
            stream.write(b"changed")

    labels_opened = False

    def open_dataset(*_args, **_kwargs):
        nonlocal labels_opened
        labels_opened = True
        raise AssertionError("labels opened before artifact preflight")

    monkeypatch.setattr(subset_report, "open_research_dataset", open_dataset)
    with pytest.raises(ValueError, match="missing|hash mismatch"):
        subset_report.evaluate_subset_split(dataset_root, output)
    assert not labels_opened


def test_subset_evaluation_refuses_incomplete_capture(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "subset_manifest_schema": 1,
                "analysis_complete": False,
                "labels_used_for_capture": False,
            }
        ),
        encoding="utf-8",
    )
    try:
        subset_report.evaluate_subset_split(tmp_path / "cache", output)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete capture unexpectedly opened labels")
