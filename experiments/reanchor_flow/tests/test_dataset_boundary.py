from __future__ import annotations

import json

import numpy as np

from experiments.reanchor_flow.artifacts import safe_sample_key
from experiments.reanchor_flow.dataset import (
    ManifestAuditCorpus,
    ManifestLabelSource,
    SampleRecord,
    select_records,
)


def _write_manifest(path) -> None:
    path.write_text(
        json.dumps(
            {
                "audit_dataset_schema": 1,
                "name": "toy-grounding",
                "split": "test",
                "tokenizer_id": "tiny-tokenizer",
                "samples": [
                    {
                        "sample_id": "question/one",
                        "source_id": "document-a",
                        "task_type": "custom-grounding",
                        "generator_model": "generator-a",
                        "token_ids": [10, 11, 12, 13],
                        "response_start": 2,
                        "source_units": {
                            "token_unit_id": [0, 1, 2],
                            "name": ["other_prompt", "fact:1", "response:2"],
                            "kind": ["other_prompt", "fact", "response"],
                        },
                        "evidence_unit_id": [1],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_manifest_corpus_supplies_aligned_inputs_without_ragtruth(tmp_path) -> None:
    path = tmp_path / "test.json"
    _write_manifest(path)

    corpus = ManifestAuditCorpus.open(path)
    assert corpus.identity.kind == "manifest"
    assert corpus.identity.split == "test"
    assert corpus.identity.tokenizer_id == "tiny-tokenizer"
    assert corpus.records == (
        SampleRecord(
            "question/one",
            "document-a",
            "custom-grounding",
            "generator-a",
        ),
    )

    sample = corpus.load_sample(corpus.records[0])
    assert sample.token_ids.tolist() == [10, 11, 12, 13]
    assert sample.response_start == 2
    assert sample.full_response_tokens == 2
    assert sample.units.name == ("other_prompt", "fact:1", "response:2")
    assert sample.evidence_unit_id == (1,)


def test_manifest_labels_are_a_separate_post_capture_boundary(tmp_path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "audit_labels_schema": 1,
                "samples": {"question/one": [0, 1]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    labels = ManifestLabelSource.open(path).load(("question/one",))
    assert np.array_equal(labels["question/one"], np.asarray([0, 1]))


def test_sample_key_is_readable_reversible_and_contains_no_digest() -> None:
    assert safe_sample_key("plain-id") == "plain-id"
    assert safe_sample_key("folder/sample:1") == "folder%2Fsample%3A1"


def test_seeded_selection_is_independent_of_input_iteration_order() -> None:
    records = (
        SampleRecord("q1", "source-a", "QA", "g"),
        SampleRecord("q2", "source-a", "QA", "g"),
        SampleRecord("q3", "source-b", "QA", "g"),
        SampleRecord("q4", "source-c", "QA", "g"),
    )
    forward = select_records(
        records,
        tasks=("QA",),
        samples_per_task=2,
        seed=19,
    )
    reverse = select_records(
        reversed(records),
        tasks=("QA",),
        samples_per_task=2,
        seed=19,
    )
    assert forward == reverse
    assert len({record.source_id for record in forward}) == 2
