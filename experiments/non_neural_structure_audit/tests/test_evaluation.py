import json
from types import SimpleNamespace

import numpy as np
import torch

from experiment_protocol import file_sha256
from experiments.non_neural_structure_audit.artifacts import save_npz, write_json
from experiments.non_neural_structure_audit.config import EvaluationConfig
from experiments.non_neural_structure_audit.evaluation import StructureEvaluator
from experiments.non_neural_structure_audit.features import RELATION_NAMES
from experiments.non_neural_structure_audit.protocol import method_sha256


class EvaluationSample:
    sample_id = "sample-0"
    source_id = "source-0"

    def __init__(self, dataset):
        self.dataset = dataset
        self.release_calls = 0
        self.attention_calls = 0

    def attention(self):
        self.attention_calls += 1
        return SimpleNamespace(
            num_response_tokens=4,
            response_idx=0,
            token_ids=torch.arange(4),
        )

    def release_attention(self):
        self.release_calls += 1


class EvaluationDataset:
    def __init__(self, root):
        self.root = root
        self.root.mkdir()
        self.manifest = {"split": "test"}
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        self.sample = EvaluationSample(self)
        self.sample_ids = ["sample-0"]
        self.labels_opened = False

    def __getitem__(self, sample_id):
        assert sample_id == "sample-0"
        return self.sample

    def prepare_evaluation_labels(self):
        self.labels_opened = True
        return SimpleNamespace(
            positive_runs=lambda sample_id, response_count: [[1, 2], [3, 4]]
        )


def test_evaluation_opens_labels_only_after_scores_and_marks_small_run_as_smoke(
    tmp_path, monkeypatch
):
    score_dir = tmp_path / "scores"
    reference_path = tmp_path / "reference.npz"
    reference_path.write_bytes(b"frozen reference")
    sample_path = score_dir / "samples" / "sample-0.npz"
    relations = np.tile(
        np.linspace(0.0, 1.0, 4, dtype=np.float32)[:, None],
        (1, len(RELATION_NAMES)),
    )
    save_npz(
        sample_path,
        schema=np.asarray("non-neural-structure-score-v1"),
        sample_id=np.asarray("sample-0"),
        source_id=np.asarray("source-0"),
        relation_names=np.asarray(RELATION_NAMES),
        response_token_ids=np.arange(4, dtype=np.int32),
        relation_scores=relations,
        final_relation_scores=relations,
        response_endpoint_null_relation_scores=np.stack(
            (relations[::-1], relations[::-1])
        ),
        response_endpoint_null_changed_fraction=np.asarray([0.8, 0.9]),
        layer_shuffle_relation_scores=np.stack((relations[::-1], relations[::-1])),
    )
    write_json(
        score_dir / "manifest.json",
        {
            "schema": "non-neural-structure-manifest-v2",
            "labels_read": False,
            "trace_alignment": "post_token_query_at_same_position",
            "evaluation_alignment": "query_t_to_response_token_t_plus_1",
            "claim_scope": "prompt-connected attention-routing proxy",
            "method_sha256": method_sha256(),
            "dataset_manifest_sha256": file_sha256(
                tmp_path / "dataset" / "manifest.json"
            )
            if (tmp_path / "dataset" / "manifest.json").exists()
            else "pending",
            "reference_path": str(reference_path.resolve()),
            "reference_sha256": file_sha256(reference_path),
            "reference_source_ids": ["source-train"],
            "test_source_ids": ["source-0"],
            "test_sample_ids": ["sample-0"],
            "audit_scope": "complete_split",
            "config": {"null_replicates": 2, "layer_shuffle_replicates": 2},
            "relation_names": list(RELATION_NAMES),
            "response_endpoint_null_relations": [
                "prompt_connected_lineage",
                "inherited_response_base",
                "multihop_response_base",
                "lineage_margin",
            ],
            "samples": [
                {
                    "sample_id": "sample-0",
                    "source_id": "source-0",
                    "response_length": 4,
                    "score_path": "samples/sample-0.npz",
                    "score_sha256": file_sha256(sample_path),
                    "null_audit": {
                        "row_mass_max_error": 0,
                        "role_mass_max_error": 0,
                        "source_count_degree_max_error": 0,
                        "causal_violations": 0,
                        "duplicate_edges": 0,
                    },
                }
            ],
        },
    )
    dataset = EvaluationDataset(tmp_path / "dataset")
    manifest = json.loads((score_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_manifest_sha256"] = file_sha256(dataset.root / "manifest.json")
    write_json(score_dir / "manifest.json", manifest)
    monkeypatch.setattr(
        "experiments.non_neural_structure_audit.evaluation.open_research_dataset",
        lambda path, device: dataset,
    )
    from experiments.non_neural_structure_audit import evaluation, evaluation_data

    original_loader = evaluation.load_frozen_samples
    original_validator = evaluation_data.validate_frozen_scores
    captured = {}

    def tracked_validator(**arguments):
        assert dataset.labels_opened is False
        original_validator(**arguments)
        captured["scores_validated"] = True

    def tracked_loader(**arguments):
        result = original_loader(**arguments)
        assert captured["scores_validated"] is True
        captured["bundle"] = result
        return result

    monkeypatch.setattr(evaluation_data, "validate_frozen_scores", tracked_validator)
    monkeypatch.setattr(evaluation, "load_frozen_samples", tracked_loader)

    output = tmp_path / "evaluation"
    StructureEvaluator(
        EvaluationConfig(
            bootstrap_replicates=5,
            permutation_replicates=5,
            minimum_confirmation_samples=100,
        )
    ).run(split_root="test", score_dir=score_dir, output_dir=output)

    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert report["labels_read"] is True
    assert report["scope"] == "smoke"
    assert report["a0_components"] == {
        "artifact_binding_verified": True,
        "gold_alignment_verified": False,
        "pipeline_label_permutation_verified": False,
    }
    assert report["decisions"][0]["status"] == "INCONCLUSIVE_A0_CONTROLS_MISSING"
    assert report["scientific_status"] == "ENGINEERING_SMOKE_ONLY"
    assert all(
        row["scientific_status"] == "ENGINEERING_SMOKE_ONLY"
        for row in report["relation_metrics"]
    )
    assert all(
        row["status"] == "NOT_EVALUATED_SMOKE"
        for row in report["decisions"]
        if row["audit"] != "A0"
    )
    assert dataset.sample.attention_calls == 1
    assert dataset.sample.release_calls == 2
    assert isinstance(captured["bundle"].samples[0].relation, np.memmap)
    assert list((output / ".scratch").iterdir()) == []
    assert captured["bundle"].samples[0].endpoint_null.shape == relations.shape
    assert captured["bundle"].samples[0].layer_shuffle.shape == relations.shape


def test_evaluation_rejects_a_mismatched_trace_before_opening_labels(
    tmp_path, monkeypatch
):
    score_dir = tmp_path / "scores"
    write_json(
        score_dir / "manifest.json",
        {
            "labels_read": False,
            "trace_alignment": "wrong_alignment",
            "evaluation_alignment": "query_t_to_response_token_t_plus_1",
        },
    )
    opened = False

    def unexpected_loader(**arguments):
        nonlocal opened
        opened = True

    monkeypatch.setattr(
        "experiments.non_neural_structure_audit.evaluation.load_frozen_samples",
        unexpected_loader,
    )

    import pytest

    with pytest.raises(ValueError, match="trace alignment"):
        StructureEvaluator().run(
            split_root="test", score_dir=score_dir, output_dir=tmp_path / "evaluation"
        )
    assert opened is False
