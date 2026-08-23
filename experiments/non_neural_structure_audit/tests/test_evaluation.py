import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.non_neural_structure_audit.artifacts import save_npz, write_json
from experiments.non_neural_structure_audit.config import EvaluationConfig
from experiments.non_neural_structure_audit.evaluation import StructureEvaluator
from experiments.non_neural_structure_audit.features import RELATION_NAMES


class EvaluationSample:
    sample_id = "sample-0"
    source_id = "source-0"

    def __init__(self):
        self.release_calls = 0

    def attention(self):
        return SimpleNamespace(num_response_tokens=4)

    def release_attention(self):
        self.release_calls += 1


class EvaluationDataset:
    def __init__(self):
        self.sample = EvaluationSample()

    def __getitem__(self, sample_id):
        assert sample_id == "sample-0"
        return self.sample

    def prepare_evaluation_labels(self):
        return SimpleNamespace(
            response_labels=lambda sample: torch.tensor([0, 1, 0, 1])
        )


def test_evaluation_opens_labels_only_after_scores_and_marks_small_run_as_smoke(
    tmp_path, monkeypatch
):
    score_dir = tmp_path / "scores"
    sample_path = score_dir / "samples" / "sample-0.npz"
    relations = np.tile(
        np.linspace(0.0, 1.0, 4, dtype=np.float32)[:, None],
        (1, len(RELATION_NAMES)),
    )
    save_npz(
        sample_path,
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
            "labels_read": False,
            "trace_alignment": "post_token_query_at_same_position",
            "evaluation_alignment": "query_t_to_response_token_t_plus_1",
            "claim_scope": "prompt-connected attention-routing proxy",
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
                    "score_path": "samples/sample-0.npz",
                }
            ],
        },
    )
    dataset = EvaluationDataset()
    monkeypatch.setattr(
        "experiments.non_neural_structure_audit.evaluation.open_research_dataset",
        lambda path, device: dataset,
    )

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
    assert all(
        row["status"] == "NOT_EVALUATED_SMOKE"
        for row in report["decisions"]
        if row["audit"] != "A0"
    )
    assert dataset.sample.release_calls == 1
