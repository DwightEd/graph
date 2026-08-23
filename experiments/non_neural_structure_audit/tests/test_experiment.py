import gc
import json
from types import SimpleNamespace
import weakref

import numpy as np
import torch

from experiments.non_neural_structure_audit.config import AuditConfig
from experiments.non_neural_structure_audit.experiment import StructureAudit


class Sample:
    def __init__(self, sample_id):
        self.sample_id = sample_id
        self.source_id = f"source-{sample_id}"
        self.task_type = "QA"
        self.release_calls = 0
        self._attention = SimpleNamespace(
            response_idx=1,
            num_response_tokens=2,
            num_tokens=3,
            num_layers=2,
            num_heads=1,
            attention_floor=0.01,
            response_values=torch.empty(0),
            attention_diagonal=torch.zeros((2, 1, 3)),
            token_ids=torch.arange(3),
        )

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=8192):
        del block_rows
        yield SimpleNamespace(
            layer=torch.tensor([0, 1]),
            head=torch.tensor([0, 0]),
            query=torch.tensor([0, 1]),
            source=torch.tensor([0, 1]),
            weight=torch.tensor([1.0, 1.0]),
        )

    def release_attention(self):
        self.release_calls += 1


class Dataset:
    def __init__(self, prefix, count=1):
        self.samples = {
            f"{prefix}-{index}": Sample(f"{prefix}-{index}")
            for index in range(count)
        }
        self.sample_ids = list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def prepare_evaluation_labels(self):
        raise AssertionError("fit and score must not open labels")


def test_fit_and_score_freeze_compact_artifacts_without_opening_labels(
    tmp_path, monkeypatch
):
    train = Dataset("train")
    test = Dataset("test")
    monkeypatch.setattr(
        "experiments.non_neural_structure_audit.experiment.open_research_dataset",
        lambda path, device: train if str(path) == "train" else test,
    )
    config = AuditConfig(
        null_replicates=2,
        layer_shuffle_replicates=2,
        show_progress=False,
    )
    audit = StructureAudit(config)
    reference = tmp_path / "reference.npz"
    score_dir = tmp_path / "scores"

    audit.fit(train_split="train", output=reference, device="cpu")
    audit.score(
        split_root="test",
        reference_path=reference,
        output_dir=score_dir,
        device="cpu",
    )

    manifest = json.loads((score_dir / "manifest.json").read_text())
    assert manifest["labels_read"] is False
    assert manifest["trace_alignment"] == "post_token_query_at_same_position"
    assert manifest["evaluation_alignment"] == "query_t_to_response_token_t_plus_1"
    score_path = score_dir / manifest["samples"][0]["score_path"]
    assert score_path.is_file()
    with np.load(score_path, allow_pickle=False) as arrays:
        assert arrays["layer_shuffle_relation_scores"].shape[0] == 2
        direct = list(arrays["relation_names"]).index("direct_role")
        np.testing.assert_allclose(
            arrays["layer_shuffle_relation_scores"][:, :, direct],
            np.repeat(
                arrays["final_relation_scores"][None, :, direct], 2, axis=0
            ),
        )
    assert train["train-0"].release_calls == 1
    assert test["test-0"].release_calls == 1


def test_fit_drops_previous_sample_graph_before_loading_the_next(
    tmp_path, monkeypatch
):
    train = Dataset("train", count=2)
    monkeypatch.setattr(
        "experiments.non_neural_structure_audit.experiment.open_research_dataset",
        lambda path, device: train,
    )
    from experiments.non_neural_structure_audit import experiment

    original = experiment._real_analysis
    previous_operator = None

    def tracked(sample, config):
        nonlocal previous_operator
        gc.collect()
        if previous_operator is not None:
            assert previous_operator() is None
        result = original(sample, config)
        previous_operator = weakref.ref(result[2])
        return result

    monkeypatch.setattr(experiment, "_real_analysis", tracked)
    StructureAudit(AuditConfig(show_progress=False)).fit(
        train_split="train",
        output=tmp_path / "reference.npz",
        device="cpu",
    )
