import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.non_neural_structure_audit.config import AuditConfig
from experiments.non_neural_structure_audit.experiment import (
    StructureAudit,
    _selected_ids,
)
from experiments.non_neural_structure_audit.features import LAYER_ORDER_RELATION_NAMES


class Attention:
    pass


class Sample:
    def __init__(self, sample_id, dataset):
        self.dataset = dataset
        self.sample_id = sample_id
        self.source_id = f"source-{sample_id}"
        self.task_type = "QA"
        self.release_calls = 0
        self._attention = self._new_attention()
        self.released_attention = None

    @staticmethod
    def _new_attention():
        attention = Attention()
        attention.response_idx = 1
        attention.num_response_tokens = 2
        attention.num_tokens = 3
        attention.num_layers = 2
        attention.num_heads = 1
        attention.attention_floor = 0.01
        attention.response_values = torch.empty(2)
        attention.attention_diagonal = torch.zeros((2, 1, 3))
        attention.token_ids = torch.arange(3)
        return attention

    def attention(self):
        if self._attention is None:
            self._attention = self._new_attention()
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
        self.released_attention = weakref.ref(self._attention)
        self._attention = None


class Dataset:
    def __init__(self, prefix, root, count=1):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        self.manifest = {"split": prefix}
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        self.samples = {
            f"{prefix}-{index}": Sample(f"{prefix}-{index}", self)
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
    train = Dataset("train", tmp_path / "train")
    test = Dataset("test", tmp_path / "test")
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

    from experiments.non_neural_structure_audit import experiment

    original_build = experiment.build_routing_state

    def tracked_build(edges):
        assert test["test-0"].release_calls == 1
        gc.collect()
        assert test["test-0"].released_attention() is None
        return original_build(edges)

    monkeypatch.setattr(experiment, "build_routing_state", tracked_build)
    original_swap = experiment.EndpointSwapPlan.sample

    def tracked_swap(plan, *arguments, **keywords):
        assert test["test-0"].release_calls == 1
        gc.collect()
        assert test["test-0"].released_attention() is None
        return original_swap(plan, *arguments, **keywords)

    monkeypatch.setattr(experiment.EndpointSwapPlan, "sample", tracked_swap)
    test["test-0"].source_id = None
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
    assert len(manifest["dataset_manifest_sha256"]) == 64
    assert manifest["reference_source_ids"] == ["source-train-0"]
    assert manifest["test_source_ids"] == ["test-0"]
    assert manifest["samples"][0]["source_id"] == "test-0"
    assert manifest["layer_order_null_relations"] == list(LAYER_ORDER_RELATION_NAMES)
    null_audit = manifest["samples"][0]["null_audit"]
    assert null_audit["source_count_degree_max_error"] == 0
    assert null_audit["changed_fraction_min"] <= null_audit["changed_fraction_mean"]
    score_path = score_dir / manifest["samples"][0]["score_path"]
    assert score_path.is_file()
    assert len(manifest["samples"][0]["score_sha256"]) == 64
    with np.load(score_path, allow_pickle=False) as arrays:
        assert arrays["layer_shuffle_relation_scores"].shape[0] == 2
        assert all(
            not np.array_equal(order, np.arange(len(order)))
            for order in arrays["layer_shuffle_order"]
        )
        direct = list(arrays["relation_names"]).index("direct_role")
        np.testing.assert_allclose(
            arrays["layer_shuffle_relation_scores"][:, :, direct],
            np.repeat(arrays["final_relation_scores"][None, :, direct], 2, axis=0),
        )
    assert train["train-0"].release_calls == 1
    assert test["test-0"].release_calls == 1


def test_fit_releases_attention_before_analysis_and_drops_previous_graph(
    tmp_path, monkeypatch
):
    train = Dataset("train", tmp_path / "train", count=2)
    monkeypatch.setattr(
        "experiments.non_neural_structure_audit.experiment.open_research_dataset",
        lambda path, device: train,
    )
    from experiments.non_neural_structure_audit import experiment

    original = experiment.build_routing_state
    previous_routing = None
    sample_ids = iter(train.sample_ids)

    def tracked(edges):
        nonlocal previous_routing
        sample = train[next(sample_ids)]
        assert sample.release_calls == 1
        gc.collect()
        assert sample.released_attention() is None
        if previous_routing is not None:
            assert previous_routing() is None
        result = original(edges)
        previous_routing = weakref.ref(result)
        return result

    monkeypatch.setattr(experiment, "build_routing_state", tracked)
    StructureAudit(AuditConfig(show_progress=False)).fit(
        train_split="train",
        output=tmp_path / "reference.npz",
        device="cpu",
    )


def test_default_selection_can_restrict_the_pipeline_to_qa(tmp_path):
    dataset = Dataset("test", tmp_path / "test", count=2)
    dataset["test-1"].task_type = "Summary"

    assert _selected_ids(dataset, limit=None, task_type="QA") == ["test-0"]
    assert _selected_ids(dataset, limit=None, task_type="all") == [
        "test-0",
        "test-1",
    ]
