import json

import numpy as np
import torch

from experiments.grounded_route.artifacts import (
    EmbeddingIndex,
    EncodedTokenGraph,
    GraphSpec,
    load_checkpoint,
    load_embedding_index,
    load_encoded_graph,
    load_graph_spec,
    load_scores,
    merge_embedding_index,
    save_checkpoint,
    save_embedding_index,
    save_encoded_graph,
    save_graph_spec,
    save_scores,
)


def encoded_graph() -> EncodedTokenGraph:
    return EncodedTokenGraph(
        sample_id="sample",
        source_id="source",
        task_type="QA",
        response_start=2,
        layer_count=2,
        head_count=2,
        attention_floor=0.01,
        token_ids=torch.tensor([101, 102, 103, 104]),
        node_embedding=torch.arange(24, dtype=torch.float32).reshape(4, 6),
        edge_index=torch.tensor([[0, 1, 2], [2, 3, 3]]),
        edge_layer=torch.tensor([0, 0, 1]),
        edge_head=torch.tensor([0, 1, 0]),
        edge_weight=torch.tensor([0.1234, 0.2345, 0.3456]),
        diagonal=torch.full((2, 2, 2), 0.1234),
        unresolved=torch.full((2, 2, 2), 0.5432),
        lineage=torch.softmax(torch.arange(24).reshape(2, 2, 2, 3).float(), dim=-1),
    )


def test_encoded_graph_is_compact_on_disk_and_compute_ready_after_load(tmp_path):
    original = encoded_graph()
    path = tmp_path / "graph.pt"
    save_encoded_graph(path, original)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["token_ids"].dtype == torch.int32
    assert payload["edge_index"].dtype == torch.int32
    assert payload["edge_layer"].dtype == torch.int16
    assert payload["edge_head"].dtype == torch.int16
    assert payload["edge_weight"].dtype == torch.float16
    assert payload["diagonal"].dtype == torch.float16
    assert payload["unresolved"].dtype == torch.float16
    assert payload["lineage"].dtype == torch.float16
    assert payload["node_embedding"].dtype == torch.float32

    restored = load_encoded_graph(path)
    assert restored.token_ids.dtype == torch.int64
    assert restored.edge_index.dtype == torch.int64
    assert restored.edge_layer.dtype == torch.int64
    assert restored.edge_head.dtype == torch.int64
    assert restored.edge_weight.dtype == torch.float32
    assert restored.diagonal.dtype == torch.float32
    assert restored.unresolved.dtype == torch.float32
    assert restored.lineage.dtype == torch.float32
    assert torch.equal(restored.node_embedding, original.node_embedding)
    assert torch.allclose(restored.edge_weight, original.edge_weight, atol=5e-4)
    assert torch.allclose(restored.diagonal, original.diagonal, atol=5e-4)
    assert torch.allclose(restored.unresolved, original.unresolved, atol=5e-4)
    assert torch.allclose(restored.lineage, original.lineage, atol=5e-4)


def test_graph_spec_checkpoint_embedding_and_score_round_trip(tmp_path):
    spec = GraphSpec(
        dataset_root="/data/train",
        dataset_manifest_sha256="a" * 64,
        split="train",
        task="QA",
        sample_ids=("sample",),
        layer_count=2,
        head_count=2,
        graph_config={"block_rows": 128, "numerical_tolerance": 4e-3},
    )
    spec_path = tmp_path / "graph.json"
    save_graph_spec(spec_path, spec)
    assert load_graph_spec(spec_path) == spec
    assert json.loads(spec_path.read_text())["labels_included"] is False

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        {"state_dict": {"weight": torch.ones(3)}, "config": {"hidden": 6}},
    )
    checkpoint = load_checkpoint(checkpoint_path)
    assert torch.equal(checkpoint["state_dict"]["weight"], torch.ones(3))
    assert checkpoint["labels_included"] is False

    index = merge_embedding_index([encoded_graph()])
    assert isinstance(index, EmbeddingIndex)
    index_path = tmp_path / "index.npz"
    save_embedding_index(index_path, index, scope="calibration")
    restored, metadata = load_embedding_index(index_path)
    assert np.array_equal(restored.embedding, index.embedding)
    assert metadata["scope"].item() == "calibration"

    score_path = tmp_path / "scores.npz"
    save_scores(score_path, restored, np.array([0.1, 0.9], dtype=np.float32))
    scores = load_scores(score_path)
    assert "embedding" not in scores
    assert np.allclose(scores["score"], [0.1, 0.9])
    assert not bool(scores["labels_included"].item())
