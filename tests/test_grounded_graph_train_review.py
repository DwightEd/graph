import argparse
import json

import numpy as np
import pytest
import torch

from next_iteration.grounded_graph_adapter import GroundedGraphAdapter, pointer_loss, token_loss
from next_iteration.grounded_graph_train import batch
import next_iteration.grounded_graph_train as train


def test_adapter_initial_hidden_is_frozen_lm_state_and_rejects_cross_sample_edges():
    adapter = GroundedGraphAdapter(input_dim=4, adapter_dim=3, node_types=3, edge_types=4, message_steps=1)
    source = torch.randn(3, 4)
    query = torch.randn(2, 4)
    node_types = torch.tensor([0, 1, 2])
    graph_ids = torch.tensor([0, 0, 1])
    query_graph_ids = torch.tensor([0, 1])
    available = torch.tensor([True, True, True])

    output = adapter(
        source,
        query,
        node_types,
        torch.tensor([[0], [1]]),
        torch.tensor([0]),
        graph_ids,
        query_graph_ids,
        available,
    )

    assert torch.allclose(output["hidden"], query)
    assert output["eligible"].tolist() == [[True, True, False], [False, False, True]]

    with pytest.raises(ValueError, match="edge crosses independent samples"):
        adapter(
            source,
            query,
            node_types,
            torch.tensor([[0], [2]]),
            torch.tensor([0]),
            graph_ids,
            query_graph_ids,
            available,
        )


def test_batch_offsets_nodes_queries_edges_and_first_token_pointer_supervision():
    packet0 = {
        "nodes": [{"type": 0, "available": True}, {"type": 1, "available": True}],
        "target_token_ids": [3, 4],
        "edges": [[0, 1, 0]],
        "pointer_supervision": [{"query_index": 0, "owner_node_indices": [1]}],
    }
    packet1 = {
        "nodes": [{"type": 2, "available": True}],
        "target_token_ids": [2],
        "edges": [],
        "pointer_supervision": [{"query_index": 0, "owner_node_indices": [0]}],
    }
    arrays0 = {"source": np.ones((2, 4), dtype=np.float32), "query": np.ones((2, 4), dtype=np.float32)}
    arrays1 = {"source": np.ones((1, 4), dtype=np.float32) * 2, "query": np.ones((1, 4), dtype=np.float32) * 3}

    arguments, targets, positives, query_mask = batch([(packet0, arrays0), (packet1, arrays1)], "cpu")

    assert arguments["source_states"].shape == (3, 4)
    assert arguments["query_states"].shape == (3, 4)
    assert arguments["graph_ids"].tolist() == [0, 0, 1]
    assert arguments["query_graph_ids"].tolist() == [0, 0, 1]
    assert arguments["edge_index"].tolist() == [[0], [1]]
    assert arguments["edge_types"].tolist() == [0]
    assert targets.tolist() == [3, 4, 2]
    assert query_mask.tolist() == [True, False, True]
    assert positives.nonzero().tolist() == [[0, 1], [2, 2]]


def test_losses_reject_trainable_lm_head_and_unavailable_positive_owner():
    hidden = torch.randn(2, 4)
    head = torch.randn(5, 4, requires_grad=True)
    with pytest.raises(ValueError, match="LM head must remain frozen"):
        token_loss(hidden, torch.tensor([1, 2]), head)

    output = {
        "pointer_logits": torch.zeros(1, 2),
        "eligible": torch.tensor([[True, False]]),
        "hidden": hidden[:1],
    }
    with pytest.raises(ValueError, match="escapes its sample or is unavailable"):
        pointer_loss(output, torch.tensor([[False, True]]), torch.tensor([True]))


def test_run_checkpoint_selection_can_keep_initial_if_all_epochs_worse(tmp_path, monkeypatch):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "manifest.json").write_text("{}\n")
    output = tmp_path / "train"
    monkeypatch.setitem(train.PROTOCOL, "epochs", 1)
    monkeypatch.setitem(train.PROTOCOL, "arms", ["graph"])
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.cuda, "mem_get_info", lambda: (5 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(train.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "frozen_head", lambda settings, device: torch.zeros((5, 1)))
    monkeypatch.setattr(train, "new_adapter", lambda device: torch.nn.Linear(1, 1))
    monkeypatch.setattr(train, "load_example", lambda *args, **kwargs: ({"packet": True}, {"arrays": True}, {}))
    monkeypatch.setattr(train, "model_manifest", lambda path: [])
    monkeypatch.setattr(train, "verify", lambda *args, **kwargs: ({
        "parent": {"kind": "source_reconstruction"},
        "model_path": str(tmp_path / "model"),
        "model_files": [],
        "code_sha256": {},
    }, [
        {"split": "train", "source_text_sha256": "train-sha", "status": "available"},
        {"split": "validation", "source_text_sha256": "val-sha", "status": "available"},
    ]))
    calls = iter([
        {"objective": 0.1, "token_ce": 0.1, "pointer_nll": 0.0, "tokens": 1, "pointers": 1, "pointer_top1": 1.0, "adapter_forwards": 1},
        {"objective": 0.2, "token_ce": 0.2, "pointer_nll": 0.0, "tokens": 1, "pointers": 1, "pointer_top1": 1.0, "adapter_forwards": 1},
        {"objective": 0.3, "token_ce": 0.3, "pointer_nll": 0.0, "tokens": 1, "pointers": 1, "pointer_top1": 1.0, "adapter_forwards": 1},
    ])
    monkeypatch.setattr(train, "measure", lambda *args, **kwargs: next(calls))

    train.run(argparse.Namespace(features=feature_dir, output=output))

    summary = json.loads((output / "summary.json").read_text())
    assert summary["arms"]["graph"]["selected"]["validation"]["objective"] == 0.1
