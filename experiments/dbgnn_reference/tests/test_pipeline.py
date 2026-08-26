from pathlib import Path

import torch

from experiments.dbgnn_reference.config import DBGNNConfig
from experiments.dbgnn_reference.pipeline import detect, encode, fit, load_checkpoint
from experiments.grounded_route.artifacts import (
    EncodedTokenGraph,
    merge_embedding_index,
    save_embedding_index,
    save_encoded_graph,
    sha256,
)
from experiments.grounded_route.graph_effectiveness.data import load_bundle
from experiments.grounded_route.tests.helpers import make_graph


def _encoded(sample_id: str, source_id: str) -> EncodedTokenGraph:
    graph = make_graph(layers=3, heads=2, response_count=5).canonicalize()
    lineage = torch.zeros(graph.response_count, graph.layer_count, graph.head_count, 3)
    lineage[..., 2] = 1.0
    return EncodedTokenGraph(
        sample_id=sample_id,
        source_id=source_id,
        task_type="QA",
        response_start=graph.response_start,
        layer_count=graph.layer_count,
        head_count=graph.head_count,
        attention_floor=graph.attention_floor,
        token_ids=graph.token_ids,
        node_embedding=torch.zeros(graph.token_count, 4),
        edge_index=torch.stack((graph.edges.source, graph.edges.target)),
        edge_layer=graph.edges.layer,
        edge_head=graph.edges.head,
        edge_weight=graph.edges.weight,
        diagonal=graph.diagonal,
        unresolved=graph.unresolved,
        lineage=lineage,
    )


def _bundle(root: Path, split: str, samples) -> Path:
    graph_dir = root / "graphs"
    graph_dir.mkdir(parents=True)
    paths = []
    hashes = []
    for number, graph in enumerate(samples):
        relative = Path("graphs") / f"{number:08d}.pt"
        save_encoded_graph(root / relative, graph)
        paths.append(relative.as_posix())
        hashes.append(sha256(root / relative))
    index = root / "index.npz"
    save_embedding_index(
        index,
        merge_embedding_index(samples),
        dataset_manifest_sha256="a" * 64,
        graph_spec_sha256=("b" if split == "train" else "c") * 64,
        split=split,
        scope="calibration" if split == "train" else "all",
        encoded_graph_sample_ids=[graph.sample_id for graph in samples],
        encoded_graph_paths=paths,
        encoded_graph_sha256=hashes,
        variant="real",
        message_mode="neighbor",
        changed_fraction=0.0,
        audit_scope="selected_samples",
    )
    return index


def test_fit_encode_detect_exports_verified_node_graphs(tmp_path):
    train = _bundle(
        tmp_path / "source_train",
        "train",
        [
            _encoded("train-a-0", "source-a"),
            _encoded("train-a-1", "source-a"),
            _encoded("train-b-0", "source-b"),
            _encoded("train-b-1", "source-b"),
            _encoded("train-c-0", "source-c"),
            _encoded("train-c-1", "source-c"),
        ],
    )
    test = _bundle(
        tmp_path / "source_test",
        "test",
        [_encoded("test-d-0", "source-d"), _encoded("test-d-1", "source-d")],
    )
    checkpoint = tmp_path / "run" / "checkpoint.pt"
    fit(
        train,
        checkpoint,
        config=DBGNNConfig(
            hidden_dim=8,
            embedding_dim=6,
            dropout=0.0,
            positives_per_graph=8,
            epochs=1,
        ),
    )
    checkpoint_payload = load_checkpoint(checkpoint)
    assert set(checkpoint_payload["fit_source_ids"]).isdisjoint(
        checkpoint_payload["validation_source_ids"]
    )
    assert set(checkpoint_payload["calibration_source_ids"]).isdisjoint(
        set(checkpoint_payload["fit_source_ids"])
        | set(checkpoint_payload["validation_source_ids"])
    )
    assert checkpoint_payload["history"][0]["positive_pairs"] > 0
    assert checkpoint_payload["history"][0]["validation_positive_pairs"] > 0
    calibration_out = tmp_path / "run" / "calibration"
    test_out = tmp_path / "run" / "test"
    encode(train, checkpoint, calibration_out, scope="calibration")
    encode(test, checkpoint, test_out, scope="all")

    calibration_bundle = load_bundle(calibration_out / "index.npz")
    test_bundle = load_bundle(test_out / "index.npz")
    assert calibration_bundle.index.embedding.shape[1] == 6
    assert set(calibration_bundle.index.source_id) == set(
        checkpoint_payload["calibration_source_ids"]
    )
    assert test_bundle.index.embedding.shape[1] == 6
    assert all(graph.node_embedding.shape[1] == 6 for graph in test_bundle.iter_graphs())

    report = detect(
        calibration_out / "index.npz",
        test_out / "index.npz",
        tmp_path / "run" / "detector.npz",
        tmp_path / "run" / "scores.npz",
    )
    assert report["nodes"] == len(test_bundle.index.embedding)
    assert report["labels_read"] is False
