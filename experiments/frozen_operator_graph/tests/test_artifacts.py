from pathlib import Path

import torch

from ..artifacts import load_graph_artifact, save_graph_artifact
from ..encoding import build_node_encoding
from ..graph import build_graph_tensors
from ..schema import OperatorGraphArtifact
from .helpers import synthetic_bundle


def _artifact() -> OperatorGraphArtifact:
    bundle = synthetic_bundle()
    graph = build_graph_tensors(bundle.capture, bundle.basis)
    encoding = build_node_encoding(graph)
    return OperatorGraphArtifact(
        sample_id="7",
        source_id="source-7",
        metadata={"task_type": "QA"},
        token_ids=bundle.capture.token_ids,
        response_start=bundle.capture.response_start,
        edge_index=graph.edge_index,
        edge_layer=graph.edge_layer,
        edge_role=graph.edge_role,
        edge_attention_code=graph.edge_attention_code,
        edge_features=graph.edge_features,
        edge_feature_names=graph.edge_feature_names,
        remainder_features=graph.remainder_features,
        remainder_feature_names=graph.remainder_feature_names,
        route_features=graph.route_features,
        route_feature_names=graph.route_feature_names,
        layer_features=graph.layer_features,
        layer_feature_names=graph.layer_feature_names,
        temporal_features=encoding.temporal_features,
        temporal_feature_names=encoding.temporal_feature_names,
        final_hidden=graph.final_hidden,
        node_embedding=encoding.node_embedding,
        node_feature_names=encoding.node_feature_names,
        audit=graph.audit,
        provenance={"labels_consumed_by_construction": False},
    ).validate()


def test_artifact_roundtrip_is_shape_and_value_faithful(tmp_path: Path):
    artifact = _artifact()
    path = tmp_path / "graph_7.pt"
    row = save_graph_artifact(path, artifact, output_dtype="float32")
    restored = load_graph_artifact(path, verify_sha256=row["sha256"])
    assert restored.node_feature_names == artifact.node_feature_names
    assert torch.equal(restored.edge_index, artifact.edge_index)
    assert torch.allclose(restored.node_embedding, artifact.node_embedding)
    assert restored.provenance["labels_consumed_by_construction"] is False


def test_constructed_split_dataset_reads_without_any_label_interface(tmp_path: Path):
    from ..artifacts import write_split_manifest
    from ..dataset import OperatorGraphDataset

    artifact = _artifact()
    root = tmp_path / "split"
    samples = root / "samples"
    samples.mkdir(parents=True)
    path = samples / "graph_7.pt"
    row = save_graph_artifact(path, artifact, output_dtype="float32")
    row["path"] = path.relative_to(root).as_posix()
    write_split_manifest(
        root,
        [row],
        checkpoint="synthetic",
        dataset_manifest_sha256="0" * 64,
        configuration={},
        feature_contract={"node_feature_names": list(artifact.node_feature_names)},
    )
    dataset = OperatorGraphDataset(root)
    restored = dataset["7"]
    assert restored.sample_id == "7"
    assert not hasattr(dataset, "labels")
