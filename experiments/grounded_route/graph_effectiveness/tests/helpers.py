from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.grounded_route.artifacts import (
    EncodedTokenGraph,
    merge_embedding_index,
    save_embedding_index,
    save_encoded_graph,
    sha256,
)
from experiments.grounded_route.tests.helpers import make_graph


def make_encoded_graph(sample: int, *, embedding_shift: float = 0.0):
    graph = replace(
        make_graph(layers=2, heads=2, response_count=5),
        sample_id=f"sample-{sample}",
        source_id=f"source-{sample}",
    )
    generator = torch.Generator().manual_seed(100 + sample)
    embedding = torch.randn(
        graph.token_count,
        8,
        generator=generator,
    ) + embedding_shift
    lineage = torch.zeros(graph.response_count, 2, 2, 3)
    lineage[..., 0] = 1.0
    return EncodedTokenGraph(
        sample_id=graph.sample_id,
        source_id=graph.source_id,
        task_type=graph.task_type,
        response_start=graph.response_start,
        layer_count=graph.layer_count,
        head_count=graph.head_count,
        attention_floor=graph.attention_floor,
        token_ids=graph.token_ids,
        node_embedding=embedding,
        edge_index=torch.stack((graph.edges.source, graph.edges.target)),
        edge_layer=graph.edges.layer,
        edge_head=graph.edges.head,
        edge_weight=graph.edges.weight,
        diagonal=graph.diagonal,
        unresolved=graph.unresolved,
        lineage=lineage,
    )


def write_bundle(
    root: Path,
    *,
    variant: str = "real",
    message_mode: str = "neighbor",
    split: str = "test",
    embedding_shift: float = 0.0,
) -> Path:
    graphs = [
        make_encoded_graph(sample, embedding_shift=embedding_shift)
        for sample in range(5)
    ]
    graph_dir = root / "graphs"
    graph_dir.mkdir(parents=True)
    paths = []
    hashes = []
    for number, graph in enumerate(graphs):
        path = graph_dir / f"{number:08d}.pt"
        save_encoded_graph(path, graph)
        paths.append(path.relative_to(root).as_posix())
        hashes.append(sha256(path))

    metadata = {
        "dataset_manifest_sha256": "a" * 64,
        "checkpoint_sha256": ("c" if variant == "real" else "d") * 64,
        "graph_spec_sha256": "b" * 64,
        "split": split,
        "scope": "all" if split == "test" else "calibration",
        "variant": variant,
        "message_mode": message_mode,
        "changed_fraction": 0.0 if variant == "real" else 0.25,
        "encoded_graph_sample_ids": [graph.sample_id for graph in graphs],
        "encoded_graph_paths": paths,
        "encoded_graph_sha256": hashes,
    }
    if split == "test":
        metadata.update(
            audit_scope="selected_samples",
            reserved_source_ids=["train-source"],
            test_source_ids=[graph.source_id for graph in graphs],
            test_sample_ids=[graph.sample_id for graph in graphs],
        )
    else:
        metadata.update(
            calibration_sample_ids=[graph.sample_id for graph in graphs],
            calibration_source_ids=[graph.source_id for graph in graphs],
            encoder_source_ids=["encoder-source"],
        )
    index_path = root / "index.npz"
    save_embedding_index(
        index_path,
        merge_embedding_index(graphs),
        **metadata,
    )
    return index_path
