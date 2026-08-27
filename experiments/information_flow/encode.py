"""Create flow-transport node embeddings from frozen GCN token graphs."""

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.grounded_route.graph_effectiveness.data import load_bundle

from .transport import flow_embedding


def save_sample(path: Path, graph, output) -> None:
    np.savez_compressed(
        path,
        sample_id=np.asarray(graph.sample_id),
        source_id=np.asarray(graph.source_id),
        task_type=np.asarray(str(graph.task_type)),
        response_start=np.asarray(graph.response_start, dtype=np.int32),
        token_ids=graph.token_ids.cpu().numpy().astype(np.int64),
        base_embedding=graph.response_embedding.cpu().numpy().astype(np.float32),
        trajectory=output.trajectory.cpu().numpy().astype(np.float32),
        node_embedding=output.embedding.cpu().numpy().astype(np.float32),
    )


def encode_bundle(
    source_index,
    output_dir,
    *,
    mode: str = "sketch",
    checkpoints: int = 4,
    seed: int = 20260827,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, object]:
    bundle = load_bundle(source_index)
    records = bundle.records if limit is None else bundle.records[:limit]
    output_dir = Path(output_dir)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = []
    source_ids = []
    task_types = []
    token_indices = []
    response_lengths = []
    response_token_ids = []
    embeddings = []

    with torch.no_grad():
        for number, record in enumerate(
            tqdm(records, desc=f"encode information flow ({mode})", unit="graph")
        ):
            graph = record.load()
            output = flow_embedding(
                graph,
                mode=mode,
                checkpoints=checkpoints,
                seed=seed,
                device=device,
            )
            response_count = graph.response_count
            save_sample(graph_dir / f"{number:08d}.npz", graph, output)

            sample_ids.append(np.repeat(graph.sample_id, response_count))
            source_ids.append(np.repeat(graph.source_id, response_count))
            task_types.append(np.repeat(str(graph.task_type), response_count))
            token_indices.append(np.arange(response_count, dtype=np.int32))
            response_lengths.append(
                np.full(response_count, response_count, dtype=np.int32)
            )
            response_token_ids.append(
                graph.token_ids[graph.response_start :].cpu().numpy().astype(np.int64)
            )
            embeddings.append(output.embedding.cpu().numpy().astype(np.float32))

    index_path = output_dir / "index.npz"
    np.savez_compressed(
        index_path,
        sample_id=np.concatenate(sample_ids),
        source_id=np.concatenate(source_ids),
        task_type=np.concatenate(task_types),
        token_index=np.concatenate(token_indices),
        response_length=np.concatenate(response_lengths),
        response_token_id=np.concatenate(response_token_ids),
        embedding=np.concatenate(embeddings),
        method=np.asarray("gcn_information_flow_sketch"),
        head_mode=np.asarray(mode),
        checkpoints=np.asarray(checkpoints, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int64),
        labels_included=np.asarray(False),
        source_index=np.asarray(str(Path(source_index).resolve())),
    )
    return {
        "index": str(index_path.resolve()),
        "samples": len(records),
        "nodes": int(sum(len(block) for block in embeddings)),
        "dimension": int(embeddings[0].shape[1]),
        "head_mode": mode,
        "labels_read": False,
    }
