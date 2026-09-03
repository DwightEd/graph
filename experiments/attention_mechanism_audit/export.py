"""Export response-node embeddings without opening hallucination labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def export_nodes(state_root: str | Path, output: str | Path, task: str | None = None) -> Path:
    root = Path(state_root)
    rows = [
        json.loads(line)
        for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if task is not None:
        rows = [row for row in rows if row["task_type"].casefold() == task.casefold()]

    embedding, sample_id, source_id, position, target_id = [], [], [], [], []
    for row in rows:
        graph = torch.load(root / row["path"], map_location="cpu", weights_only=True)
        value = graph["node_embedding"].float().numpy()
        count = len(value)
        embedding.append(value)
        sample_id.extend([row["sample_id"]] * count)
        source_id.extend([row["source_id"]] * count)
        position.extend(range(count))
        start = int(graph["response_start"])
        target_id.extend(graph["token_ids"][start:].tolist())

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embedding=np.concatenate(embedding) if embedding else np.empty((0, 0)),
        sample_id=np.asarray(sample_id),
        source_id=np.asarray(source_id),
        response_position=np.asarray(position, dtype=np.int32),
        target_token_id=np.asarray(target_id, dtype=np.int32),
    )
    return path
