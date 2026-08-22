"""Small artifact helpers for graph recovery experiments."""

import json
from pathlib import Path

import numpy as np

CHECKPOINT_SCHEMA = "multiplex-graph-recovery-checkpoint-v1"
SCORE_SCHEMA = "multiplex-graph-recovery-scores-v1"
EVALUATION_SCHEMA = "multiplex-graph-recovery-evaluation-v1"
GRAPH_SCHEMA = "multiplex-attention-graph-v1"


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}
