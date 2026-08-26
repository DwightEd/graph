"""Export the GCN node representations as one compact evaluation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.grounded_route.evaluation.data import EmbeddingTable, load_labels


def export_node_data(
    calibration_index: str | Path,
    test_index: str | Path,
    calibration_split: str | Path,
    test_split: str | Path,
    output: str | Path,
) -> dict[str, int]:
    """Combine frozen node embeddings and aligned binary labels in one NPZ."""

    calibration = EmbeddingTable.load(calibration_index)
    test = EmbeddingTable.load(test_index)
    calibration_labels = load_labels(calibration, str(calibration_split)).astype(
        np.int8
    )
    test_labels = load_labels(test, str(test_split)).astype(np.int8)
    embeddings = np.concatenate((calibration.embedding, test.embedding)).astype(
        np.float32
    )
    labels = np.concatenate((calibration_labels, test_labels)).astype(np.int8)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        node_embeddings=embeddings,
        node_labels=labels,
    )
    return {
        "nodes": len(embeddings),
        "positive_nodes": int(labels.sum()),
        "embedding_dim": int(embeddings.shape[1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack GCN node embeddings and aligned binary node labels."
    )
    parser.add_argument("--calibration-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--calibration-split", required=True)
    parser.add_argument("--test-split", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_node_data(
        args.calibration_index,
        args.test_index,
        args.calibration_split,
        args.test_split,
        args.output,
    )
    print(f"created: {Path(args.output).resolve()}")
    for name, value in report.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
