"""Write the row identities of an existing compact GCN node bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FIELDS = (
    "sample_id",
    "source_id",
    "task_type",
    "token_index",
    "response_length",
    "response_token_id",
)


def load_index(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in ("embedding", *FIELDS)}


def export_node_mapping(
    bundle_path: str | Path,
    calibration_index: str | Path,
    test_index: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Write one CSV row for every compact embedding without copying embeddings."""

    calibration = load_index(calibration_index)
    test = load_index(test_index)
    calibration_rows = len(calibration["embedding"])
    test_rows = len(test["embedding"])

    with np.load(bundle_path, allow_pickle=False) as bundle:
        embeddings = bundle["node_embeddings"]
        labels = bundle["node_labels"]
        if len(embeddings) != calibration_rows + test_rows or len(labels) != len(
            embeddings
        ):
            raise ValueError("compact bundle row count does not match its source indices")
        if not np.array_equal(
            embeddings[:calibration_rows], calibration["embedding"]
        ) or not np.array_equal(embeddings[calibration_rows:], test["embedding"]):
            raise ValueError("compact bundle order does not match calibration then test")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("row_index", "split", "split_row", *FIELDS))
        row_index = 0
        for split, index in (("calibration", calibration), ("test", test)):
            for split_row in range(len(index["embedding"])):
                writer.writerow(
                    (
                        row_index,
                        split,
                        split_row,
                        str(index["sample_id"][split_row]),
                        str(index["source_id"][split_row]),
                        str(index["task_type"][split_row]),
                        int(index["token_index"][split_row]),
                        int(index["response_length"][split_row]),
                        int(index["response_token_id"][split_row]),
                    )
                )
                row_index += 1

    return {
        "rows": row_index,
        "calibration_rows": calibration_rows,
        "test_rows": test_rows,
        "samples": len(
            set(calibration["sample_id"].astype(str))
            | set(test["sample_id"].astype(str))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the exact row mapping of a compact GCN node bundle."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--calibration-index", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = export_node_mapping(
        args.bundle, args.calibration_index, args.test_index, args.output
    )
    print(f"created: {Path(args.output).resolve()}")
    for name, value in report.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
