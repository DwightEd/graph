import csv

import numpy as np
import pytest

from experiments.dbgnn_reference.export_node_mapping import export_node_mapping


def save_index(path, embeddings, sample_ids, token_indices):
    rows = len(embeddings)
    np.savez_compressed(
        path,
        embedding=np.asarray(embeddings, dtype=np.float32),
        sample_id=np.asarray(sample_ids),
        source_id=np.asarray([f"source-{sample}" for sample in sample_ids]),
        task_type=np.asarray(["QA"] * rows),
        token_index=np.asarray(token_indices, dtype=np.int32),
        response_length=np.asarray([2] * rows, dtype=np.int32),
        response_token_id=np.arange(10, 10 + rows, dtype=np.int64),
    )


def test_mapping_preserves_compact_row_order_without_embeddings(tmp_path):
    calibration = tmp_path / "calibration.npz"
    test = tmp_path / "test.npz"
    bundle = tmp_path / "bundle.npz"
    output = tmp_path / "rows.csv"
    save_index(calibration, [[1, 2], [3, 4]], ["c", "c"], [0, 1])
    save_index(test, [[7, 8], [5, 6]], ["b", "a"], [0, 0])
    np.savez_compressed(
        bundle,
        node_embeddings=np.asarray([[1, 2], [3, 4], [7, 8], [5, 6]], np.float32),
        node_labels=np.asarray([0, 1, 0, 1], np.int8),
    )

    report = export_node_mapping(bundle, calibration, test, output)

    with output.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [(row["row_index"], row["split"], row["sample_id"]) for row in rows] == [
        ("0", "calibration", "c"),
        ("1", "calibration", "c"),
        ("2", "test", "b"),
        ("3", "test", "a"),
    ]
    assert [row["token_index"] for row in rows] == ["0", "1", "0", "0"]
    assert [row["response_token_id"] for row in rows] == ["10", "11", "10", "11"]
    assert "embedding" not in rows[0]
    assert report == {
        "rows": 4,
        "calibration_rows": 2,
        "test_rows": 2,
        "samples": 3,
    }


def test_mapping_rejects_a_bundle_with_different_row_order(tmp_path):
    calibration = tmp_path / "calibration.npz"
    test = tmp_path / "test.npz"
    bundle = tmp_path / "bundle.npz"
    save_index(calibration, [[1, 2]], ["c"], [0])
    save_index(test, [[3, 4]], ["t"], [0])
    np.savez_compressed(
        bundle,
        node_embeddings=np.asarray([[3, 4], [1, 2]], np.float32),
        node_labels=np.asarray([0, 0], np.int8),
    )

    with pytest.raises(ValueError, match="order"):
        export_node_mapping(bundle, calibration, test, tmp_path / "rows.csv")
