import numpy as np

from experiments.dbgnn_reference import export_node_data as exporter


def save_index(path, embeddings, sample_ids, token_indices):
    rows = len(embeddings)
    np.savez_compressed(
        path,
        embedding=np.asarray(embeddings, dtype=np.float32),
        sample_id=np.asarray(sample_ids),
        source_id=np.asarray(["source"] * rows),
        token_index=np.asarray(token_indices, dtype=np.int32),
        response_length=np.asarray([2] * rows, dtype=np.int32),
        response_token_id=np.arange(rows, dtype=np.int64),
    )


def test_export_contains_only_embeddings_and_aligned_node_labels(
    tmp_path, monkeypatch
):
    calibration = tmp_path / "calibration.npz"
    test = tmp_path / "test.npz"
    output = tmp_path / "node_data.npz"
    save_index(calibration, [[1, 2], [3, 4]], ["c", "c"], [0, 1])
    save_index(test, [[5, 6], [7, 8]], ["b", "a"], [1, 0])

    def aligned_labels(table, split_root):
        labels = {
            "calibration-root": {("c", 0): 1, ("c", 1): 0},
            "test-root": {("a", 0): 1, ("b", 1): 0},
        }[split_root]
        return np.asarray(
            [labels[key] for key in zip(table.sample_id, table.token_index)],
            dtype=np.int8,
        )

    monkeypatch.setattr(exporter, "load_labels", aligned_labels)
    report = exporter.export_node_data(
        calibration, test, "calibration-root", "test-root", output
    )

    with np.load(output, allow_pickle=False) as data:
        assert set(data.files) == {
            "node_embeddings",
            "node_labels",
        }
        np.testing.assert_array_equal(
            data["node_embeddings"],
            [[1, 2], [3, 4], [5, 6], [7, 8]],
        )
        np.testing.assert_array_equal(data["node_labels"], [1, 0, 0, 1])
        assert data["node_embeddings"].dtype == np.float32
        assert data["node_labels"].dtype == np.int8

    assert report == {
        "nodes": 4,
        "positive_nodes": 2,
        "embedding_dim": 2,
    }
