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


def test_export_contains_only_embeddings_and_aligned_test_labels(
    tmp_path, monkeypatch
):
    calibration = tmp_path / "calibration.npz"
    test = tmp_path / "test.npz"
    output = tmp_path / "node_data.npz"
    save_index(calibration, [[1, 2], [3, 4]], ["c", "c"], [0, 1])
    save_index(test, [[5, 6], [7, 8]], ["b", "a"], [1, 0])

    def aligned_labels(table, split_root):
        assert split_root == "test-root"
        labels = {("a", 0): 1, ("b", 1): 0}
        return np.asarray(
            [labels[key] for key in zip(table.sample_id, table.token_index)],
            dtype=np.int8,
        )

    monkeypatch.setattr(exporter, "load_labels", aligned_labels)
    report = exporter.export_node_data(
        calibration, test, "test-root", output
    )

    with np.load(output, allow_pickle=False) as data:
        assert set(data.files) == {
            "calibration_embeddings",
            "test_embeddings",
            "test_labels",
        }
        np.testing.assert_array_equal(
            data["calibration_embeddings"], [[1, 2], [3, 4]]
        )
        np.testing.assert_array_equal(data["test_embeddings"], [[5, 6], [7, 8]])
        np.testing.assert_array_equal(data["test_labels"], [0, 1])
        assert data["calibration_embeddings"].dtype == np.float32
        assert data["test_embeddings"].dtype == np.float32
        assert data["test_labels"].dtype == np.int8

    assert report == {
        "calibration_nodes": 2,
        "test_nodes": 2,
        "positive_test_nodes": 1,
        "embedding_dim": 2,
    }
