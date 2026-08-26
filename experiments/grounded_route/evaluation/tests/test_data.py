import numpy as np

from experiments.grounded_route.evaluation.data import EmbeddingTable, align_table


def table(order):
    return EmbeddingTable(
        sample_id=np.asarray(["a", "a", "b"])[order],
        source_id=np.asarray(["x", "x", "y"])[order],
        token_index=np.asarray([0, 1, 0], dtype=np.int32)[order],
        response_length=np.asarray([2, 2, 1], dtype=np.int32)[order],
        response_token_id=np.asarray([10, 11, 12], dtype=np.int64)[order],
        embedding=np.eye(3, dtype=np.float32)[order],
    )


def test_align_table_restores_reference_token_order():
    reference = table(np.asarray([0, 1, 2]))
    candidate = table(np.asarray([2, 0, 1]))
    aligned = align_table(reference, candidate)
    assert np.array_equal(aligned.sample_id, reference.sample_id)
    assert np.array_equal(aligned.token_index, reference.token_index)
