import numpy as np

from experiments.unsupervised_token_graph.data import ResponseCache
from experiments.unsupervised_token_graph.graph import TokenGraph


def dense_sample(tmp_path):
    attention = np.zeros((2, 2, 6, 6), np.float32)
    attention[:, :, 1:, :-1] = np.eye(5, dtype=np.float32)
    path = tmp_path / "response.npz"
    np.savez_compressed(path, attention=attention, response_idx=np.array(3), source_id=np.array("source"))
    return path


def test_dense_cache_becomes_one_token_graph(tmp_path):
    record = ResponseCache().load(dense_sample(tmp_path))
    graph = TokenGraph.from_cache(record, 0.05)
    assert graph.x.shape == (6, 7)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == 5


def test_csr_cache_stays_compressed_until_graph_aggregation(tmp_path):
    path = tmp_path / "response.npz"
    diagonal = np.zeros((2, 2, 6), np.float32)
    row_ptr = np.arange(1, 13, dtype=np.int64)
    columns = np.tile(np.arange(1, 6, dtype=np.int32), 2)
    values = np.ones(10, np.float32) * 0.1
    row_ptr = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.int64)
    columns = np.arange(1, 10, dtype=np.int32) % 5
    values = np.ones(9, np.float32) * 0.2
    np.savez_compressed(path, attention_diagonal=diagonal, response_row_ptr=row_ptr,
                        response_column_indices=columns, response_values=values,
                        response_idx=np.array(3), source_id=np.array("source"))
    record = ResponseCache().load(path)
    assert record.attention is None and record.sparse is not None
    graph = TokenGraph.from_cache(record, 0.05)
    assert graph.edge_index.shape[0] == 2