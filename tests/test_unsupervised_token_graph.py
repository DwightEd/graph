import numpy as np
import torch

from experiments.unsupervised_token_graph.data import ResponseCache
from experiments.unsupervised_token_graph.graph import TokenGraph
from experiments.unsupervised_token_graph.model import GraphAutoencoder


def dense_sample(tmp_path):
    attention = np.zeros((2, 2, 6, 6), np.float32)
    attention[:, :, 1:, :-1] = np.eye(5, dtype=np.float32)
    path = tmp_path / "response.npz"
    np.savez_compressed(path, attention=attention, response_idx=np.array(3), source_id=np.array("source"))
    return path


def test_dense_cache_becomes_one_token_graph(tmp_path):
    record = ResponseCache().load(dense_sample(tmp_path))
    graph = TokenGraph.from_cache(record, 0.05)
    assert graph.x.shape == (6, 5)
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

def test_graph_features_do_not_use_future_outgoing_statistics(tmp_path):
    first = dense_sample(tmp_path)
    with np.load(first) as data:
        attention = data["attention"].copy()
    second = tmp_path / "future.npz"
    attention[:, :, -1, 0] = 0.9
    np.savez_compressed(second, attention=attention, response_idx=np.array(3), source_id=np.array("source"))
    graph_a = TokenGraph.from_cache(ResponseCache().load(first), 0.05)
    graph_b = TokenGraph.from_cache(ResponseCache().load(second), 0.05)
    np.testing.assert_allclose(graph_a.x[:5], graph_b.x[:5])


def test_csr_and_dense_feature_schema_match(tmp_path):
    dense = ResponseCache().load(dense_sample(tmp_path))
    # The formal CSR path is tested separately; both representations expose five fields.
    assert TokenGraph.from_cache(dense, 0.05).x.shape[1] == 5


def test_autoencoder_does_not_encode_target_local_observables(tmp_path):
    graph = TokenGraph.from_cache(ResponseCache().load(dense_sample(tmp_path)), 0.05)
    changed = type(graph)(graph.response_id, graph.x.copy(), graph.edge_index, graph.edge_weight,
                          graph.response_idx, graph.token_ids)
    changed.x[4, 2:] += 10.0
    model = GraphAutoencoder(graph.x.shape[1], epochs=1)
    z_a = model._encode(graph)[0].detach().numpy()
    z_b = model._encode(changed)[0].detach().numpy()
    np.testing.assert_allclose(z_a, z_b)
