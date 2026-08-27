from types import SimpleNamespace

import numpy as np
import torch

from experiments.grounded_route.artifacts import (
    EncodedTokenGraph,
    load_embedding_index,
    load_encoded_graph,
    merge_embedding_index,
    save_embedding_index,
    save_encoded_graph,
)
from experiments.grounded_route.evaluation.data import EmbeddingTable
from experiments.grounded_route.tests.helpers import make_graph


def test_arbitrary_export_dimension_survives_graph_and_index_round_trip(tmp_path):
    graph = make_graph()
    dimension = 128
    embedding = torch.arange(
        graph.token_count * dimension,
        dtype=torch.float32,
    ).reshape(graph.token_count, dimension)
    lineage = torch.softmax(
        torch.randn(
            graph.response_count,
            graph.layer_count,
            graph.head_count,
            3,
        ),
        dim=-1,
    )
    encoded = EncodedTokenGraph.from_output(
        graph,
        SimpleNamespace(node_embedding=embedding, lineage=lineage),
    )

    graph_path = tmp_path / "graph.pt"
    save_encoded_graph(graph_path, encoded)
    restored_graph = load_encoded_graph(graph_path)
    assert restored_graph.node_embedding.shape == (graph.token_count, dimension)
    assert torch.equal(restored_graph.node_embedding, embedding)

    index = merge_embedding_index([restored_graph])
    index_path = tmp_path / "index.npz"
    save_embedding_index(index_path, index, scope="all")
    restored_index, _ = load_embedding_index(index_path)
    evaluation_table = EmbeddingTable.load(index_path)

    assert restored_index.embedding.shape == (graph.response_count, dimension)
    assert evaluation_table.embedding.shape == (graph.response_count, dimension)
    np.testing.assert_array_equal(restored_index.embedding, index.embedding)
    np.testing.assert_array_equal(evaluation_table.embedding, index.embedding)
