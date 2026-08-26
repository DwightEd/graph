import pytest
import torch

from experiments.dbgnn_reference.graph import build_dbgnn_graph
from experiments.dbgnn_reference.tests.test_graph import encoded_graph
from experiments.dbgnn_reference.upstream import (
    LinkPredictionModel,
    OfficialNodeEncoder,
)


@pytest.mark.parametrize("encoder_name", ("gcn", "dbgnn"))
def test_copied_encoder_exports_node_embeddings_and_backpropagates(encoder_name):
    graph = build_dbgnn_graph(encoded_graph())
    encoder = OfficialNodeEncoder(
        encoder=encoder_name,
        first_order_dim=graph.x_fo.shape[1],
        higher_order_dim=graph.x_ho.shape[1],
        hidden_dim=8,
        embedding_dim=6,
        dropout=0.0,
    )
    model = LinkPredictionModel(encoder, embedding_dim=6)

    embedding = model.encode(graph)
    score = model.edge_score(
        embedding,
        graph.edge_index_fo[0],
        graph.edge_index_fo[1],
    )

    assert embedding.shape == (graph.num_nodes, 6)
    assert score.shape == (graph.edge_index_fo.shape[1],)
    (embedding.square().mean() + score.square().mean()).backward()
    gradients = [value.grad for value in model.parameters() if value.requires_grad]
    assert any(value is not None for value in gradients)
    assert all(value is None or torch.isfinite(value).all() for value in gradients)
