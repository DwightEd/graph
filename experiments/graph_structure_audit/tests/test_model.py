import torch

from experiments.graph_structure_audit.graph_data import build_multiplex_graph
from experiments.graph_structure_audit.masking import mask_graph
from experiments.graph_structure_audit.model import LayeredGraphRecovery
from .helpers import raw_graph, tiny_config


def test_layered_messages_and_gradients_are_active():
    raw, _ = raw_graph()
    graph = build_multiplex_graph(raw)
    config = tiny_config()
    model = LayeredGraphRecovery(
        num_layers=graph.num_layers,
        num_heads=graph.num_heads,
        config=config,
    )
    generator = torch.Generator().manual_seed(7)
    masked = mask_graph(graph, config, generator=generator)
    full = model(graph, masked, message_passing=True)
    isolated = model(graph, masked, message_passing=False)
    full.loss.backward()

    assert full.embedding.shape == (graph.num_response_tokens, config.hidden_dim)
    assert torch.isfinite(full.token_loss).all()
    assert not torch.allclose(full.embedding, isolated.embedding)
    assert sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    ) > 0
