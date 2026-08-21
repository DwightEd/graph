import torch

from experiments.source_reuse_contrast.data import collect_source_reuse_graph
from experiments.source_reuse_contrast.grounding_model import GroundingSensitiveGraphModel
from experiments.source_reuse_contrast.provenance import compute_grounding_targets

from .grounding_helpers import sequence_sample, tiny_config


def test_prompt_provenance_reaches_response_relays():
    graph = collect_source_reuse_graph(sequence_sample())
    target = compute_grounding_targets(graph, received_topk=2)
    assert target.provenance.shape == (4, 3)
    assert torch.all(target.provenance[:, 1:] >= 0)
    assert target.provenance[1, -1] > 0
    response_edges = graph.source >= graph.response_idx
    assert torch.all(
        (target.edge_origin[response_edges] >= 0)
        & (target.edge_origin[response_edges] <= 1)
    )


def test_grounding_model_has_finite_counterfactual_scores_and_gradients():
    graph = collect_source_reuse_graph(sequence_sample())
    model = GroundingSensitiveGraphModel(
        num_layers=graph.num_layers,
        num_heads=graph.num_heads,
        config=tiny_config(),
    )
    output = model(graph, seed=11)
    output.loss.backward()
    assert output.embedding.shape == (4, 16)
    assert output.closure.shape == (4,)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.reconstruction).all()
    assert torch.isfinite(output.closure).all()
    assert torch.all((output.gate_mean >= 0) & (output.gate_mean <= 1))
    assert sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    ) > 0
