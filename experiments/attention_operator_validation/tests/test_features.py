import torch

from experiments.attention_operator_validation.features import extract_answer_features
from experiments.attention_operator_validation.operators import geometry_from_factors
from experiments.attention_operator_validation.pair_codes import build_pair_code_field
from experiments.grounded_route.tests.helpers import make_graph


def test_answer_features_include_mass_operator_and_permutation_controls():
    torch.manual_seed(19)
    graph = make_graph(layers=3, heads=4, response_count=6)
    output = [torch.randn(graph.head_count, 8, 2) for _ in range(graph.layer_count)]
    value = [torch.randn(graph.head_count, 2, 8) for _ in range(graph.layer_count)]
    geometry = geometry_from_factors(output, value)
    field = build_pair_code_field(graph)

    feature = extract_answer_features(graph, field, geometry, seed=23)

    assert "prompt_mass_mean" in feature
    assert "identity_history_dispersion_mean" in feature
    assert "operator_normalized_history_dispersion_mean" in feature
    assert "operator_permuted_response_operator_lockin" in feature
    assert feature["row_mass_conservation_error"] < 1e-5
    assert feature["prompt_code_effective_heads_mean"] > 0.0
    assert feature["history_code_effective_heads_mean"] > 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in feature.values())
