import numpy as np
import torch

from experiments.non_neural_structure_audit.features import (
    DYNAMICS_FEATURE_NAMES,
    FEATURE_INDEX,
    LINEAGE_FEATURE_NAMES,
    RELATION_NAMES,
    _head_disagreement,
    build_layer_features,
    layer_order_relation_scores,
    relation_scores,
    replace_layer_order_features,
    replace_lineage_features,
)
from experiments.non_neural_structure_audit.lineage import (
    LineageOperator,
    propagate_lineage,
)

from .helpers import routing_state


def test_features_keep_prompt_and_response_base_paths_separate():
    routing = routing_state(
        layers=[0, 1],
        heads=[0, 0],
        queries=[0, 1],
        sources=[0, 1],
        weights=[1.0, 1.0],
        diagonal=torch.zeros((2, 2, 1)),
    )

    features = build_layer_features(routing, propagate_lineage(routing))

    assert features[1, 1, FEATURE_INDEX["prompt_connected_relay"]] == 1.0
    assert features[1, 1, FEATURE_INDEX["inherited_response_base"]] == 0.0
    assert features.shape[-1] == len(FEATURE_INDEX)


def test_structure_features_skip_unused_exact_source_velocity(monkeypatch):
    from experiments.attention_phenomenology import sources

    def unexpected_velocity(*args, **kwargs):
        raise AssertionError("structure audit does not consume exact-source velocity")

    monkeypatch.setattr(sources, "_adjacent_velocity", unexpected_velocity)
    routing = routing_state(
        layers=[0, 1],
        heads=[0, 0],
        queries=[0, 1],
        sources=[0, 1],
        weights=[1.0, 1.0],
        diagonal=torch.zeros((2, 2, 1)),
    )

    features = build_layer_features(routing, propagate_lineage(routing))

    assert features.shape[-1] == len(FEATURE_INDEX)


def test_endpoint_control_reuses_routing_features_and_replaces_only_lineage():
    routing = routing_state(
        layers=[0, 1],
        heads=[0, 0],
        queries=[0, 1],
        sources=[0, 1],
        weights=[1.0, 1.0],
        diagonal=torch.zeros((2, 2, 1)),
    )
    operator = LineageOperator(routing)
    base = build_layer_features(routing, operator.run())
    controlled_lineage = operator.run(source=torch.tensor([0, 0]))

    replaced = replace_lineage_features(base, controlled_lineage)
    rebuilt = build_layer_features(routing, controlled_lineage)

    torch.testing.assert_close(replaced, rebuilt)
    assert not torch.equal(
        replaced[..., FEATURE_INDEX["prompt_connected_total"]],
        base[..., FEATURE_INDEX["prompt_connected_total"]],
    )


def test_relation_scores_are_single_oriented_coordinates_not_a_learned_fusion():
    standardized = np.zeros((3, 2, len(FEATURE_INDEX)), dtype=np.float32)
    standardized[:, :, FEATURE_INDEX["prompt_connected_total"]] = 2.0
    standardized[:, :, FEATURE_INDEX["inherited_response_base"]] = 3.0

    scores = relation_scores(standardized)

    assert scores.shape == (3, len(RELATION_NAMES))
    np.testing.assert_allclose(
        scores[:, RELATION_NAMES.index("prompt_connected_lineage")], -2.0
    )
    np.testing.assert_allclose(
        scores[:, RELATION_NAMES.index("inherited_response_base")], 3.0
    )


def test_cross_layer_dynamics_are_observed_role_transition_magnitudes():
    diagonal = torch.zeros((2, 3, 1))
    diagonal[1, :, 0] = torch.tensor([0.3, 0.1, 0.6])
    routing = routing_state(
        layers=[0, 0, 1, 1, 2, 2],
        heads=[0] * 6,
        queries=[1] * 6,
        sources=[0, 1] * 3,
        weights=[0.2, 0.1, 0.5, 0.4, 0.3, 0.1],
        diagonal=diagonal,
        response_idx=1,
        response_tokens=2,
        num_layers=3,
    )

    features = build_layer_features(routing, propagate_lineage(routing))
    token = features[1]

    np.testing.assert_allclose(
        token[:, FEATURE_INDEX["prompt_transition_magnitude"]],
        [0.0, 0.3, 0.2],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        token[:, FEATURE_INDEX["history_transition_magnitude"]],
        [0.0, 0.3, 0.3],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        token[:, FEATURE_INDEX["diagonal_transition_magnitude"]],
        [0.0, 0.2, 0.5],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        token[:, FEATURE_INDEX["origin_transition_gap"]],
        [0.0, 0.0, 0.1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        token[:, FEATURE_INDEX["offdiagonal_diagonal_transition_gap"]],
        [0.0, 0.1, -0.25],
        atol=1e-6,
    )
    assert set(DYNAMICS_FEATURE_NAMES) <= set(FEATURE_INDEX)


def test_layer_order_replacement_changes_only_lineage_and_transition_fields():
    diagonal = torch.zeros((2, 3, 1))
    routing = routing_state(
        layers=[0, 0, 1, 1, 2, 2],
        heads=[0] * 6,
        queries=[1] * 6,
        sources=[0, 1] * 3,
        weights=[0.2, 0.1, 0.5, 0.4, 0.3, 0.1],
        diagonal=diagonal,
        response_idx=1,
        response_tokens=2,
        num_layers=3,
    )
    operator = LineageOperator(routing)
    real = build_layer_features(routing, operator.run())
    order = (2, 0, 1)

    replaced = replace_layer_order_features(
        real, routing, operator.run(layer_order=order)
    )
    rebuilt = build_layer_features(routing, operator.run(layer_order=order))

    for name in DYNAMICS_FEATURE_NAMES:
        torch.testing.assert_close(
            replaced[..., FEATURE_INDEX[name]], rebuilt[..., FEATURE_INDEX[name]]
        )
    untouched = [
        index
        for name, index in FEATURE_INDEX.items()
        if name not in DYNAMICS_FEATURE_NAMES and name not in LINEAGE_FEATURE_NAMES
    ]
    torch.testing.assert_close(replaced[..., untouched], real[..., untouched])


def test_head_disagreement_matches_pairwise_definition_without_head_square_output():
    routing = routing_state(
        layers=[0, 0, 0],
        heads=[0, 1, 2],
        queries=[1, 1, 1],
        sources=[0, 1, 0],
        weights=[0.8, 0.6, 0.2],
        diagonal=torch.zeros((2, 1, 3)),
        response_idx=1,
        response_tokens=2,
        num_layers=1,
        num_heads=3,
    )

    actual = _head_disagreement(routing, 1e-8)
    mass = torch.stack((routing.prompt_mass, routing.response_mass), dim=-1)
    total = mass.sum(dim=-1)
    valid = total > 1e-8
    root = (mass / total[..., None].clamp_min(1e-8)).sqrt()
    affinity = (root.unsqueeze(-2) * root.unsqueeze(-3)).sum(dim=-1)
    distance = (1.0 - affinity.clamp(0.0, 1.0)).sqrt()
    pairs = torch.triu(torch.ones((3, 3), dtype=torch.bool), diagonal=1)
    selected = valid.unsqueeze(-1) & valid.unsqueeze(-2) & pairs
    expected = torch.where(
        selected.sum(dim=(-2, -1)) > 0,
        torch.where(selected, distance, 0.0).sum(dim=(-2, -1))
        / selected.sum(dim=(-2, -1)).clamp_min(1),
        0.0,
    )

    torch.testing.assert_close(actual, expected)


def test_layer_order_statistic_uses_full_transition_but_final_lineage_state():
    standardized = np.zeros((1, 3, len(FEATURE_INDEX)), dtype=np.float32)
    standardized[0, :, FEATURE_INDEX["prompt_transition_magnitude"]] = [0, 1, 2]
    standardized[0, :, FEATURE_INDEX["prompt_connected_total"]] = [1, 2, 9]

    scores = layer_order_relation_scores(standardized)

    assert scores[0, RELATION_NAMES.index("prompt_transition_volatility")] == 1.0
    assert scores[0, RELATION_NAMES.index("prompt_connected_lineage")] == -9.0
