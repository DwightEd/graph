import numpy as np
import torch

from experiments.non_neural_structure_audit.features import (
    DYNAMICS_FEATURE_NAMES,
    FEATURE_INDEX,
    RELATION_NAMES,
    build_layer_features,
    relation_scores,
)
from experiments.non_neural_structure_audit.lineage import propagate_lineage

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
        token[:, FEATURE_INDEX["interaction_diagonal_transition_gap"]],
        [0.0, 0.1, -0.25],
        atol=1e-6,
    )
    assert set(DYNAMICS_FEATURE_NAMES) <= set(FEATURE_INDEX)
