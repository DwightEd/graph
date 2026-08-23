import numpy as np
import torch

from experiments.non_neural_structure_audit.features import (
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
