import numpy as np
import torch

from experiments.attention_mechanism_audit.capture import ROLE_NAMES, SELF
from experiments.attention_mechanism_audit.data import EVIDENCE
from experiments.attention_mechanism_audit.evaluate import _mean_or_none, token_metrics


def test_missing_single_sample_class_is_serialized_as_null():
    assert _mean_or_none(np.asarray([], dtype=np.float32)) is None
    assert _mean_or_none(np.asarray([1.0, 3.0])) == 2.0


def test_token_metrics_keep_layer_drift_and_separate_causal_effects():
    layers, responses, heads, roles = 3, 3, 2, len(ROLE_NAMES)
    edge = torch.zeros(layers, responses, heads, roles)
    edge[:, 0, :, EVIDENCE] = 1.0
    edge[0, 1, :, EVIDENCE] = 2.0
    edge[1:, 1, :, SELF] = 2.0
    edge[:, 2, :, len(ROLE_NAMES) - 2] = 1.0
    route = torch.zeros_like(edge)
    route[:, 0, :, EVIDENCE] = 1.0
    route[0, 1, :, EVIDENCE] = 1.0
    route[1, 1, :, SELF] = 1.0
    route[2, 1, 0, EVIDENCE] = 1.0
    route[2, 1, 1, SELF] = 1.0
    route[:, 2, :, len(ROLE_NAMES) - 2] = 1.0
    artifact = {
        "trace": {
            "role_edge_magnitude": edge,
            "role_attention": route,
            "source_message_entropy": torch.zeros(layers, responses),
            "message_coherence": torch.ones(layers, responses),
            "source_role": torch.tensor(
                [
                    [EVIDENCE, SELF, -1, -1],
                    [EVIDENCE, 1, SELF, -1],
                    [EVIDENCE, 1, len(ROLE_NAMES) - 2, SELF],
                ],
                dtype=torch.int8,
            ),
        },
        "mechanism": {
            "evidence_message_effect": torch.tensor([0.25, -0.5, 0.1]),
            "response_message_effect": torch.tensor([0.1, 0.8, 0.4]),
            "evidence_response_removed_margin": torch.tensor([-0.1, 0.2, 0.2]),
            "full_margin": torch.tensor([-0.2, 0.3, -0.1]),
        },
    }

    metrics = token_metrics(artifact)

    np.testing.assert_allclose(
        metrics["message_evidence_share_mean"], [1.0, 1 / 3, 0.0]
    )
    np.testing.assert_allclose(
        metrics["message_response_share_mean"], [0.0, 2 / 3, 1.0]
    )
    np.testing.assert_allclose(
        metrics["message_routing_drift_mean"], [-1.0, 1 / 3, 1.0]
    )
    np.testing.assert_allclose(
        metrics["message_routing_drift_layer_shift"], [0.0, 2.0, 0.0]
    )
    np.testing.assert_allclose(
        metrics["attention_routing_drift_mean"], [-1.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(
        metrics["attention_routing_drift_layer_shift"], [0.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(metrics["message_source_dispersion_mean"], 0.0)
    np.testing.assert_allclose(
        metrics["head_role_disagreement_mean"], [0.0, np.log(2) / 3, 0.0]
    )
    np.testing.assert_allclose(
        metrics["head_role_disagreement_layer_shift"], [0.0, np.log(2), 0.0]
    )
    np.testing.assert_allclose(metrics["message_coherence_mean"], 1.0)
    np.testing.assert_allclose(
        metrics["evidence_message_effect"], [0.25, -0.5, 0.1]
    )
    np.testing.assert_allclose(
        metrics["message_independent_capture_signature"], [0.0, 1.0, 0.0]
    )
