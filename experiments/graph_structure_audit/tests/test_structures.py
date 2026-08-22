import numpy as np

from experiments.graph_structure_audit.config import GraphAuditConfig
from experiments.graph_structure_audit.structures import (
    RECOVERY_METRICS,
    STRUCTURAL_METRICS,
    audit_graph,
)

from .helpers import synthetic_graph


def metric(result, name):
    return result.structural[:, STRUCTURAL_METRICS.index(name)]


def recovery(result, name):
    return result.recovery[:, RECOVERY_METRICS.index(name)]


def test_prompt_relay_and_reuse_motifs_are_prefix_causal():
    result = audit_graph(
        synthetic_graph(),
        GraphAuditConfig(
            minimum_sources_for_recovery=2,
            minimum_channels_for_recovery=2,
            coalition_top_sources=6,
            prompt_bins=4,
        ),
    )
    reachability = metric(result, "prompt_reachability")
    reuse_degree = metric(result, "reuse_degree_mean")
    co_use = metric(result, "coalition_co_use_strength")
    assert reachability[0] > 0
    assert reachability[1] > 0
    assert reuse_degree[0] == 0
    assert reuse_degree[-1] > reuse_degree[1]
    assert co_use[-1] > 0


def test_layer_head_and_recovery_outputs_are_finite_when_defined():
    result = audit_graph(
        synthetic_graph(),
        GraphAuditConfig(
            minimum_sources_for_recovery=2,
            minimum_channels_for_recovery=2,
            source_mask_fraction=0.34,
            channel_mask_fraction=0.34,
            prompt_bins=4,
        ),
    )
    assert result.structural.shape == (4, len(STRUCTURAL_METRICS))
    assert result.recovery.shape == (4, len(RECOVERY_METRICS))
    assert np.isfinite(result.structural).all()
    endpoint = recovery(result, "endpoint_mrr")
    channel = recovery(result, "channel_mrr")
    assert np.isfinite(endpoint[result.valid_recovery]).any() or np.isfinite(channel[result.valid_recovery]).any()
    assert np.nanmin(recovery(result, "endpoint_recovery_error")) >= 0
    assert np.nanmax(recovery(result, "endpoint_recovery_error")) <= 1
