import numpy as np

from experiments.attention_holonomy_audit.config import GraphConfig, TransportConfig
from experiments.attention_holonomy_audit.features import compute_mechanism_audit
from experiments.attention_holonomy_audit.graph import build_attention_event_graph
from experiments.attention_holonomy_audit.transport import TransportFitter
from experiments.attention_holonomy_audit.tests.helpers import make_sample


def test_transport_audit_emits_finite_structure_features():
    graph_config = GraphConfig(max_relay_predecessors=8, max_query_events=16)
    transport_config = TransportConfig(ridge_alpha=1e-3, minimum_pairs=1)
    fitter = TransportFitter(
        3,
        2,
        graph_config=graph_config,
        transport_config=transport_config,
    )
    for index in range(6):
        fitter.update(
            build_attention_event_graph(
                make_sample(f"train-{index}", f"g{index}", shift=0.01 * index),
                config=graph_config,
            )
        )
    reference = fitter.freeze()
    graph = build_attention_event_graph(make_sample("test", "gt"), config=graph_config)
    audit = compute_mechanism_audit(graph, reference, seed=7)
    assert audit.primary.shape == (4, 6)
    assert audit.primary_maps.shape == (4, 3, 6)
    assert np.isfinite(audit.primary[:, :5]).any(axis=0).all()
    assert np.isfinite(audit.controls[:, :4]).any(axis=0).all()
