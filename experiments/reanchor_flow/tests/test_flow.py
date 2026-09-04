import numpy as np

from experiments.reanchor_flow.claims import ClaimSpan
from experiments.reanchor_flow.flow import claim_metrics, dominant_backbone
from experiments.reanchor_flow.graph import (
    build_token_dag,
    matched_endpoint_mask,
    rewire_by_role_lag,
    token_edges_to_query_mask,
)
from experiments.reanchor_flow.potential import conditioned_flow, first_hit_path_mass


def chain_graph():
    graph = np.zeros((6, 6), dtype=float)
    graph[0, 3] = 1.0
    graph[3, 4] = 1.0
    graph[4, 5] = 1.0
    return graph


def test_first_hit_path_mass_counts_a_multi_anchor_path_once():
    result = first_hit_path_mass(chain_graph(), 5, [0], [3, 4])
    assert result.total == 1.0
    assert result.through_anchor == 1.0
    assert result.closure == 1.0


def test_first_hit_path_mass_detects_direct_bypass():
    graph = chain_graph()
    graph[0, 5] = 1.0
    result = first_hit_path_mass(graph, 5, [0], [3])
    assert result.total == 2.0
    assert result.through_anchor == 1.0
    assert result.closure == 0.5


def test_conditioned_flow_recovers_the_chain_backbone():
    edge, node = conditioned_flow(chain_graph(), 5, [0])
    assert edge[0, 3] == 1.0
    assert edge[3, 4] == 1.0
    assert edge[4, 5] == 1.0
    assert node[5] == 1.0


def test_query_rows_are_lifted_to_predicted_tokens():
    rows = np.zeros((3, 5), dtype=float)
    rows[0, 0] = 2.0
    rows[1, 3] = 4.0
    rows[2, 4] = 8.0
    dag = build_token_dag(rows, response_start=3)
    assert dag.transition.shape == (6, 6)
    assert dag.transition[0, 3] == 1.0
    assert dag.transition[3, 4] == 1.0
    assert dag.transition[4, 5] == 1.0
    assert not np.tril(dag.transition).any()


def test_claim_flow_is_global_not_direct_sink_mass():
    evidence = np.array([True, False, False, False, False, False])
    result = claim_metrics(
        chain_graph(),
        ClaimSpan(3, 6),
        response_start=3,
        evidence_mask=evidence,
        anchor_width=1,
    )
    assert result["evidence_reanchor_flow"] == 1.0
    assert result["evidence_closure"] == 1.0
    assert result["direct_evidence_sink"] == 0.0
    assert result["anchor_throughput"] == 1.0


def test_role_lag_rewire_preserves_target_totals_and_weight_multisets():
    graph = np.zeros((8, 8), dtype=float)
    graph[0, 6] = 0.1
    graph[1, 6] = 0.2
    graph[2, 6] = 0.3
    graph[3, 6] = 0.4
    evidence = np.array([True, True, True, True, False, False, False, False])
    rewired = rewire_by_role_lag(graph, 4, evidence, seed=3)
    np.testing.assert_allclose(rewired.sum(axis=0), graph.sum(axis=0))
    np.testing.assert_allclose(np.sort(rewired[:4, 6]), np.sort(graph[:4, 6]))


def test_backbone_maps_back_to_the_predictor_query():
    selected, edge_flow = dominant_backbone(chain_graph(), 5, [0])
    assert selected.sum() == 3
    assert edge_flow[selected].sum() == 3.0
    query_mask = token_edges_to_query_mask(selected)
    assert query_mask[2, 0]  # 0 -> predicted token 3 uses query 2
    assert query_mask[3, 3]  # 3 -> predicted token 4 uses query 3
    assert query_mask[4, 4]  # 4 -> predicted token 5 uses query 4


def test_matched_endpoint_control_changes_source_identity():
    graph = np.zeros((7, 7), dtype=float)
    graph[0, 5] = 0.8
    graph[1, 5] = 0.7
    graph[2, 5] = 0.2
    graph[3, 5] = 0.1
    selected = np.zeros_like(graph, dtype=bool)
    selected[0, 5] = True
    evidence = np.array([True, True, True, True, False, False, False])
    matched = matched_endpoint_mask(selected, graph, 4, evidence)
    assert matched.sum() == 1
    assert not matched[0, 5]
