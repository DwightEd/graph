import gc
import weakref
from dataclasses import fields, replace

import torch

from experiments.evidence_route_state.lineage import LineageTracker, propagate_lineage

from .helpers import route_row, two_layer_history


def test_response_history_can_be_an_evidence_rooted_relay():
    rows, roots = two_layer_history(grounded=True)
    lineage = propagate_lineage(rows, roots, response_start=3, evidence_unit_count=1)

    assert lineage.query_position.tolist() == [2, 3, 4]
    assert lineage.prediction_position.tolist() == [3, 4, 5]
    assert lineage.history_valid.tolist() == [False, False, True]

    # q=3 acquired evidence at layer 1; at layer 2 it is a strict response
    # source for q=4. No direct prompt edge is needed at the final hop.
    torch.testing.assert_close(lineage.ancestry[2, 4, 1], torch.tensor(1.0))
    torch.testing.assert_close(
        lineage.grounded_response_relay[1, 2, 0], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        lineage.unrooted_response_feedback[1, 2, 0], torch.tensor(0.0)
    )


def test_one_hop_control_cannot_inherit_evidence_through_response_history():
    rows, roots = two_layer_history(grounded=True)
    tracker = LineageTracker(
        roots,
        response_start=3,
        evidence_unit_count=1,
        layer_count=2,
        head_count=1,
        multi_hop=False,
    )
    tracker.add_many(rows[:3])
    tracker.add_many(rows[3:])
    lineage = tracker.finish()

    torch.testing.assert_close(
        lineage.grounded_response_relay[1, 2, 0], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        lineage.unrooted_response_feedback[1, 2, 0], torch.tensor(1.0)
    )


def test_same_history_edge_is_unrooted_without_a_prompt_ancestor():
    grounded_rows, roots = two_layer_history(grounded=True)
    feedback_rows, _ = two_layer_history(grounded=False)
    grounded = propagate_lineage(
        grounded_rows, roots, response_start=3, evidence_unit_count=1
    )
    feedback = propagate_lineage(
        feedback_rows, roots, response_start=3, evidence_unit_count=1
    )

    # The final response-history endpoint and edge weight are identical. Only
    # the multi-hop ancestry of source position 3 differs.
    grounded_final = grounded_rows[-1]
    feedback_final = feedback_rows[-1]
    assert grounded_final.source.tolist() == feedback_final.source.tolist() == [3]
    torch.testing.assert_close(grounded_final.support, feedback_final.support)

    torch.testing.assert_close(
        feedback.grounded_response_relay[1, 2, 0], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        feedback.unrooted_response_feedback[1, 2, 0], torch.tensor(1.0)
    )
    assert grounded.unrooted_response_feedback[1, 2, 0] < 1e-7


def test_sparse_tail_stays_unknown_instead_of_becoming_feedback():
    roots = torch.tensor([1, 0, 0, 2, 3, 4])
    rows = [
        route_row(
            layer=0,
            query=query,
            residual_support=0.75 if query == 4 else 1.0,
            unknown_support=0.25 if query == 4 else 0.0,
        )
        for query in (2, 3, 4)
    ]
    lineage = propagate_lineage(rows, roots, response_start=3, evidence_unit_count=1)

    unknown_root = lineage.ancestry.shape[-1] - 1
    torch.testing.assert_close(lineage.ancestry[1, 4].sum(), torch.tensor(1.0))
    torch.testing.assert_close(lineage.ancestry[1, 4, unknown_root], torch.tensor(0.25))
    torch.testing.assert_close(lineage.unknown[0, 2, 0], torch.tensor(0.25))
    torch.testing.assert_close(
        lineage.unrooted_response_feedback[0, 2, 0], torch.tensor(0.0)
    )


def test_streaming_lineage_matches_the_small_graph_oracle_without_holding_rows():
    rows, roots = two_layer_history(grounded=True)
    expected = propagate_lineage(rows, roots, response_start=3, evidence_unit_count=1)
    tracker = LineageTracker(
        roots,
        response_start=3,
        evidence_unit_count=1,
        layer_count=2,
        head_count=1,
    )
    tracker.add_many(rows[:3])
    tracker.add_many(rows[3:])
    actual = tracker.finish()

    for name in (
        "query_position",
        "prediction_position",
        "history_valid",
        "ancestry",
        "prompt_evidence",
        "grounded_response_relay",
        "unrooted_response_feedback",
        "predictor_self",
        "unknown",
        "effective_sources",
        "effective_head_rank",
        "anchor_source",
    ):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            rtol=0,
            atol=0,
        )

    transient = route_row(0, 0)
    reference = weakref.ref(transient)
    tracker = LineageTracker(
        roots,
        response_start=3,
        evidence_unit_count=1,
        layer_count=1,
        head_count=1,
    )
    tracker.add_many((transient,))
    del transient
    gc.collect()
    assert reference() is None


def test_topology_uses_capacity_while_lineage_transition_uses_positive_support():
    roots = torch.tensor([1, 1, 0, 2, 3, 4])
    row = route_row(
        0,
        2,
        source=(0, 1),
        support=(0.25, 0.0),
        residual_support=0.75,
    )
    row = replace(row, capacity=torch.tensor([3.0, 1.0]))
    lineage = propagate_lineage([row], roots, response_start=3, evidence_unit_count=1)

    # D/G/U and ancestry use the normalized constructive support account.
    torch.testing.assert_close(lineage.prompt_evidence[0, 0, 0], torch.tensor(0.25))
    torch.testing.assert_close(lineage.ancestry[1, 2, 1], torch.tensor(0.25))

    # Capacity sees both physical evidence endpoints even though the second
    # edge provides no positive signed support to the lineage transition.
    expected_sources = torch.exp(
        -(torch.tensor([0.75, 0.25]) * torch.log(torch.tensor([0.75, 0.25]))).sum()
    )
    torch.testing.assert_close(lineage.effective_sources[0, 0], expected_sources)
    assert lineage.anchor_source[0, 0, 0] == 0


def test_dense_lineage_matches_full_sparse_oracle():
    rows, roots = two_layer_history(grounded=True)
    sparse = propagate_lineage(rows, roots, response_start=3, evidence_unit_count=1)
    dense_tracker = LineageTracker(
        roots,
        response_start=3,
        evidence_unit_count=1,
        layer_count=2,
        head_count=1,
    )
    for layer in range(2):
        layer_rows = rows[layer * 3 : (layer + 1) * 3]
        capacity = torch.zeros(3, 1, len(roots))
        support = torch.zeros_like(capacity)
        for local, row in enumerate(layer_rows):
            capacity[local, row.head, row.source] = row.capacity
            support[local, row.head, row.source] = row.support
        dense_tracker.add_dense(
            layer,
            torch.tensor([row.query_position for row in layer_rows]),
            capacity,
            support,
        )
    dense = dense_tracker.finish()

    for field in fields(type(dense)):
        name = field.name
        torch.testing.assert_close(
            getattr(dense, name), getattr(sparse, name), rtol=0, atol=0
        )


def test_dense_topology_keeps_head_and_source_endpoints():
    roots = torch.tensor([1, 1, 0, 2, 3, 4])
    tracker = LineageTracker(
        roots,
        response_start=3,
        evidence_unit_count=1,
        layer_count=1,
        head_count=2,
    )
    capacity = torch.zeros(1, 2, len(roots))
    support = torch.zeros_like(capacity)
    capacity[0, 0, 0] = 2.0
    capacity[0, 1, 1] = 3.0
    support[0, 0, 0] = 0.25
    support[0, 1, 1] = 0.25
    tracker.add_dense(0, torch.tensor([2]), capacity, support)
    lineage = tracker.finish()

    torch.testing.assert_close(lineage.effective_sources[0, 0], torch.tensor(2.0))
    torch.testing.assert_close(lineage.effective_head_rank[0, 0], torch.tensor(2.0))
    assert lineage.anchor_source[0, 0].tolist() == [0, 1]
    torch.testing.assert_close(
        lineage.prompt_evidence[0, 0], torch.tensor([0.25, 0.25])
    )
