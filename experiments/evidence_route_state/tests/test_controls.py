import torch

from experiments.evidence_route_state.controls import (
    dense_endpoint_rewire,
    dense_weight_shuffle,
    dense_without_messages,
)
from experiments.evidence_route_state.lineage import LineageTracker
from experiments.evidence_route_state.state import build_route_state


def test_dense_controls_preserve_every_row_head_role_total():
    capacity = torch.tensor(
        [
            [
                [1.0, 2.0, 4.0, 8.0, 16.0, 0.0],
                [10.0, 20.0, 40.0, 80.0, 160.0, 0.0],
            ]
        ]
    )
    support = -capacity / 10
    query = torch.tensor([5])

    rewired = dense_endpoint_rewire(capacity, support, query, response_start=3)
    shuffled = dense_weight_shuffle(capacity, support, query, response_start=3)
    for changed_capacity, changed_support in (rewired, shuffled):
        assert not torch.equal(changed_capacity, capacity)
        torch.testing.assert_close(changed_support, -changed_capacity / 10)
        for role in (slice(0, 3), slice(3, 5), slice(5, 6)):
            torch.testing.assert_close(
                changed_capacity[..., role].sum(2), capacity[..., role].sum(2)
            )
            torch.testing.assert_close(
                changed_support[..., role].sum(2), support[..., role].sum(2)
            )

    # Endpoint rewiring moves a complete cross-head source signature.  Weight
    # shuffling is head-specific and therefore destroys that pairing.
    torch.testing.assert_close(rewired[0][:, 1], rewired[0][:, 0] * 10)
    assert not torch.equal(shuffled[0][:, 1], shuffled[0][:, 0] * 10)


def test_dense_controls_never_move_mass_to_future_sources_and_ignore_chunking():
    capacity = torch.zeros(3, 2, 6)
    for row, query in enumerate((1, 3, 5)):
        capacity[row, :, : query + 1] = torch.arange(1, query + 2)
    support = capacity / 10
    query = torch.tensor([1, 3, 5])

    complete = dense_endpoint_rewire(capacity, support, query, response_start=4)
    pieces = [
        dense_endpoint_rewire(
            capacity[row : row + 1],
            support[row : row + 1],
            query[row : row + 1],
            response_start=4,
        )
        for row in range(3)
    ]
    for account, changed in enumerate(complete):
        torch.testing.assert_close(
            changed, torch.cat([piece[account] for piece in pieces])
        )
        for row, position in enumerate(query):
            assert torch.count_nonzero(changed[row, :, position + 1 :]) == 0


def test_weight_shuffle_changes_every_head_when_a_role_has_multiple_sources():
    capacity = torch.arange(1, 21, dtype=torch.float32).reshape(1, 4, 5)
    support = capacity / 100
    shuffled, _ = dense_weight_shuffle(
        capacity,
        support,
        torch.tensor([4]),
        response_start=5,
    )

    assert torch.all(torch.any(shuffled != capacity, dim=2))


def test_no_message_lineage_has_no_identifiable_route_state():
    tracker = LineageTracker(
        torch.tensor([1, 0, 2, 3, 4, 5]),
        response_start=3,
        evidence_unit_count=1,
        layer_count=1,
        head_count=1,
    )
    capacity, support = dense_without_messages(
        torch.ones(3, 1, 6),
        torch.ones(3, 1, 6),
    )
    tracker.add_dense(0, torch.tensor([2, 3, 4]), capacity, support)
    lineage = tracker.finish()
    state = build_route_state(lineage)

    assert torch.count_nonzero(lineage.prompt_evidence) == 0
    assert torch.count_nonzero(lineage.grounded_response_relay) == 0
    assert torch.count_nonzero(lineage.unrooted_response_feedback) == 0
    assert torch.count_nonzero(lineage.effective_sources) == 0
    assert torch.count_nonzero(lineage.effective_head_rank) == 0
    assert torch.all(lineage.anchor_source == -1)
    assert not state.valid.any()
