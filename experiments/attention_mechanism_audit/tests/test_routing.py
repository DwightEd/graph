import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.attention_mechanism_audit.routing import (
    direct_role_bases_and_carrier,
    response_carrier_ancestry,
    routing_flow,
)
from experiments.grounded_route.graph import TokenEdges, TokenGraph


def recurrence_graph():
    # Layer 0 gives q0 .5 evidence; q1 is still entirely response-rooted.
    # Layer 1 lets q1 read .4 from q0.  Correct parallel layer semantics thus
    # relay .4 * .5 evidence, whereas an invalid same-layer recurrence would
    # already relay q0's evidence during layer 0.
    return TokenGraph(
        sample_id="recurrence",
        source_id="source",
        task_type="QA",
        response_start=1,
        token_count=4,
        response_count=3,
        layer_count=2,
        head_count=1,
        attention_floor=0.0,
        edges=TokenEdges(
            source=torch.tensor([0, 1]),
            target=torch.tensor([1, 2]),
            layer=torch.tensor([0, 1]),
            head=torch.zeros(2, dtype=torch.long),
            weight=torch.tensor([0.5, 0.4]),
        ),
        diagonal=torch.tensor(
            [
                [[0.5], [1.0]],
                [[1.0], [0.6]],
                [[1.0], [1.0]],
            ]
        ),
        unresolved=torch.zeros((3, 2, 1)),
        token_ids=torch.tensor([10, 20, 30, 40]),
    ).check().canonicalize()


def test_response_carrier_recurrence_separates_relayed_and_ungrounded():
    graph = recurrence_graph()
    direct, carrier, diagonal, unresolved = direct_role_bases_and_carrier(
        graph, np.asarray([0])
    )
    assert torch.allclose(direct[0, 0, 0], torch.tensor(0.5))
    selected = (carrier.query == 1) & (carrier.source == 0) & (carrier.layer == 1)
    assert int(selected.sum()) == 1
    assert torch.allclose(carrier.weight[selected], torch.tensor([0.4]))

    ancestry = response_carrier_ancestry(direct, carrier, diagonal, unresolved)
    # q1 cannot consume q0's same-layer update.
    torch.testing.assert_close(
        ancestry["routing_grounded_role_ancestry"][1, 0, 0],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        ancestry["routing_relayed_role_ancestry"][1, 1, 0],
        torch.tensor(0.2),
    )
    torch.testing.assert_close(
        ancestry["routing_grounded_role_ancestry"][1, 1, 0],
        torch.tensor(0.2),
    )
    torch.testing.assert_close(
        ancestry["routing_ungrounded_history_ancestry"][1, 1],
        torch.tensor(0.8),
    )


def test_routing_flow_preserves_unavailable_token_zero_as_nan():
    output = routing_flow(recurrence_graph(), np.asarray([0]))
    assert output["routing_available"].tolist() == [False, True, True]
    assert np.isnan(output["routing_entropy_bounds"][0]).all()
    assert np.isnan(output["routing_direct_role_ancestry"][0]).all()
    np.testing.assert_allclose(
        output["routing_relayed_role_ancestry"][2, 1, 0], 0.2
    )
