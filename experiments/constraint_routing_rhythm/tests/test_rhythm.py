from __future__ import annotations

import torch

from experiments.constraint_routing_rhythm.rhythm import (
    build_rhythm,
    relay_diagnostics,
)
from experiments.constraint_routing_rhythm.routes import FunctionalRoutes


def route_fixture(global_map: torch.Tensor | None = None) -> FunctionalRoutes:
    # Rows are absolute query positions 2..6; sources are positions 0..6.
    local = torch.zeros(5, 7)
    local[0, 0] = 1.0
    local[1, 1] = 1.0
    local[2, 4] = 1.0
    local[3, 5] = 1.0
    local[4, 6] = 1.0

    if global_map is None:
        global_map = torch.zeros(5, 7)
        global_map[3, 4] = 0.8  # carrier 4 -> target query 5
        global_map[4, 4] = 0.6  # carrier 4 -> target query 6
        global_map[4, 5] = 0.1

    all_map = torch.zeros(5, 7)
    all_map[:, 0] = 0.2
    all_map[:, 1] = 0.1
    all_map[:, 2] = 0.2
    all_map[:, 3] = 0.1
    all_map[:, 4] = 0.2
    all_map[:, 5] = 0.1
    all_map[:, 6] = 0.1
    all_map[2] = torch.tensor([0.4, 0.2, 0.1, 0.1, 0.2, 0.0, 0.0])

    return FunctionalRoutes(
        row_start=2,
        split_layer=2,
        absolute_map=2 * all_map,
        all_map=all_map,
        early_absolute_map=2 * all_map,
        early_map=all_map,
        late_absolute_map=2 * global_map,
        late_map=global_map,
        local_map=local,
        global_map=global_map,
    )


def build(routes: FunctionalRoutes | None = None):
    return build_rhythm(
        routes or route_fixture(),
        response_start=3,
        evidence_mask=torch.tensor([True, True, False]),
        window=2,
        horizon_low=1,
        horizon_high=2,
        carrier_quantile=0.75,
        max_carriers=8,
        split_layer=2,
    )


def test_reach_delivery_binding_and_capacity_use_the_exact_maps() -> None:
    rhythm = build()

    torch.testing.assert_close(
        rhythm.functional_reach, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    )
    # Carrier 4 reads future query rows 5 and 6 from source column 4.
    torch.testing.assert_close(rhythm.future_influence[1], torch.tensor(0.7))
    torch.testing.assert_close(rhythm.future_delivery[1], torch.tensor(1.4))
    # Its carrier state is absolute row 4, not query row 3 that generated it.
    torch.testing.assert_close(rhythm.evidence_binding[1], torch.tensor(0.6))
    torch.testing.assert_close(rhythm.evidence_uptake[1], torch.tensor(1.2))
    torch.testing.assert_close(rhythm.relay_capacity[1], torch.tensor(0.6))
    torch.testing.assert_close(rhythm.relay_mass[1], torch.tensor(1.2))


def test_query_prediction_alignment_and_carrier_state_are_distinct() -> None:
    rhythm = build()

    torch.testing.assert_close(rhythm.query_position, torch.arange(2, 7))
    torch.testing.assert_close(rhythm.prediction_position, torch.arange(3, 8))
    torch.testing.assert_close(rhythm.prediction_position, rhythm.query_position + 1)
    assert rhythm.carrier_mask.tolist() == [False, True, False, False, False]
    assert torch.isnan(rhythm.relay_capacity[-2:]).all()


def test_upstream_and_downstream_edges_use_carrier_endpoints() -> None:
    rhythm = build()

    expected_upstream = torch.zeros(7, 7, dtype=torch.bool)
    expected_upstream[4, 0] = True
    expected_upstream[4, 1] = True
    expected_downstream = torch.zeros(7, 7, dtype=torch.bool)
    expected_downstream[5, 4] = True
    expected_downstream[6, 4] = True
    torch.testing.assert_close(rhythm.upstream_edges, expected_upstream)
    torch.testing.assert_close(rhythm.downstream_edges, expected_downstream)


def test_functional_reach_does_not_preselect_relay_endpoints() -> None:
    global_map = torch.zeros(5, 7)
    global_map[4, 5] = 1.0  # carrier 5 is not at a local-reach peak.
    routes = route_fixture(global_map)
    routes.local_map.zero_()
    routes.local_map[4, 4] = 1.0

    rhythm = build_rhythm(
        routes,
        response_start=3,
        evidence_mask=torch.tensor([True, True, False]),
        window=2,
        horizon_low=1,
        horizon_high=2,
        carrier_quantile=1.0,
        split_layer=2,
    )

    assert rhythm.carrier_mask.tolist() == [False, False, True, False, False]
    assert rhythm.upstream_edges.any()
    assert rhythm.downstream_edges.any()


def test_absolute_floor_can_reject_a_high_normalized_two_hop_route() -> None:
    routes = route_fixture()
    # Carrier 5 wins the normalized bottleneck but carries negligible mass.
    routes.early_map[3, :2] = torch.tensor([0.4, 0.3])
    routes.late_map[4, 5] = 0.8
    routes.early_absolute_map[3, :2] = torch.tensor([0.005, 0.005])
    routes.late_absolute_map[4, 5] = 0.01

    rhythm = build_rhythm(
        routes,
        response_start=3,
        evidence_mask=torch.tensor([True, True, False]),
        window=2,
        horizon_low=1,
        horizon_high=2,
        carrier_quantile=0.75,
        mass_floor=0.02,
        split_layer=2,
    )

    assert rhythm.relay_capacity[2] > rhythm.relay_capacity[1]
    assert rhythm.relay_mass[2] < 0.02
    assert rhythm.carrier_mask.tolist() == [False, True, False, False, False]


def test_relay_diagnostics_are_raw_four_cell_effects() -> None:
    diagnostic = relay_diagnostics(
        u_delta=torch.tensor([-1.0, 0.2]),
        d_delta=torch.tensor([-0.5, -0.3]),
        ud_delta=torch.tensor([-2.0, -0.4]),
    )

    expected = torch.tensor(
        [
            [-1.0, -0.5, -0.5],
            [0.2, -0.3, -0.3],
        ]
    )
    torch.testing.assert_close(diagnostic, expected)
