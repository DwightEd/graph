"""Matched controls for ordered attention-lineage diagnostics.

Layer-order controls leave every cached attention row unchanged and alter only
the order in which layers are composed.  The carrier control leaves prompt
endpoints untouched and rewires only retained response-to-response edges.  It
therefore tests whether exact response-token continuity matters beyond direct
prompt lookback, coarse lag, degree, and per-row attention mass.
"""

from dataclasses import replace

import torch

from experiments.grounded_route.controls import rewire_endpoints_keep_roles
from experiments.grounded_route.graph import TokenEdges, TokenGraph


LINEAGE_CONTROLS = (
    "ordered",
    "reverse",
    "random_layer",
    "last_layer",
    "carrier_rewire",
)


def layer_order(
    control: str,
    layer_count: int,
    seed: int,
) -> tuple[int, ...]:
    """Return the deterministic layer composition order for one control."""

    layer_count = int(layer_count)
    if layer_count < 1:
        raise ValueError("layer_count must be positive")

    ordered = tuple(range(layer_count))
    if control in ("ordered", "carrier_rewire"):
        return ordered
    if control == "reverse":
        return tuple(reversed(ordered))
    if control == "last_layer":
        return (layer_count - 1,)
    if control == "random_layer":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = ordered
        # Rejection sampling keeps the random null symmetric while ensuring it
        # actually changes layer order whenever a change is possible.
        while layer_count > 1 and permutation == ordered:
            permutation = tuple(
                map(
                    int,
                    torch.randperm(layer_count, generator=generator).tolist(),
                )
            )
        return permutation
    raise ValueError(f"unknown lineage control: {control}")


def response_carrier_edges(graph: TokenGraph) -> torch.Tensor:
    """Select retained edges whose source is an earlier response token."""

    return graph.edges.source >= graph.response_start


def rewire_response_carriers(
    graph: TokenGraph,
    generator: torch.Generator,
    *,
    passes: int = 4,
) -> TokenGraph:
    """Break exact response carriers with matched causal double-edge swaps.

    The existing endpoint control swaps sources only within layer, head,
    source role, and logarithmic lag bucket.  Applying it to the response-edge
    subgraph additionally keeps every prompt endpoint exact.  Source and target
    degrees, target-row weights, diagonal mass, unresolved mass, and causal
    validity are preserved.  Sparse graphs can admit no legal swap; callers
    should report the realized changed-edge fraction rather than assume one.
    """

    graph = graph.canonicalize()
    carrier = response_carrier_edges(graph)
    if int(carrier.sum().item()) < 2:
        return graph

    carrier_graph = replace(graph, edges=graph.edges.select(carrier)).check()
    rewired = rewire_endpoints_keep_roles(
        carrier_graph,
        generator,
        passes=int(passes),
    )
    prompt_edges = graph.edges.select(~carrier)
    carrier_edges = rewired.edges
    combined = TokenEdges(
        source=torch.cat((prompt_edges.source, carrier_edges.source)),
        target=torch.cat((prompt_edges.target, carrier_edges.target)),
        layer=torch.cat((prompt_edges.layer, carrier_edges.layer)),
        head=torch.cat((prompt_edges.head, carrier_edges.head)),
        weight=torch.cat((prompt_edges.weight, carrier_edges.weight)),
    )
    return replace(graph, edges=combined).check().canonicalize()


def apply_lineage_control(
    graph: TokenGraph,
    control: str,
    *,
    seed: int,
    carrier_rewire_passes: int = 4,
) -> tuple[TokenGraph, tuple[int, ...]]:
    """Return the controlled graph and layer order consumed by lineage DP."""

    order = layer_order(control, graph.layer_count, seed)
    if control != "carrier_rewire":
        return graph, order

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    controlled = rewire_response_carriers(
        graph,
        generator,
        passes=carrier_rewire_passes,
    )
    return controlled, order
