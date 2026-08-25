"""Matched graph nulls used to test endpoint and weight claims.

Both controls leave node attributes, censoring, diagonal mass and unresolved
mass untouched.  They operate only on the retained directed attention edges.
"""

from dataclasses import replace

import torch

from .graph import TokenEdges, TokenGraph


def source_role(graph: TokenGraph) -> torch.Tensor:
    return (graph.edges.source >= graph.response_start).long()


def lag_bucket(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lag = (target - source).clamp_min(1).float()
    return torch.floor(torch.log2(lag)).long()


def with_edges(graph: TokenGraph, *, source=None, weight=None) -> TokenGraph:
    edges = graph.edges
    changed = TokenEdges(
        source=edges.source if source is None else source,
        target=edges.target,
        layer=edges.layer,
        head=edges.head,
        weight=edges.weight if weight is None else weight,
    )
    return replace(graph, edges=changed).check()


def group_index(key: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Return one stable group sort and its CPU slice boundaries."""

    order = torch.argsort(key, stable=True)
    if not len(order):
        return order, [0]
    ordered_key = key[order]
    boundaries = torch.nonzero(
        ordered_key[1:] != ordered_key[:-1], as_tuple=False
    ).flatten() + 1
    offsets = [0, *boundaries.detach().cpu().tolist(), len(order)]
    return order, offsets


def canonical_sources(graph: TokenGraph, source: torch.Tensor) -> torch.Tensor:
    """Restore increasing CSR source order inside every unchanged target row."""

    row = (
        (graph.edges.layer * graph.head_count + graph.edges.head)
        * graph.response_count
        + graph.edge_response_target
    )
    destination = torch.argsort(row, stable=True)
    source_order = torch.argsort(
        row * graph.token_count + source,
        stable=True,
    )
    result = source.clone()
    result[destination] = source[source_order]
    return result


def packed_endpoint_key(
    graph: TokenGraph,
    source,
    target,
    layer,
    head,
):
    """Pack typed endpoints in the ordering used by negative sampling."""

    row = (
        (layer * graph.head_count + head) * graph.response_count
        + target
        - graph.response_start
    )
    return row * graph.token_count + source


def endpoint_key_set(graph: TokenGraph, chunk_size: int = 262_144) -> set[int]:
    """Build compact endpoint membership without an edge-sized Python list."""

    result: set[int] = set()
    edges = graph.edges
    for start in range(0, graph.edge_count, chunk_size):
        stop = min(start + chunk_size, graph.edge_count)
        packed = packed_endpoint_key(
            graph,
            edges.source[start:stop],
            edges.target[start:stop],
            edges.layer[start:stop],
            edges.head[start:stop],
        )
        result.update(packed.detach().cpu().tolist())
    return result


def shuffle_weights_keep_endpoints(
    graph: TokenGraph,
    generator: torch.Generator,
) -> TokenGraph:
    """Permute weights within ``(target, layer, head, source-role)`` rows.

    Endpoints, support, row mass, role mass and each group's complete weight
    multiset are invariant.  Only the association between a retained weight
    and its exact source endpoint is destroyed.
    """

    edges = graph.edges
    weight = edges.weight.clone()
    role = source_role(graph)
    key = (
        ((edges.target * graph.layer_count + edges.layer) * graph.head_count + edges.head)
        * 2
        + role
    )
    destination, _ = group_index(key)
    random_order = torch.randperm(
        edges.count,
        generator=generator,
        device=edges.weight.device,
    )
    source_order = random_order[
        torch.argsort(key[random_order], stable=True)
    ]
    weight[destination] = edges.weight[source_order]
    return with_edges(graph, weight=weight)


def rewire_endpoints_keep_roles(
    graph: TokenGraph,
    generator: torch.Generator,
    passes: int = 4,
) -> TokenGraph:
    """Degree-preserving causal endpoint swaps within matched edge strata.

    A swap ``(s1 -> t1, s2 -> t2)`` becomes
    ``(s2 -> t1, s1 -> t2)``.  Candidates share layer, head, source role and
    original coarse lag bucket.  A swap is accepted only when both new edges
    are causal, remain in that lag bucket and do not duplicate an existing
    edge.  Weights remain attached to their original target rows, so all
    target-row weight and censoring summaries are unchanged.  A successfully
    rewired edge is frozen; later passes only retry unchanged candidates.
    """

    edges = graph.edges
    source = edges.source.clone()
    role = source_role(graph)
    bucket = lag_bucket(source, edges.target)
    bucket_count = max(1, graph.token_count.bit_length())
    strata = (
        ((edges.layer * graph.head_count + edges.head) * 2 + role) * bucket_count
        + bucket
    )
    existing = endpoint_key_set(graph)
    group_order, offsets = group_index(strata)
    used = torch.zeros(edges.count, dtype=torch.bool, device=source.device)

    for _ in range(int(passes)):
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            members = group_order[start:stop]
            members = members[~used[members]]
            if len(members) < 2:
                continue
            permutation = torch.randperm(
                len(members), generator=generator, device=members.device
            )
            paired = members[
                permutation[: 2 * (len(members) // 2)]
            ].reshape(-1, 2)
            expected_bucket = int(bucket[members[0]].item())
            for left, right in paired.detach().cpu().tolist():
                s1, s2 = int(source[left]), int(source[right])
                t1, t2 = int(edges.target[left]), int(edges.target[right])
                layer = int(edges.layer[left])
                head = int(edges.head[left])
                if s1 == s2 or t1 == t2 or s2 >= t1 or s1 >= t2:
                    continue
                if (t1 - s2).bit_length() - 1 != expected_bucket:
                    continue
                if (t2 - s1).bit_length() - 1 != expected_bucket:
                    continue

                old_left = packed_endpoint_key(graph, s1, t1, layer, head)
                old_right = packed_endpoint_key(graph, s2, t2, layer, head)
                new_left = packed_endpoint_key(graph, s2, t1, layer, head)
                new_right = packed_endpoint_key(graph, s1, t2, layer, head)
                if (
                    new_left not in {old_left, old_right}
                    and new_left in existing
                ) or (
                    new_right not in {old_left, old_right}
                    and new_right in existing
                ):
                    continue

                existing.remove(old_left)
                existing.remove(old_right)
                existing.add(new_left)
                existing.add(new_right)
                source[left], source[right] = s2, s1
                used[left] = True
                used[right] = True

    return with_edges(graph, source=canonical_sources(graph, source))


def apply_variant(
    graph: TokenGraph,
    variant: str,
    generator: torch.Generator,
    *,
    endpoint_rewire_passes: int = 4,
) -> TokenGraph:
    """Apply one frozen graph variant without changing the public graph contract."""

    if variant == "real":
        return graph
    if variant == "weight_shuffle":
        return shuffle_weights_keep_endpoints(graph, generator)
    if variant == "endpoint_rewire":
        return rewire_endpoints_keep_roles(
            graph,
            generator,
            passes=endpoint_rewire_passes,
        )
    raise ValueError(f"unknown graph variant: {variant}")
