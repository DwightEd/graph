"""Label-free dispersion diagnostics for the observed sparse attention rows.

The cache records retained off-diagonal endpoints, the exact diagonal, and a
single unresolved mass bucket.  Consequently, endpoint entropy and
concentration are intervals rather than point estimates.  This module keeps
that censoring explicit and computes every interval per head before any head
average.  These are attention-routing statistics, not semantic uncertainty or
evidence of reliance on parameterized model knowledge.
"""

from dataclasses import dataclass

import torch

from experiments.grounded_route.graph import TokenGraph


LOWER = 0
UPPER = 1

PROMPT_MASS = 0
RESPONSE_MASS = 1
DIAGONAL_MASS = 2
UNRESOLVED_MASS = 3
ROLE_MASSES = 4


@dataclass(frozen=True)
class PerHeadEndpointBounds:
    """Endpoint-distribution intervals with shape ``[R, L, H, 2]``.

    The last dimension is ``(lower, upper)``.  ``censored_endpoint_count`` has
    shape ``[R, L, H]`` and counts causally eligible off-diagonal endpoints
    absent from the retained sparse row.
    """

    normalized_entropy: torch.Tensor
    hhi: torch.Tensor
    normalized_hhi: torch.Tensor
    censored_endpoint_count: torch.Tensor


@dataclass(frozen=True)
class RoutingDispersion:
    """Per-query routing dispersion and its predecessor-aligned token view.

    ``query_*`` rows index cached response queries.  Cached response query
    ``i`` predicts generated response token ``i + 1``.  The ``token_*``
    properties therefore drop the final query, omit the unavailable first
    token prediction, and align query row ``i`` with response token ``i + 1``.
    """

    per_head: PerHeadEndpointBounds
    query_entropy_bounds: torch.Tensor
    query_hhi_bounds: torch.Tensor
    query_concentration_bounds: torch.Tensor
    head_role_mass: torch.Tensor
    query_role_mass: torch.Tensor
    query_role_mass_disagreement: torch.Tensor
    query_role_js_disagreement: torch.Tensor
    predictor_response_index: torch.Tensor
    token_response_index: torch.Tensor
    token_id: torch.Tensor

    @property
    def token_entropy_bounds(self) -> torch.Tensor:
        return self.query_entropy_bounds[:-1]

    @property
    def token_hhi_bounds(self) -> torch.Tensor:
        return self.query_hhi_bounds[:-1]

    @property
    def token_concentration_bounds(self) -> torch.Tensor:
        return self.query_concentration_bounds[:-1]

    @property
    def token_role_mass(self) -> torch.Tensor:
        return self.query_role_mass[:-1]

    @property
    def token_role_mass_disagreement(self) -> torch.Tensor:
        return self.query_role_mass_disagreement[:-1]

    @property
    def token_role_js_disagreement(self) -> torch.Tensor:
        return self.query_role_js_disagreement[:-1]


def entropy_mass(mass: torch.Tensor) -> torch.Tensor:
    """Return the elementwise Shannon term ``-p log(p)`` with ``0 log 0=0``."""

    safe_mass = mass.clamp_min(torch.finfo(mass.dtype).tiny)
    return torch.where(mass > 0, -mass * safe_mass.log(), 0.0)


def retained_row_statistics(
    graph: TokenGraph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate exact retained endpoint statistics without merging heads."""

    shape = (graph.response_count, graph.layer_count, graph.head_count)
    entropy = graph.diagonal.new_zeros(shape)
    hhi = graph.diagonal.new_zeros(shape)
    count = torch.zeros(shape, dtype=torch.long, device=graph.diagonal.device)
    prompt_mass = graph.diagonal.new_zeros(shape)
    response_mass = graph.diagonal.new_zeros(shape)

    edges = graph.edges.to(graph.diagonal.device)
    if edges.count:
        row = (edges.target - graph.response_start, edges.layer, edges.head)
        entropy.index_put_(row, entropy_mass(edges.weight), accumulate=True)
        hhi.index_put_(row, edges.weight.square(), accumulate=True)
        count.index_put_(row, torch.ones_like(edges.source), accumulate=True)
        prompt = edges.source < graph.response_start
        prompt_mass.index_put_(row, edges.weight * prompt, accumulate=True)
        response_mass.index_put_(row, edges.weight * ~prompt, accumulate=True)
    return entropy, hhi, count, prompt_mass, response_mass


def endpoint_distribution_bounds(
    graph: TokenGraph,
) -> tuple[PerHeadEndpointBounds, torch.Tensor]:
    """Bound entropy and HHI for every sparse attention row.

    For unresolved mass ``u`` over ``m`` censored endpoints, entropy is lowest
    when the mass is concentrated and highest when it is uniform.  HHI has the
    opposite extrema.  These conservative bounds require only ``u`` and ``m``;
    they do not incorrectly treat a censored endpoint as observed zero.

    Returns the bounds and the exact four-way per-head role mass ordered as
    prompt, earlier response, diagonal, and unresolved.
    """

    graph = graph.canonicalize()
    retained_entropy, retained_hhi, retained_count, prompt, response = (
        retained_row_statistics(graph)
    )
    diagonal = graph.diagonal
    unresolved = graph.unresolved

    absolute_query = graph.response_start + torch.arange(
        graph.response_count,
        device=diagonal.device,
    )
    eligible_count = (absolute_query + 1)[:, None, None]
    censored_count = absolute_query[:, None, None] - retained_count
    impossible = (censored_count == 0) & (unresolved > 1e-6)
    if bool(impossible.any()):
        raise ValueError("unresolved mass has no causally eligible censored endpoint")

    censored_float = censored_count.clamp_min(1).to(diagonal.dtype)
    unresolved_entropy = entropy_mass(unresolved)
    known_entropy = retained_entropy + entropy_mass(diagonal)
    entropy_lower = known_entropy + unresolved_entropy
    entropy_upper = entropy_lower + unresolved * censored_float.log()

    normalizer = eligible_count.to(diagonal.dtype).log()
    entropy_lower = torch.where(
        normalizer > 0,
        entropy_lower / normalizer,
        torch.zeros_like(entropy_lower),
    ).clamp(0.0, 1.0)
    entropy_upper = torch.where(
        normalizer > 0,
        entropy_upper / normalizer,
        torch.zeros_like(entropy_upper),
    ).clamp(0.0, 1.0)

    known_hhi = retained_hhi + diagonal.square()
    hhi_lower = known_hhi + torch.where(
        censored_count > 0,
        unresolved.square() / censored_float,
        torch.zeros_like(unresolved),
    )
    hhi_upper = known_hhi + unresolved.square()
    uniform_hhi = eligible_count.to(diagonal.dtype).reciprocal()
    concentration_scale = 1.0 - uniform_hhi
    concentration_lower = torch.where(
        concentration_scale > 0,
        (hhi_lower - uniform_hhi) / concentration_scale,
        torch.zeros_like(hhi_lower),
    ).clamp(0.0, 1.0)
    concentration_upper = torch.where(
        concentration_scale > 0,
        (hhi_upper - uniform_hhi) / concentration_scale,
        torch.zeros_like(hhi_upper),
    ).clamp(0.0, 1.0)

    bounds = PerHeadEndpointBounds(
        normalized_entropy=torch.stack((entropy_lower, entropy_upper), dim=-1),
        hhi=torch.stack((hhi_lower, hhi_upper), dim=-1),
        normalized_hhi=torch.stack(
            (concentration_lower, concentration_upper), dim=-1
        ),
        censored_endpoint_count=censored_count,
    )
    role_mass = torch.stack((prompt, response, diagonal, unresolved), dim=-1)
    return bounds, role_mass


def head_role_disagreement(
    role_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize disagreement among heads over four endpoint roles.

    Returns mean role mass, population standard deviation for each role, and
    generalized Jensen-Shannon disagreement normalized to ``[0, 1]``.
    """

    mean = role_mass.mean(dim=2)
    deviation = (role_mass - mean[:, :, None]).square().mean(dim=2).sqrt()
    head_count = role_mass.shape[2]
    if head_count == 1:
        js = mean[..., 0] * 0.0
    else:
        mixture_entropy = entropy_mass(mean).sum(dim=-1)
        head_entropy = entropy_mass(role_mass).sum(dim=-1).mean(dim=2)
        maximum = torch.log(
            role_mass.new_tensor(float(min(head_count, ROLE_MASSES)))
        )
        js = ((mixture_entropy - head_entropy) / maximum).clamp(0.0, 1.0)
    return mean, deviation, js


@torch.no_grad()
def attention_routing_dispersion(graph: TokenGraph) -> RoutingDispersion:
    """Compute label-free per-layer routing dispersion from one token graph."""

    graph = graph.canonicalize()
    per_head, role_mass = endpoint_distribution_bounds(graph)
    mean_role, role_deviation, role_js = head_role_disagreement(role_mass)

    aligned_count = max(graph.response_count - 1, 0)
    predictor = torch.arange(aligned_count, device=graph.diagonal.device)
    token_index = predictor + 1
    return RoutingDispersion(
        per_head=per_head,
        query_entropy_bounds=per_head.normalized_entropy.mean(dim=2),
        query_hhi_bounds=per_head.hhi.mean(dim=2),
        query_concentration_bounds=per_head.normalized_hhi.mean(dim=2),
        head_role_mass=role_mass,
        query_role_mass=mean_role,
        query_role_mass_disagreement=role_deviation,
        query_role_js_disagreement=role_js,
        predictor_response_index=predictor,
        token_response_index=token_index,
        token_id=graph.response_token_ids[1:].to(graph.diagonal.device),
    )
