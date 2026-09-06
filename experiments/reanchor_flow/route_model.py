"""Head-resolved provenance, function, and integration on one causal route graph.

The class in this file is the analysis model.  Capture code supplies exact
``(layer, head, source, target)`` edges; this model turns them into three
ledgers without averaging heads:

1. provenance carried from evidence, other prompt tokens, and response tokens;
2. signed first-order action on one fixed target margin;
3. source-cut message integration before it enters the residual stream.

Residual differences and gradients are deliberately not called causal effects.
The exact cut/patch reruns in :mod:`experiments.reanchor_flow.native` provide
that final test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .flow import PairedFlow
from .message_norm import model_gram_cache, output_gram
from .native_world import NativeWorld
from .throughput import transition_probabilities

EVIDENCE = 0
OTHER_PROMPT = 1
RESPONSE = 2
UNOBSERVED = 3
CHANNEL_NAMES = ("evidence", "other_prompt", "response", "unobserved")


@dataclass(frozen=True)
class RouteEvent:
    """One head-specific node proposed for exact intervention."""

    layer: int
    head: int
    position: int
    score: float
    evidence_transport: float
    evidence_gradient_action: float
    local_response_transport: float
    local_response_gradient_action: float


@dataclass(frozen=True)
class RouteDynamics:
    """The non-averaged state returned by :class:`HeadResolvedRouteModel`.

    Shapes use ``L`` layers, ``H`` heads, ``P`` represented destination rows,
    ``N`` causal token positions, ``E`` retained edges, and four provenance
    channels.  ``head_*`` tensors always retain both ``L`` and ``H`` axes.
    """

    local_window: int
    row_position: Tensor
    node_register: Tensor  # [L + 1, N, 4]
    edge_register: Tensor  # [E, 4]
    head_transport: Tensor  # [L, H, P, 4]
    # Native edge gradient action allocated by the transport provenance model;
    # this is not an exact semantic decomposition of a nonlinear hidden state.
    head_gradient_action: Tensor  # [L, H, P, 4]
    head_direct_evidence: Tensor  # [L, H, P, 2]: transport, action
    head_local_response: Tensor  # [L, H, P, 2]: transport, action
    head_integration: Tensor  # [L, H, P, 4]: budget, net, coherence, action
    head_backward_distance: Tensor  # [L, H, P]
    head_span: Tensor  # [L, H]
    source_reuse: Tensor  # [L, H, N, 2]: future transport, action
    stage_position: Tensor  # [A]
    stage_presence: Tensor  # [L, A, 3]: residual, attention write, MLP write
    stage_gradient_action: Tensor  # [L, A, 3]


class HeadResolvedRouteModel:
    """Analyze information conversion without collapsing attention heads.

    ``local_window`` has the same role as the clipped local window in WAAD,
    but is applied to graph transport and target-gradient action rather than to
    an attention map averaged over a selected head group.
    """

    def __init__(self, local_window: int = 10) -> None:
        if local_window < 1:
            raise ValueError("local_window must be positive")
        self.local_window = int(local_window)

    @staticmethod
    def _slots(flow: PairedFlow, tokens: int) -> tuple[Tensor, Tensor]:
        lookup = torch.full((tokens,), -1, dtype=torch.long)
        lookup[flow.row_position.long()] = torch.arange(len(flow.row_position))
        edge_slot = lookup.index_select(0, flow.edges.target.long())
        if bool((edge_slot < 0).any()):
            raise ValueError("route edge target is absent from represented rows")
        return lookup, edge_slot

    @staticmethod
    def _scatter_head(
        flow: PairedFlow,
        edge_slot: Tensor,
        values: Tensor,
        layers: int,
        heads: int,
    ) -> Tensor:
        rows = len(flow.row_position)
        scalar = values.ndim == 1
        if scalar:
            values = values[:, None]
        flat = torch.zeros(
            layers * heads * rows,
            values.shape[1],
            dtype=torch.float32,
        )
        index = (
            (flow.edges.layer.long() * heads + flow.edges.head.long()) * rows
            + edge_slot
        )
        flat.index_add_(0, index, values.float())
        result = flat.view(layers, heads, rows, values.shape[1])
        return result[..., 0] if scalar else result

    @staticmethod
    def _initial_register(
        token_unit_id: Tensor,
        evidence_unit_id: tuple[int, ...],
        response_start: int,
    ) -> Tensor:
        tokens = len(token_unit_id)
        position = torch.arange(tokens)
        evidence = torch.zeros(tokens, dtype=torch.bool)
        for unit_id in evidence_unit_id:
            evidence |= token_unit_id.long() == unit_id
        evidence &= position < response_start
        response = position >= response_start
        other_prompt = ~(evidence | response)
        register = torch.zeros(tokens, len(CHANNEL_NAMES))
        register[evidence, EVIDENCE] = 1
        register[other_prompt, OTHER_PROMPT] = 1
        register[response, RESPONSE] = 1
        return register

    def _provenance(
        self,
        flow: PairedFlow,
        world: NativeWorld,
        edge_probability: Tensor,
        residual_probability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        tokens = len(world.units.token_unit_id)
        layers = flow.clean_cache.layer_count
        node = torch.zeros(layers + 1, tokens, len(CHANNEL_NAMES))
        node[0] = self._initial_register(
            world.units.token_unit_id,
            world.evidence_unit_id,
            world.response_start,
        )
        edge_register = torch.zeros(flow.edges.count, len(CHANNEL_NAMES))
        for layer in range(layers):
            node[layer + 1] = node[layer] * residual_probability[layer, :, None]
            selected = torch.nonzero(
                flow.edges.layer == layer, as_tuple=False
            ).flatten()
            if len(selected):
                source = flow.edges.source.index_select(0, selected).long()
                target = flow.edges.target.index_select(0, selected).long()
                carried = node[layer].index_select(0, source)
                carried *= edge_probability.index_select(0, selected)[:, None]
                edge_register[selected] = carried
                node[layer + 1].index_add_(0, target, carried)

            # Coverage-pruned routes remain explicitly unknown.  They are not
            # renormalized onto whichever edges happened to be retained.
            accounted = node[layer + 1].sum(-1)
            if bool((accounted > 1.0 + 2e-5).any()):
                raise FloatingPointError("route provenance is not conservative")
            node[layer + 1, :, UNOBSERVED] += (1 - accounted).clamp_min(0)
        return node, edge_register

    @staticmethod
    def _head_integration(
        model,
        flow: PairedFlow,
        edge_slot: Tensor,
        layers: int,
        heads: int,
    ) -> Tensor:
        """Aggregate real source-cut ``W_O(A V)`` deltas within each head."""

        rows = len(flow.row_position)
        edges = flow.edges
        budget = HeadResolvedRouteModel._scatter_head(
            flow, edge_slot, edges.delta_message_norm, layers, heads
        )
        action = HeadResolvedRouteModel._scatter_head(
            flow,
            edge_slot,
            torch.nan_to_num(
                edges.clean_target_score - edges.corrupt_target_score
            ),
            layers,
            heads,
        )
        net = torch.zeros_like(budget)
        delta_code = edges.clean_code.float() - edges.corrupt_code.float()
        head_dim = delta_code.shape[1]
        gram_cache = model_gram_cache(model)
        for layer, module in enumerate(model.model.layers):
            selected = torch.nonzero(edges.layer == layer, as_tuple=False).flatten()
            if not len(selected):
                continue
            code_sum = torch.zeros(heads * rows, head_dim)
            local_head = edges.head.index_select(0, selected).long()
            local_slot = edge_slot.index_select(0, selected)
            index = local_head * rows + local_slot
            code_sum.index_add_(0, index, delta_code.index_select(0, selected))
            code_sum = code_sum.view(heads, rows, head_dim)
            gram = gram_cache.get(layer)
            if gram is None:
                gram = output_gram(
                    module.self_attn.o_proj.weight.detach(), heads, head_dim
                )
                gram_cache[layer] = gram
            squared = torch.einsum(
                "hrd,hde,hre->hr", code_sum, gram.float(), code_sum
            )
            net[layer] = squared.clamp_min(0).sqrt()
        coherence = torch.where(budget > 0, net / budget, torch.zeros_like(net))
        return torch.stack((budget, net, coherence, action), dim=-1)

    def analyze(self, model, flow: PairedFlow, world: NativeWorld) -> RouteDynamics:
        """Build the three ledgers for one frozen target and one native cut.

        The signed score is ``gradient · message`` for the fixed observed-token
        versus runner-up margin.  It is a first-order screening quantity.  The
        caller must use the exact effects stored by ``NativeTargetAudit`` before
        making a causal claim.
        """

        world = world.check()
        tokens = len(world.units.token_unit_id)
        layers = flow.clean_cache.layer_count
        heads = int(model.config.num_attention_heads)
        if flow.row_total.shape[:2] != (layers, heads):
            raise ValueError("route tensor does not match model layer/head axes")
        _, edge_slot = self._slots(flow, tokens)
        edge_probability, residual_probability = transition_probabilities(
            flow, tokens
        )
        node, edge_register = self._provenance(
            flow, world, edge_probability, residual_probability
        )
        head_transport = self._scatter_head(
            flow, edge_slot, edge_register, layers, heads
        )

        native_action = torch.nan_to_num(flow.edges.clean_target_score.float())
        source_register = node[
            flow.edges.layer.long(), flow.edges.source.long()
        ]
        edge_action = source_register * native_action[:, None]
        head_action = self._scatter_head(
            flow, edge_slot, edge_action, layers, heads
        )

        evidence_unit = torch.zeros(tokens, dtype=torch.bool)
        for unit_id in world.evidence_unit_id:
            evidence_unit |= world.units.token_unit_id.long() == unit_id
        direct = evidence_unit.index_select(0, flow.edges.source.long())
        direct &= flow.edges.source.long() < world.response_start
        direct_values = torch.stack((edge_probability, native_action), dim=-1)
        direct_values *= direct[:, None]
        head_direct = self._scatter_head(
            flow, edge_slot, direct_values, layers, heads
        )

        distance = flow.edges.target.long() - flow.edges.source.long()
        response_origin = source_register[:, RESPONSE]
        local = (
            (flow.edges.source.long() >= world.response_start)
            & (distance > 0)
            & (distance <= self.local_window)
        )
        local_values = torch.stack(
            (edge_probability * response_origin, native_action * response_origin),
            dim=-1,
        )
        local_values *= local[:, None]
        head_local = self._scatter_head(
            flow, edge_slot, local_values, layers, heads
        )

        distance_weight = edge_probability * distance.clamp_min(0).float()
        distance_sum = self._scatter_head(
            flow, edge_slot, distance_weight, layers, heads
        )
        incoming = self._scatter_head(
            flow, edge_slot, edge_probability, layers, heads
        )
        backward_distance = torch.where(
            incoming > 0, distance_sum / incoming, torch.zeros_like(incoming)
        )
        response_row = flow.row_position >= world.response_start - 1
        response_incoming = incoming[:, :, response_row]
        span_denominator = response_incoming.sum(-1)
        span = torch.where(
            span_denominator > 0,
            distance_sum[:, :, response_row].sum(-1) / span_denominator,
            torch.zeros_like(span_denominator),
        )

        reuse = torch.zeros(layers * heads * tokens, 2)
        future = distance > 0
        reuse_index = (
            (flow.edges.layer.long() * heads + flow.edges.head.long()) * tokens
            + flow.edges.source.long()
        )
        reuse_value = torch.stack((edge_probability, native_action), dim=-1)
        reuse.index_add_(0, reuse_index[future], reuse_value[future])
        reuse = reuse.view(layers, heads, tokens, 2)

        integration = self._head_integration(
            model, flow, edge_slot, layers, heads
        )
        if flow.stages is None:
            stage_position = torch.empty(0, dtype=torch.long)
            stage_presence = torch.empty((0, 0, 3))
            stage_action = torch.empty((0, 0, 3))
        else:
            stage_position = flow.stages.position.clone()
            stage_presence = torch.stack(
                (
                    flow.stages.state_delta_norm,
                    flow.stages.attention_delta_norm,
                    flow.stages.mlp_delta_norm,
                ),
                dim=-1,
            )
            stage_action = torch.stack(
                (
                    flow.stages.state_score,
                    flow.stages.attention_score,
                    flow.stages.mlp_score,
                ),
                dim=-1,
            )
        return RouteDynamics(
            self.local_window,
            flow.row_position.clone(),
            node,
            edge_register,
            head_transport,
            head_action,
            head_direct,
            head_local,
            integration,
            backward_distance,
            span,
            reuse,
            stage_position,
            stage_presence,
            stage_action,
        )

    @staticmethod
    def head_groups(
        dynamics: RouteDynamics,
        quantile: float = 0.3,
    ) -> tuple[Tensor, Tensor]:
        """Return local/global masks while retaining every head coordinate."""

        if not 0 < quantile < 0.5:
            raise ValueError("head quantile must lie in (0, 0.5)")
        span = dynamics.head_span.flatten()
        local_threshold = torch.quantile(span, quantile)
        global_threshold = torch.quantile(span, 1 - quantile)
        return (
            dynamics.head_span <= local_threshold,
            dynamics.head_span >= global_threshold,
        )

    @staticmethod
    def reanchor_events(
        dynamics: RouteDynamics,
        limit: int = 16,
    ) -> tuple[RouteEvent, ...]:
        """Rank internal evidence rereads; no sentence boundary is supplied.

        A candidate is a local peak of the absolute signed action of messages
        whose exact source endpoint is an evidence token.  The peak remains
        head-specific, and its sign is retained in the returned event.
        """

        if limit < 0:
            raise ValueError("event limit must be non-negative")
        if limit == 0:
            return ()
        action = dynamics.head_direct_evidence[..., 1]
        score = action.abs()
        left = torch.zeros_like(score)
        right = torch.zeros_like(score)
        left[..., 1:] = score[..., :-1]
        right[..., :-1] = score[..., 1:]
        peak = (score > 0) & (score >= left) & (score >= right)
        candidate = torch.nonzero(peak, as_tuple=False)
        if not len(candidate):
            return ()
        value = score[peak]
        order = torch.topk(value, min(limit, len(value)), sorted=True).indices
        events = []
        for layer, head, slot in candidate.index_select(0, order).tolist():
            direct = dynamics.head_direct_evidence[layer, head, slot]
            local = dynamics.head_local_response[layer, head, slot]
            events.append(
                RouteEvent(
                    layer,
                    head,
                    int(dynamics.row_position[slot]),
                    float(abs(direct[1])),
                    float(direct[0]),
                    float(direct[1]),
                    float(local[0]),
                    float(local[1]),
                )
            )
        return tuple(events)

    @staticmethod
    def _top_coordinates(score: Tensor, limit: int) -> Tensor:
        candidate = torch.nonzero(score > 0, as_tuple=False)
        if not len(candidate) or limit == 0:
            return candidate[:0]
        value = score[tuple(candidate.T)]
        order = torch.topk(value, min(limit, len(value)), sorted=True).indices
        return candidate.index_select(0, order)

    def compact_arrays(
        self,
        dynamics: RouteDynamics,
        *,
        response_start: int,
        limit: int = 16,
    ) -> dict[str, object]:
        """Export decisive head coordinates rather than an all-head mean.

        Full ledgers remain available in memory for mechanism plots.  The
        compact artifact retains head spans and the strongest exact nodes for
        re-reading, local support, read-without-use, and repeated reuse.
        """

        if limit < 0:
            raise ValueError("event limit must be non-negative")
        if dynamics.local_window != self.local_window:
            raise ValueError("route dynamics use a different local window")
        events = self.reanchor_events(dynamics, limit)
        local_score = dynamics.head_local_response[..., 1].clamp_min(0)
        local_index = self._top_coordinates(local_score, limit)

        incoming = dynamics.head_transport.sum(-1)
        direct = dynamics.head_direct_evidence
        read_share = torch.where(
            incoming > 0, direct[..., 0] / incoming, torch.zeros_like(incoming)
        )
        action_total = dynamics.head_gradient_action.abs().sum(-1)
        use_share = torch.where(
            action_total > 0,
            direct[..., 1].abs() / action_total,
            torch.zeros_like(action_total),
        )
        read_without_use = (read_share - use_share).clamp_min(0)
        silent_index = self._top_coordinates(read_without_use, limit)

        response_reuse = dynamics.source_reuse[..., 0].clone()
        response_reuse[..., :response_start] = 0
        reuse_index = self._top_coordinates(response_reuse, limit)
        local_head, global_head = self.head_groups(dynamics)

        def coordinate(index: Tensor, axis: int) -> Tensor:
            if len(index):
                return index[:, axis]
            return torch.empty(0, dtype=torch.long)

        def selected(value: Tensor, index: Tensor) -> Tensor:
            if len(index):
                return value[tuple(index.T)]
            return torch.empty(0, dtype=value.dtype)

        event_layer = torch.tensor([item.layer for item in events], dtype=torch.int16)
        event_head = torch.tensor([item.head for item in events], dtype=torch.int16)
        event_position = torch.tensor(
            [item.position for item in events], dtype=torch.int32
        )
        return {
            "route_model_schema": 1,
            "route_channel_name": CHANNEL_NAMES,
            "route_local_window": dynamics.local_window,
            "route_head_span": dynamics.head_span,
            "route_local_head": local_head,
            "route_global_head": global_head,
            "reanchor_event_layer": event_layer,
            "reanchor_event_head": event_head,
            "reanchor_event_position": event_position,
            "reanchor_event_score": torch.tensor(
                [item.score for item in events], dtype=torch.float32
            ),
            "reanchor_event_transport": torch.tensor(
                [item.evidence_transport for item in events], dtype=torch.float32
            ),
            "reanchor_event_gradient_action": torch.tensor(
                [item.evidence_gradient_action for item in events],
                dtype=torch.float32,
            ),
            "local_event_layer": coordinate(local_index, 0).to(torch.int16),
            "local_event_head": coordinate(local_index, 1).to(torch.int16),
            "local_event_position": dynamics.row_position.index_select(
                0, coordinate(local_index, 2)
            ).to(torch.int32),
            "local_event_transport": selected(
                dynamics.head_local_response[..., 0], local_index
            ),
            "local_event_gradient_action": selected(
                dynamics.head_local_response[..., 1], local_index
            ),
            "silent_event_layer": coordinate(silent_index, 0).to(torch.int16),
            "silent_event_head": coordinate(silent_index, 1).to(torch.int16),
            "silent_event_position": dynamics.row_position.index_select(
                0, coordinate(silent_index, 2)
            ).to(torch.int32),
            "silent_event_read_use_gap": selected(
                read_without_use, silent_index
            ),
            "reuse_event_layer": coordinate(reuse_index, 0).to(torch.int16),
            "reuse_event_head": coordinate(reuse_index, 1).to(torch.int16),
            "reuse_event_source": coordinate(reuse_index, 2).to(torch.int32),
            "reuse_event_transport": selected(response_reuse, reuse_index),
            "reuse_event_gradient_action": selected(
                dynamics.source_reuse[..., 1], reuse_index
            ),
            "reanchor_support_peak": float(direct[..., 1].clamp_min(0).max()),
            "reanchor_opposition_peak": float((-direct[..., 1]).clamp_min(0).max()),
            "read_without_use_peak": float(read_without_use.max()),
            "local_reinforcement_peak": float(local_score.max()),
            "response_reuse_peak": float(response_reuse.max()),
        }
