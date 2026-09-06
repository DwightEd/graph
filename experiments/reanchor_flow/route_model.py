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
EDGE_CHUNK = 4096


@dataclass(frozen=True)
class RouteDynamics:
    """The non-averaged state returned by :class:`HeadResolvedRouteModel`.

    Shapes use ``L`` layers, ``H`` heads, ``P`` represented destination rows,
    ``N`` causal token positions, ``E`` retained edges, and four provenance
    channels.  ``head_*`` tensors always retain both ``L`` and ``H`` axes.
    """

    root_unit_id: int
    local_window: int
    row_position: Tensor
    node_register: Tensor  # [L + 1, N, 4]
    edge_register: Tensor  # [E, 4]
    head_transport: Tensor  # [L, H, P, 4]
    # Native edge gradient action allocated by the transport provenance model;
    # this is not an exact semantic decomposition of a nonlinear hidden state.
    head_gradient_action: Tensor  # [L, H, P, 4]
    # Direct contribution from the selected explanatory root only.
    head_direct_evidence: Tensor  # [L, H, P, 2]: transport, action
    head_integration: Tensor  # [L, H, P, 4]: budget, net, coherence, action
    cross_head_vector_coherence: Tensor  # [L, P]
    cross_head_functional_agreement: Tensor  # [L, P]
    layer_integration: Tensor  # [L, P, 4]: budget, net, coherence, action
    evidence_source_reuse: Tensor  # [L, H, N, 2]: future transport, action
    response_source_reuse: Tensor  # [L, H, N, 2]: future transport, action
    stage_position: Tensor  # [A]
    stage_displacement: Tensor  # [L, A, 3]: residual, attention write, MLP write
    stage_gradient_action: Tensor  # [L, A, 3]
    attention_mlp_vector_cosine: Tensor  # [L, A]
    attention_mlp_functional_agreement: Tensor  # [L, A]
    state_continuity: Tensor  # [L, A]


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
            flow.edges.layer.long() * heads + flow.edges.head.long()
        ) * rows + edge_slot
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

    @staticmethod
    def _selected_root(
        flow: PairedFlow,
        world: NativeWorld,
        root_unit_id: int | None,
    ) -> tuple[int, Tensor]:
        """Resolve one root and verify that every ledger uses its native cut."""

        cut_mask = flow.corrupt_source_mask
        if cut_mask is None:
            raise ValueError("route analysis requires a native root-cut cache")
        cut_mask = cut_mask.to(dtype=torch.bool, device="cpu")
        token_unit_id = world.units.token_unit_id.long()
        if cut_mask.shape != token_unit_id.shape:
            raise ValueError("root-cut mask does not match the source-token axis")
        if root_unit_id is None:
            cut_units = torch.unique(token_unit_id[cut_mask])
            if len(cut_units) != 1:
                raise ValueError("root_unit_id is required for an ambiguous root cut")
            root_unit_id = int(cut_units[0])
        root_unit_id = int(root_unit_id)
        if root_unit_id not in world.evidence_unit_id:
            raise ValueError("selected root is not an evidence unit")
        root_mask = token_unit_id == root_unit_id
        root_mask &= torch.arange(len(token_unit_id)) < world.response_start
        if not bool(root_mask.any()):
            raise ValueError("selected root has no prompt source token")
        if not torch.equal(cut_mask, root_mask):
            raise ValueError("selected root does not match the root-cut cache")
        return root_unit_id, root_mask

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
            # Unrepresented destinations have residual probability one in the
            # transport law.  Preserve that identity so a later-layer edge can
            # still read an evidence-bearing prompt state.
        return node, edge_register

    @staticmethod
    def _head_integration(
        model,
        flow: PairedFlow,
        edge_slot: Tensor,
        layers: int,
        heads: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Aggregate real source-cut ``W_O(A V)`` deltas within each head."""

        rows = len(flow.row_position)
        edges = flow.edges
        budget = HeadResolvedRouteModel._scatter_head(
            flow, edge_slot, edges.delta_message_norm, layers, heads
        )
        action = HeadResolvedRouteModel._scatter_head(
            flow,
            edge_slot,
            torch.nan_to_num(edges.clean_target_score - edges.corrupt_target_score),
            layers,
            heads,
        )
        net = torch.zeros_like(budget)
        cross_head_vector = torch.zeros(layers, rows)
        head_dim = edges.clean_code.shape[1]
        gram_cache = model_gram_cache(model)
        for layer, module in enumerate(model.model.layers):
            selected = torch.nonzero(edges.layer == layer, as_tuple=False).flatten()
            if not len(selected):
                continue
            code_sum = torch.zeros(heads * rows, head_dim)
            for begin in range(0, len(selected), EDGE_CHUNK):
                current = selected[begin : begin + EDGE_CHUNK]
                local_head = edges.head.index_select(0, current).long()
                local_slot = edge_slot.index_select(0, current)
                index = local_head * rows + local_slot
                clean_code = edges.clean_code.index_select(0, current).float()
                corrupt_code = edges.corrupt_code.index_select(0, current).float()
                code_sum.index_add_(0, index, clean_code - corrupt_code)
            code_sum = code_sum.view(heads, rows, head_dim)
            gram = gram_cache.get(layer)
            if gram is None:
                gram = output_gram(
                    module.self_attn.o_proj.weight.detach(), heads, head_dim
                )
                gram_cache[layer] = gram
            squared = torch.einsum("hrd,hde,hre->hr", code_sum, gram.float(), code_sum)
            net[layer] = squared.clamp_min(0).sqrt()
            output = module.self_attn.o_proj.weight.detach()
            joined_code = code_sum.permute(1, 0, 2).reshape(rows, -1)
            joined_vector = joined_code.to(output.device) @ output.float().T
            cross_head_vector[layer] = joined_vector.norm(dim=-1).cpu()
        coherence = torch.where(budget > 0, net / budget, torch.zeros_like(net))
        head_norm = net.sum(dim=1)
        vector_coherence = torch.where(
            head_norm > 0,
            cross_head_vector / head_norm,
            torch.zeros_like(head_norm),
        )
        action_total = action.abs().sum(dim=1)
        functional_agreement = torch.where(
            action_total > 0,
            action.sum(dim=1).abs() / action_total,
            torch.zeros_like(action_total),
        )
        return (
            torch.stack((budget, net, coherence, action), dim=-1),
            vector_coherence,
            functional_agreement,
        )

    @staticmethod
    def _cosine(left: Tensor, right: Tensor) -> Tensor:
        denominator = left.norm(dim=-1) * right.norm(dim=-1)
        dot = (left * right).sum(dim=-1)
        return torch.where(denominator > 0, dot / denominator, 0)

    @classmethod
    def _stage_dynamics(
        cls,
        flow: PairedFlow,
        layers: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Read module interaction and continuity from the selected-root run pair.

        These tensors describe stability only.  They do not establish that a
        stable state is evidence-grounded.
        """

        stages = flow.stages
        if stages is None:
            empty_stage = torch.empty((layers, 0, 3))
            empty_metric = torch.empty((layers, 0))
            return (
                torch.empty(0, dtype=torch.long),
                empty_stage,
                empty_stage.clone(),
                empty_metric,
                empty_metric.clone(),
                empty_metric.clone(),
            )
        position = stages.position.long()
        module_cosine, continuity = [], []
        for layer in range(layers):
            clean_attention = flow.clean_cache.attention_write[layer].index_select(
                0, position
            )
            cut_attention = flow.corrupt_cache.attention_write[layer].index_select(
                0, position
            )
            clean_mlp = flow.clean_cache.mlp_write[layer].index_select(0, position)
            cut_mlp = flow.corrupt_cache.mlp_write[layer].index_select(0, position)
            attention_delta = clean_attention.float() - cut_attention.float()
            mlp_delta = clean_mlp.float() - cut_mlp.float()
            module_cosine.append(cls._cosine(attention_delta, mlp_delta))

            clean_state = flow.clean_cache.layer_input[layer].index_select(0, position)
            cut_state = flow.corrupt_cache.layer_input[layer].index_select(0, position)
            if layer + 1 < layers:
                clean_next = flow.clean_cache.layer_input[layer + 1]
                cut_next = flow.corrupt_cache.layer_input[layer + 1]
            else:
                clean_next = flow.clean_cache.final_hidden
                cut_next = flow.corrupt_cache.final_hidden
            state_delta = clean_state.float() - cut_state.float()
            next_delta = clean_next.index_select(0, position).float()
            next_delta -= cut_next.index_select(0, position).float()
            continuity.append(cls._cosine(state_delta, next_delta))

        displacement = torch.stack(
            (
                stages.state_delta_norm,
                stages.attention_delta_norm,
                stages.mlp_delta_norm,
            ),
            dim=-1,
        )
        action = torch.stack(
            (stages.state_score, stages.attention_score, stages.mlp_score),
            dim=-1,
        )
        module_action = action[..., 1:]
        action_budget = module_action.abs().sum(dim=-1)
        functional_agreement = torch.where(
            action_budget > 0,
            module_action.sum(dim=-1).abs() / action_budget,
            torch.zeros_like(action_budget),
        )
        return (
            position.clone(),
            displacement,
            action,
            torch.stack(module_cosine),
            functional_agreement,
            torch.stack(continuity),
        )

    def analyze(
        self,
        model,
        flow: PairedFlow,
        world: NativeWorld,
        root_unit_id: int | None = None,
    ) -> RouteDynamics:
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
        root_unit_id, root_mask = self._selected_root(flow, world, root_unit_id)
        _, edge_slot = self._slots(flow, tokens)
        edge_probability, residual_probability = transition_probabilities(flow, tokens)
        node, edge_register = self._provenance(
            flow, world, edge_probability, residual_probability
        )
        head_transport = self._scatter_head(
            flow, edge_slot, edge_register, layers, heads
        )

        native_action = torch.nan_to_num(flow.edges.clean_target_score.float())
        source_register = node[flow.edges.layer.long(), flow.edges.source.long()]
        edge_action = source_register * native_action[:, None]
        head_action = self._scatter_head(flow, edge_slot, edge_action, layers, heads)

        direct = root_mask.index_select(0, flow.edges.source.long())
        direct_origin = source_register[:, EVIDENCE]
        direct_values = torch.stack(
            (
                edge_probability * direct_origin,
                native_action * direct_origin,
            ),
            dim=-1,
        )
        direct_values *= direct[:, None]
        head_direct = self._scatter_head(flow, edge_slot, direct_values, layers, heads)

        distance = flow.edges.target.long() - flow.edges.source.long()
        reuse = torch.zeros(layers * heads * tokens, 2, 2)
        future = distance > 0
        reuse_index = (
            flow.edges.layer.long() * heads + flow.edges.head.long()
        ) * tokens + flow.edges.source.long()
        origin = source_register[:, (EVIDENCE, RESPONSE)]
        reuse_value = torch.stack(
            (
                edge_probability[:, None] * origin,
                native_action[:, None] * origin,
            ),
            dim=-1,
        )
        reuse.index_add_(0, reuse_index[future], reuse_value[future])
        reuse = reuse.view(layers, heads, tokens, 2, 2)

        integration, cross_head_vector, cross_head_function = self._head_integration(
            model, flow, edge_slot, layers, heads
        )
        head_net = integration[..., 1].sum(dim=1)
        layer_integration = torch.stack(
            (
                integration[..., 0].sum(dim=1),
                cross_head_vector * head_net,
                cross_head_vector,
                integration[..., 3].sum(dim=1),
            ),
            dim=-1,
        )
        (
            stage_position,
            stage_displacement,
            stage_action,
            module_cosine,
            module_agreement,
            state_continuity,
        ) = self._stage_dynamics(flow, layers)
        return RouteDynamics(
            root_unit_id=root_unit_id,
            local_window=self.local_window,
            row_position=flow.row_position.clone(),
            node_register=node,
            edge_register=edge_register,
            head_transport=head_transport,
            head_gradient_action=head_action,
            head_direct_evidence=head_direct,
            head_integration=integration,
            cross_head_vector_coherence=cross_head_vector,
            cross_head_functional_agreement=cross_head_function,
            layer_integration=layer_integration,
            evidence_source_reuse=reuse[..., 0, :],
            response_source_reuse=reuse[..., 1, :],
            stage_position=stage_position,
            stage_displacement=stage_displacement,
            stage_gradient_action=stage_action,
            attention_mlp_vector_cosine=module_cosine,
            attention_mlp_functional_agreement=module_agreement,
            state_continuity=state_continuity,
        )
