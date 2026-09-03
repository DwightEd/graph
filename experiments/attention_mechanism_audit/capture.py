"""Teacher-forced functional traces from a frozen Llama observer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .schema import (
    BRANCH_NAMES,
    BRANCH_REMOVALS,
    EVIDENCE,
    HISTORY,
    OTHER_PROMPT,
    REGISTER_BRANCH_PAIRS,
    REGISTER_NAMES,
    REGISTER_STAGE_NAMES,
    ROLE_NAMES,
    SELF,
    SHORTCUT_VECTOR_NAMES,
)
from .shortcut import capture_shortcut_geometry


@dataclass
class BranchScores:
    logprob: torch.Tensor
    margin: torch.Tensor


class FunctionalTraceReplay:
    """Capture exact A/V/W_O routes and replay four factorial branches."""

    def __init__(self, model: Any) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.layers = tuple(model.model.layers)
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(getattr(config, "num_key_value_heads", self.heads))
        self.head_dim = self.layers[0].self_attn.q_proj.out_features // self.heads
        if int(getattr(config, "pretraining_tp", 1)) != 1:
            raise ValueError("pretraining_tp must be 1 so value hooks remain exact")
        if self.heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        repeats = self.heads // self.kv_heads
        self.q_to_kv = torch.arange(self.heads, device=self.device) // repeats
        self.output_grams = tuple(self._output_gram(layer) for layer in self.layers)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> FunctionalTraceReplay:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        return cls(model)

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _output_gram(self, layer: Any) -> torch.Tensor:
        weight = layer.self_attn.o_proj.weight.detach().float()
        blocks = weight.reshape(weight.shape[0], self.heads, self.head_dim).permute(
            1, 2, 0
        )
        return blocks @ blocks.transpose(1, 2)

    def _source_norm(self, index: int, value: torch.Tensor) -> torch.Tensor:
        """Return ||W_O[l,h] V[l,s,kv(h)]|| for each source and head."""

        value = value[:, self.q_to_kv].float()
        return (
            torch.einsum("shd,hde,she->sh", value, self.output_grams[index], value)
            .clamp_min(0)
            .sqrt()
        )

    def _removed_write(self, index: int, context: torch.Tensor) -> torch.Tensor:
        """Project selected per-head context through the matching layer W_O."""

        return F.linear(
            context.flatten(-2),
            self.layers[index].self_attn.o_proj.weight,
            bias=None,
        )

    def _target_scores(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        chunk: int,
    ) -> BranchScores:
        logprobs, margins = [], []
        for start in range(0, len(hidden), chunk):
            stop = min(start + chunk, len(hidden))
            logits = self.model.lm_head(hidden[start:stop]).float()
            targets = target_ids[start:stop]
            selected = logits.gather(1, targets[:, None]).squeeze(1)
            competitor = logits.scatter(1, targets[:, None], -torch.inf).max(1).values
            logprobs.append((selected - logits.logsumexp(1)).cpu())
            margins.append((selected - competitor).cpu())
        return BranchScores(torch.cat(logprobs), torch.cat(margins))

    def _removal_mask(
        self,
        removals: tuple[str | None, ...],
        evidence_mask: torch.Tensor,
        response_start: int,
        query_start: int,
        query_stop: int,
    ) -> torch.Tensor:
        source = torch.arange(query_stop, device=self.device)
        query = torch.arange(query_start, query_stop, device=self.device)
        causal = source[None] <= query[:, None]
        response_query = query >= response_start - 1
        evidence = torch.zeros(query_stop, dtype=torch.bool, device=self.device)
        prompt_stop = min(response_start, query_stop)
        evidence[:prompt_stop] = evidence_mask[:prompt_stop].to(self.device)
        response = source >= response_start
        masks = []
        for removal in removals:
            selected = torch.zeros_like(causal)
            if removal in {"evidence", "both"}:
                selected |= (
                    causal
                    & response_query[:, None]
                    & evidence[None]
                    & (source[None] != query[:, None])
                )
            if removal in {"history", "both"}:
                selected |= (
                    causal
                    & response_query[:, None]
                    & response[None]
                    & (source[None] < query[:, None])
                )
            masks.append(selected)
        return torch.stack(masks)

    def _empty_trace(
        self,
        response_tokens: int,
        top_k: int,
    ) -> dict[str, torch.Tensor]:
        layers = len(self.layers)
        head_shape = (layers, response_tokens, self.heads)
        register_route_shape = (layers, response_tokens, len(REGISTER_NAMES))
        register_role_shape = (
            layers,
            response_tokens,
            self.heads,
            len(REGISTER_NAMES),
            len(ROLE_NAMES),
        )
        register_summary_shape = (
            layers,
            response_tokens,
            len(REGISTER_NAMES),
            len(ROLE_NAMES),
        )
        trace = {
            "register_route_source_index": torch.full(
                (*register_route_shape, top_k), -1, dtype=torch.int32
            ),
            "register_route_head_index": torch.full(
                (*register_route_shape, top_k), -1, dtype=torch.int16
            ),
            "register_route_magnitude": torch.zeros(
                *register_route_shape, top_k, dtype=torch.float16
            ),
            "register_route_contribution": torch.zeros(
                *register_route_shape, top_k, dtype=torch.float32
            ),
            "register_route_root_contribution": torch.zeros(
                *register_route_shape, top_k, dtype=torch.float32
            ),
            "register_route_carrier_contribution": torch.zeros(
                *register_route_shape, top_k, dtype=torch.float32
            ),
            "register_route_gate_contribution": torch.zeros(
                *register_route_shape, top_k, dtype=torch.float32
            ),
            "register_route_remainder_magnitude": torch.zeros(
                *register_route_shape, dtype=torch.float16
            ),
            "register_route_remainder_contribution": torch.zeros(
                *register_route_shape, dtype=torch.float32
            ),
            "register_route_remainder_root_contribution": torch.zeros(
                *register_route_shape, dtype=torch.float32
            ),
            "register_route_remainder_carrier_contribution": torch.zeros(
                *register_route_shape, dtype=torch.float32
            ),
            "register_route_remainder_gate_contribution": torch.zeros(
                *register_route_shape, dtype=torch.float32
            ),
            "register_route_cover_size": torch.zeros(
                *register_route_shape, dtype=torch.int32
            ),
            "register_role_mass": torch.zeros(
                *register_role_shape, dtype=torch.float16
            ),
            "register_role_contribution": torch.zeros(
                *register_role_shape, dtype=torch.float32
            ),
            "register_role_root_contribution": torch.zeros(
                *register_role_shape, dtype=torch.float32
            ),
            "register_role_carrier_contribution": torch.zeros(
                *register_role_shape, dtype=torch.float32
            ),
            "register_role_gate_contribution": torch.zeros(
                *register_role_shape, dtype=torch.float32
            ),
            "register_role_effective_routes": torch.zeros(
                *register_summary_shape, dtype=torch.float16
            ),
            "register_norm": torch.zeros(
                layers,
                response_tokens,
                len(REGISTER_NAMES),
                len(REGISTER_STAGE_NAMES),
                dtype=torch.float16,
            ),
            "register_mlp_alignment": torch.zeros(
                layers,
                response_tokens,
                len(REGISTER_NAMES),
                dtype=torch.float32,
            ),
            "register_conservation_error": torch.zeros(
                layers,
                response_tokens,
                len(REGISTER_NAMES),
                dtype=torch.float32,
            ),
            "register_attention_edge_error": torch.zeros(
                layers,
                response_tokens,
                len(REGISTER_NAMES),
                dtype=torch.float32,
            ),
            "register_step_gram": torch.zeros(
                layers,
                response_tokens,
                len(REGISTER_NAMES),
                layers,
                dtype=torch.float32,
            ),
            "interaction_norm": torch.zeros(
                layers,
                response_tokens,
                len(REGISTER_STAGE_NAMES),
                dtype=torch.float16,
            ),
            "final_register_norm": torch.zeros(
                1,
                response_tokens,
                len(REGISTER_NAMES),
                dtype=torch.float32,
            ),
            "shortcut_route_gram": torch.zeros(
                layers,
                response_tokens,
                len(SHORTCUT_VECTOR_NAMES),
                len(SHORTCUT_VECTOR_NAMES),
                dtype=torch.float32,
            ),
            "shortcut_head_gram": torch.zeros(
                layers,
                response_tokens,
                self.heads,
                len(SHORTCUT_VECTOR_NAMES),
                len(SHORTCUT_VECTOR_NAMES),
                dtype=torch.float32,
            ),
            "shortcut_rewire_valid": torch.zeros(
                layers, response_tokens, dtype=torch.bool
            ),
        }
        for family in ("attention", "edge"):
            trace[f"prompt_{family}_effective_sources"] = torch.zeros(
                layers, response_tokens, dtype=torch.float16
            )
            trace[f"prompt_{family}_effective_rank"] = torch.zeros(
                layers, response_tokens, dtype=torch.float16
            )
            trace[f"prompt_{family}_anchor_index"] = torch.full(
                head_shape, -1, dtype=torch.int32
            )
        return trace

    @staticmethod
    def _prompt_carriers(
        mass: torch.Tensor, prompt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Keep the established all-prompt collapse audit head-resolved."""

        selected = mass.float() * prompt[:, None]
        total = selected.sum(-1)
        valid = total > 1e-12
        probability = selected / total.clamp_min(1e-12)[..., None]
        mixture = probability.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        mixture_entropy = -(mixture * mixture.clamp_min(1e-12).log()).sum(-1)
        gram = probability @ probability.transpose(1, 2)
        trace = gram.diagonal(dim1=1, dim2=2).sum(1)
        rank = trace.square() / gram.square().sum((1, 2)).clamp_min(1e-12)
        anchor = selected.argmax(-1).masked_fill(~valid, -1)
        return {
            "effective_sources": mixture_entropy.exp(),
            "effective_rank": rank,
            "anchor_index": anchor,
        }

    @staticmethod
    def _effective_routes(
        mass: torch.Tensor,
        roles: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Effective joint ``(head, source)`` routes for every source role."""

        eps = 1e-12
        routes = []
        for role in roles:
            selected = mass.float() * role[:, None]
            total = selected.sum((1, 2))
            joint = selected / total.clamp_min(eps)[:, None, None]
            joint_entropy = -(joint * joint.clamp_min(eps).log()).sum((1, 2))
            routes.append(joint_entropy.exp().masked_fill(total <= eps, 0))
        return torch.stack(routes, -1)

    @staticmethod
    def _mass_cover(
        mass: torch.Tensor,
        *,
        top_k: int,
        cover_mass: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress each dense head route while retaining its exact leftover mass."""

        magnitude, index = mass.float().sort(dim=-1, descending=True)
        total = magnitude.sum(-1)
        cumulative = magnitude.cumsum(-1)
        required = (cumulative < cover_mass * total[..., None]).sum(-1) + 1
        required = required.masked_fill(total <= 1e-12, 0)
        required = torch.minimum(required, (magnitude > 0).sum(-1))
        width = min(top_k, mass.shape[-1])
        retained = required.clamp_max(width)
        magnitude = magnitude[..., :width]
        index = index[..., :width]
        keep = torch.arange(width, device=mass.device) < retained[..., None]
        kept_magnitude = magnitude * keep
        kept_index = index.masked_fill(~keep, -1)
        remainder = (total - kept_magnitude.sum(-1)).clamp_min(0)
        if width < top_k:
            padding = top_k - width
            kept_magnitude = F.pad(kept_magnitude, (0, padding))
            kept_index = F.pad(kept_index, (0, padding), value=-1)
        return kept_index, kept_magnitude, remainder, required

    @staticmethod
    def _registers(state: torch.Tensor) -> torch.Tensor:
        """Return evidence-adoption and autonomous-history branch differences."""

        return torch.stack((state[0] - state[1], state[1] - state[3]), dim=1)

    @staticmethod
    def _interaction(state: torch.Tensor) -> torch.Tensor:
        return state[0] - state[1] - state[2] + state[3]

    @classmethod
    def _register_statistics(
        cls,
        layer_input: torch.Tensor,
        attention_write: torch.Tensor,
        mlp_write: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Close each finite-difference register through attention and MLP."""

        states = tuple(
            value.float()
            for value in (layer_input, attention_write, mlp_write, layer_output)
        )
        registers = tuple(cls._registers(value) for value in states)
        x, attention, mlp, output = registers
        step = attention + mlp
        error = (output - x - step).norm(dim=-1)
        pre_mlp = x + attention
        denominator = pre_mlp.norm(dim=-1) * mlp.norm(dim=-1)
        alignment = (pre_mlp * mlp).sum(-1) / denominator.clamp_min(1e-12)
        alignment = alignment.masked_fill(denominator <= 1e-12, 0)
        interaction = torch.stack(
            tuple(cls._interaction(value) for value in states), dim=1
        )
        return {
            "norm": torch.stack(registers, dim=2).norm(dim=-1),
            "mlp_alignment": alignment,
            "conservation_error": error,
            "step": step,
            "interaction_norm": interaction.norm(dim=-1),
        }

    def _register_routes(
        self,
        index: int,
        attention: torch.Tensor,
        value: torch.Tensor,
        attention_register: torch.Tensor,
        roles: tuple[torch.Tensor, ...],
        *,
        top_k: int,
        cover_mass: float,
    ) -> dict[str, torch.Tensor]:
        """Measure exact branch-difference writes without materializing d_model edges.

        ``attention`` is the actual post-intervention coefficient tensor
        ``[branch, row, head, source]``.  Per-edge norms use the matching W_O
        quadratic form; signed contributions are projections onto the complete
        register attention write.  The sparse cover is only a serialization
        view: all role totals and reconstruction errors are computed densely.
        """

        weight = self.layers[index].self_attn.o_proj.weight.float()
        gram = self.output_grams[index]
        magnitudes, contributions, edge_errors = [], [], []
        components = {name: [] for name in ("root", "carrier", "gate")}
        for register, (upper, lower) in enumerate(REGISTER_BRANCH_PAIRS):
            a1, a0 = attention[upper].float(), attention[lower].float()
            v1 = value[upper, :, self.q_to_kv].float()
            v0 = value[lower, :, self.q_to_kv].float()
            q11 = torch.einsum("shd,hde,she->hs", v1, gram, v1)
            q00 = torch.einsum("shd,hde,she->hs", v0, gram, v0)
            q10 = torch.einsum("shd,hde,she->hs", v1, gram, v0)
            squared = (
                a1.square() * q11[None]
                + a0.square() * q00[None]
                - 2 * a1 * a0 * q10[None]
            )
            magnitude = squared.clamp_min(0).sqrt()

            complete = attention_register[:, register].float()
            back = F.linear(complete, weight.T).reshape(
                len(complete), self.heads, self.head_dim
            )
            dot1 = torch.einsum("shd,rhd->rhs", v1, back)
            dot0 = torch.einsum("shd,rhd->rhs", v0, back)
            numerator = a1 * dot1 - a0 * dot0
            denominator = complete.square().sum(-1)
            contribution = numerator / denominator[:, None, None].clamp_min(1e-12)
            contribution = contribution.masked_fill(
                (denominator <= 1e-12)[:, None, None], 0
            )
            root_role = EVIDENCE if register == 0 else HISTORY
            root = roles[root_role][:, None]
            nonroot = ~root
            normalizer = denominator[:, None, None].clamp_min(1e-12)
            carrier = (0.5 * (a1 + a0) * (dot1 - dot0) / normalizer).masked_fill(
                ~nonroot, 0
            )
            gate = (0.5 * (a1 - a0) * (dot1 + dot0) / normalizer).masked_fill(
                ~nonroot, 0
            )
            root_contribution = contribution.masked_fill(~root, 0)
            invalid = (denominator <= 1e-12)[:, None, None]
            carrier = carrier.masked_fill(invalid, 0)
            gate = gate.masked_fill(invalid, 0)

            context = torch.einsum("rhs,shd->rhd", a1, v1) - torch.einsum(
                "rhs,shd->rhd", a0, v0
            )
            reconstruction = F.linear(context.flatten(-2), weight)
            edge_errors.append((complete - reconstruction.float()).norm(dim=-1))
            magnitudes.append(magnitude)
            contributions.append(contribution)
            components["root"].append(root_contribution)
            components["carrier"].append(carrier)
            components["gate"].append(gate)

        magnitude = torch.stack(magnitudes, dim=2)
        contribution = torch.stack(contributions, dim=2)
        component = {
            name: torch.stack(values, dim=2) for name, values in components.items()
        }
        flat_magnitude = magnitude.permute(0, 2, 1, 3).flatten(-2)
        flat_contribution = contribution.permute(0, 2, 1, 3).flatten(-2)
        flat_index, _priority, _priority_tail, cover_size = self._mass_cover(
            flat_contribution.abs(), top_k=top_k, cover_mass=cover_mass
        )
        selected = flat_index >= 0
        kept_magnitude = flat_magnitude.gather(-1, flat_index.clamp_min(0))
        kept_magnitude = kept_magnitude.masked_fill(~selected, 0)
        remainder = (flat_magnitude.sum(-1) - kept_magnitude.sum(-1)).clamp_min(0)
        kept_contribution = flat_contribution.gather(-1, flat_index.clamp_min(0))
        kept_contribution = kept_contribution.masked_fill(~selected, 0)
        remainder_contribution = flat_contribution.sum(-1) - kept_contribution.sum(-1)
        kept_components, remainder_components = {}, {}
        for name, component_value in component.items():
            flat = component_value.permute(0, 2, 1, 3).flatten(-2)
            kept = flat.gather(-1, flat_index.clamp_min(0)).masked_fill(~selected, 0)
            kept_components[name] = kept
            remainder_components[name] = flat.sum(-1) - kept.sum(-1)
        source_index = (flat_index % attention.shape[-1]).masked_fill(~selected, -1)
        head_index = (flat_index // attention.shape[-1]).masked_fill(~selected, -1)

        role_mass, role_contribution, role_routes = [], [], []
        role_components = {name: [] for name in component}
        for register in range(len(REGISTER_NAMES)):
            current_mass = magnitude[:, :, register]
            current_contribution = contribution[:, :, register]
            role_mass.append(
                torch.stack(
                    [(current_mass * role[:, None]).sum(-1) for role in roles],
                    dim=-1,
                )
            )
            role_contribution.append(
                torch.stack(
                    [(current_contribution * role[:, None]).sum(-1) for role in roles],
                    dim=-1,
                )
            )
            for name, component_value in component.items():
                current = component_value[:, :, register]
                role_components[name].append(
                    torch.stack(
                        [(current * role[:, None]).sum(-1) for role in roles],
                        dim=-1,
                    )
                )
            role_routes.append(self._effective_routes(current_mass, roles))
        return {
            "source_index": source_index,
            "head_index": head_index,
            "magnitude": kept_magnitude,
            "contribution": kept_contribution,
            **{
                f"{name}_contribution": value for name, value in kept_components.items()
            },
            "remainder_magnitude": remainder,
            "remainder_contribution": remainder_contribution,
            **{
                f"remainder_{name}_contribution": value
                for name, value in remainder_components.items()
            },
            "cover_size": cover_size,
            "role_mass": torch.stack(role_mass, dim=2),
            "role_contribution": torch.stack(role_contribution, dim=2),
            **{
                f"role_{name}_contribution": torch.stack(value, dim=2)
                for name, value in role_components.items()
            },
            "role_effective_routes": torch.stack(role_routes, dim=1),
            "edge_error": torch.stack(edge_errors, dim=1),
        }

    def _forward(
        self,
        token_ids: torch.Tensor,
        response_start: int,
        evidence_mask: torch.Tensor,
        *,
        predictor_chunk: int,
        top_k: int,
        route_cover_mass: float,
        logit_chunk: int,
    ) -> tuple[list[BranchScores], dict[str, torch.Tensor]]:
        ids = token_ids.to(self.device)
        source_tokens = len(ids) - 1
        response_tokens = len(ids) - response_start
        batch = len(BRANCH_REMOVALS)
        dtype = self.model.get_input_embeddings().weight.dtype
        evidence_device = evidence_mask.to(self.device)
        k = min(top_k, self.heads * source_tokens)
        trace = self._empty_trace(response_tokens, k)
        values = [
            torch.empty(
                batch,
                source_tokens,
                self.kv_heads,
                self.head_dim,
                dtype=dtype,
                device=self.device,
            )
            for _ in self.layers
        ]
        source_norms = [
            torch.empty(
                source_tokens,
                self.heads,
                dtype=torch.float32,
                device=self.device,
            )
            for _ in self.layers
        ]
        layer_inputs: list[torch.Tensor | None] = [None] * len(self.layers)
        attention_writes: list[torch.Tensor | None] = [None] * len(self.layers)
        mlp_writes: list[torch.Tensor | None] = [None] * len(self.layers)
        register_steps = torch.empty(
            len(self.layers),
            response_tokens,
            len(REGISTER_NAMES),
            int(self.model.config.hidden_size),
            dtype=torch.float32,
            device="cpu",
        )
        logprob = torch.empty(batch, response_tokens)
        margin = torch.empty(batch, response_tokens)
        query_start = query_stop = 0
        remove_mask = torch.empty(0, device=self.device)
        has_removal = False
        handles = []

        def response_window() -> tuple[int, int, int] | None:
            local = max(response_start - 1 - query_start, 0)
            if query_start + local >= query_stop:
                return None
            row_start = query_start + local - (response_start - 1)
            return local, row_start, row_start + query_stop - query_start - local

        def layer_input_hook(index: int):
            def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
                layer_inputs[index] = args[0]

            return hook

        def v_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                current = output.reshape(
                    batch, query_stop - query_start, self.kv_heads, self.head_dim
                )
                values[index][:, query_start:query_stop].copy_(current)
                source_norms[index][query_start:query_stop] = self._source_norm(
                    index, current[0]
                )

            return hook

        def attention_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any):
                parts = list(output)
                attention = parts[1]
                if has_removal:
                    value_by_head = values[index][:, :query_stop, self.q_to_kv]
                    removed_context = torch.einsum(
                        "bhqs,bshd->bqhd",
                        attention * remove_mask[:, None],
                        value_by_head,
                    )
                    parts[0] = parts[0] - self._removed_write(index, removed_context)
                attention_writes[index] = parts[0]

                window = response_window()
                if window is not None:
                    local, row_start, row_stop = window
                    branch_attention = attention[:, :, local:, :query_stop].permute(
                        0, 2, 1, 3
                    )
                    actual_attention = (
                        branch_attention
                        * (~remove_mask[:, local:, :query_stop])[:, :, None]
                    )
                    full_attention = actual_attention[0]
                    attention_register = self._registers(parts[0][:, local:])
                    source = torch.arange(query_stop, device=self.device)
                    query = torch.arange(
                        query_start + local, query_stop, device=self.device
                    )
                    edge = (
                        full_attention.float()
                        * source_norms[index][:query_stop].T[None]
                    )
                    self_source = source[None] == query[:, None]
                    evidence = torch.zeros(
                        query_stop, dtype=torch.bool, device=self.device
                    )
                    prompt_stop = min(response_start, query_stop)
                    evidence[:prompt_stop] = evidence_device[:prompt_stop]
                    roles = (
                        evidence[None] & ~self_source,
                        (source[None] < response_start)
                        & ~evidence[None]
                        & ~self_source,
                        (source[None] >= response_start)
                        & (source[None] < query[:, None]),
                        self_source,
                    )
                    shortcut = capture_shortcut_geometry(
                        actual_attention,
                        values[index][:, :query_stop],
                        roles,
                        q_to_kv=self.q_to_kv,
                        output_weight=self.layers[index].self_attn.o_proj.weight,
                        output_gram=self.output_grams[index],
                    )
                    for name, value in shortcut.items():
                        target = trace[f"shortcut_{name}"][
                            index, row_start:row_stop
                        ]
                        target.copy_(
                            value.detach().to(dtype=target.dtype, device="cpu")
                        )
                    register_route = self._register_routes(
                        index,
                        actual_attention,
                        values[index][:, :query_stop],
                        attention_register,
                        roles,
                        top_k=k,
                        cover_mass=route_cover_mass,
                    )

                    for name, value in register_route.items():
                        if name == "edge_error":
                            target = trace["register_attention_edge_error"][
                                index, row_start:row_stop
                            ]
                        else:
                            target = (
                                trace[f"register_route_{name}"][
                                    index, row_start:row_stop
                                ]
                                if name
                                in {
                                    "source_index",
                                    "head_index",
                                    "magnitude",
                                    "contribution",
                                    "root_contribution",
                                    "carrier_contribution",
                                    "gate_contribution",
                                    "remainder_magnitude",
                                    "remainder_contribution",
                                    "remainder_root_contribution",
                                    "remainder_carrier_contribution",
                                    "remainder_gate_contribution",
                                    "cover_size",
                                }
                                else trace[f"register_{name}"][
                                    index, row_start:row_stop
                                ]
                            )
                        target.copy_(
                            value.detach().to(dtype=target.dtype, device="cpu")
                        )
                    for family, mass in (
                        ("attention", full_attention),
                        ("edge", edge),
                    ):
                        prompt_statistics = self._prompt_carriers(
                            mass, roles[EVIDENCE] | roles[OTHER_PROMPT]
                        )
                        for name, value in prompt_statistics.items():
                            target = trace[f"prompt_{family}_{name}"][
                                index, row_start:row_stop
                            ]
                            target.copy_(
                                value.detach().to(dtype=target.dtype, device="cpu")
                            )

                parts[1] = None
                return tuple(parts)

            return hook

        def mlp_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                mlp_writes[index] = output

            return hook

        def layer_output_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                window = response_window()
                layer_input = layer_inputs[index]
                attention_write = attention_writes[index]
                mlp_write = mlp_writes[index]
                if layer_input is None or attention_write is None or mlp_write is None:
                    raise RuntimeError(
                        "decoder hooks did not capture the register state"
                    )
                if window is not None:
                    local, row_start, row_stop = window
                    layer_output = output[0] if isinstance(output, tuple) else output
                    statistics = self._register_statistics(
                        layer_input[:, local:],
                        attention_write[:, local:],
                        mlp_write[:, local:],
                        layer_output[:, local:],
                    )
                    for name, value in statistics.items():
                        if name == "step":
                            register_steps[index, row_start:row_stop].copy_(
                                value.detach().float().cpu()
                            )
                        else:
                            target = trace[
                                "interaction_norm"
                                if name == "interaction_norm"
                                else f"register_{name}"
                            ][index, row_start:row_stop]
                            target.copy_(
                                value.detach().to(dtype=target.dtype, device="cpu")
                            )
                layer_inputs[index] = None
                attention_writes[index] = None
                mlp_writes[index] = None

            return hook

        def final_norm_hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
            window = response_window()
            if window is None:
                return
            local, row_start, row_stop = window
            trace["final_register_norm"][0, row_start:row_stop].copy_(
                self._registers(output[:, local:]).float().norm(dim=-1).detach().cpu()
            )

        for index, layer in enumerate(self.layers):
            handles.extend(
                (
                    layer.register_forward_pre_hook(layer_input_hook(index)),
                    layer.self_attn.v_proj.register_forward_hook(v_hook(index)),
                    layer.self_attn.register_forward_hook(attention_hook(index)),
                    layer.mlp.register_forward_hook(mlp_hook(index)),
                    layer.register_forward_hook(layer_output_hook(index)),
                )
            )
        handles.append(self.model.model.norm.register_forward_hook(final_norm_hook))

        try:
            from transformers.cache_utils import DynamicCache

            past = DynamicCache()
            with torch.inference_mode():
                for query_start in range(0, source_tokens, predictor_chunk):
                    query_stop = min(query_start + predictor_chunk, source_tokens)
                    remove_mask = self._removal_mask(
                        BRANCH_REMOVALS,
                        evidence_mask,
                        response_start,
                        query_start,
                        query_stop,
                    )
                    has_removal = bool(remove_mask.any())
                    output = self.model.model(
                        input_ids=ids[None, query_start:query_stop].expand(batch, -1),
                        attention_mask=torch.ones(
                            batch, query_stop, dtype=torch.long, device=self.device
                        ),
                        past_key_values=past,
                        use_cache=True,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    past = output.past_key_values
                    window = response_window()
                    if window is None:
                        continue
                    local, row_start, row_stop = window
                    hidden = output.last_hidden_state[:, local:]
                    targets = ids[
                        response_start + row_start : response_start + row_stop
                    ]
                    for branch in range(batch):
                        score = self._target_scores(
                            hidden[branch], targets, chunk=logit_chunk
                        )
                        logprob[branch, row_start:row_stop] = score.logprob
                        margin[branch, row_start:row_stop] = score.margin
        finally:
            for handle in handles:
                handle.remove()

        trace["register_step_gram"].copy_(
            torch.einsum("ltrd,ktrd->ltrk", register_steps, register_steps)
        )

        return [
            BranchScores(logprob[index], margin[index]) for index in range(batch)
        ], trace

    def capture(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        response_start: int,
        evidence_mask: Sequence[bool] | torch.Tensor,
        *,
        predictor_chunk: int = 64,
        top_k: int = 32,
        logit_chunk: int = 64,
        route_cover_mass: float = 0.8,
    ) -> dict[str, Any]:
        """Capture all formal branches once at predictor q=P-1+t."""

        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
        evidence_mask = torch.as_tensor(evidence_mask, dtype=torch.bool, device="cpu")
        if token_ids.ndim != 1 or not 0 < response_start < len(token_ids):
            raise ValueError("token_ids/response_start do not define a response")
        if evidence_mask.shape != (response_start,):
            raise ValueError("evidence_mask must align exactly with the prompt")
        if min(predictor_chunk, top_k, logit_chunk) <= 0:
            raise ValueError("capture chunk sizes and top_k must be positive")
        if not 0 < route_cover_mass <= 1:
            raise ValueError("route_cover_mass must be in (0, 1]")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        scores, trace = self._forward(
            token_ids,
            response_start,
            evidence_mask,
            predictor_chunk=predictor_chunk,
            top_k=top_k,
            route_cover_mass=route_cover_mass,
            logit_chunk=logit_chunk,
        )
        full, no_evidence, no_history, no_evidence_history = scores
        peak = (
            int(torch.cuda.max_memory_reserved(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return {
            "token_ids": token_ids,
            "response_start": int(response_start),
            "evidence_mask": evidence_mask,
            "trace": trace,
            "score_inputs": {
                "full_logprob": full.logprob,
                "full_margin": full.margin,
                "no_evidence_logprob": no_evidence.logprob,
                "no_evidence_margin": no_evidence.margin,
                "no_history_logprob": no_history.logprob,
                "no_history_margin": no_history.margin,
                "no_evidence_history_logprob": no_evidence_history.logprob,
                "no_evidence_history_margin": no_evidence_history.margin,
            },
            "peak_cuda_reserved_bytes": peak,
        }
