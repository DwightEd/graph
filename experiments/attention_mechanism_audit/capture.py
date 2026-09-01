"""Teacher-forced functional traces from a frozen Llama observer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROLE_NAMES = ("evidence", "other_prompt", "response_history", "predictor_self")
EVIDENCE, OTHER_PROMPT, HISTORY, SELF = range(len(ROLE_NAMES))
ROUTE_ROLE_NAMES = ("evidence", "response_history")

BRANCH_NAMES = ("full", "no_evidence", "no_history", "no_evidence_history")
BRANCH_REMOVALS = (None, "evidence", "history", "both")
PATHWAY_CONTRAST_NAMES = ("evidence", "history", "interaction")
PATHWAY_STAGE_NAMES = ("input", "attention", "pre_mlp", "mlp", "output")


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
        role_shape = (*head_shape, len(ROLE_NAMES))
        route_shape = (*head_shape, len(ROUTE_ROLE_NAMES))
        role_summary_shape = (layers, response_tokens, len(ROLE_NAMES))
        trace = {
            "attention_role_mass": torch.empty(*role_shape, dtype=torch.float16),
            "edge_role_mass": torch.empty(*role_shape, dtype=torch.float16),
            "head_role_write_norm": torch.empty(*role_shape, dtype=torch.float16),
            "role_head_coherence": torch.empty(
                *role_summary_shape, dtype=torch.float16
            ),
            "route_source_index": torch.full(
                (*route_shape, top_k), -1, dtype=torch.int32
            ),
            "route_source_magnitude": torch.zeros(
                *route_shape, top_k, dtype=torch.float16
            ),
            "route_source_remainder": torch.zeros(*route_shape, dtype=torch.float16),
            "route_source_cover_size": torch.zeros(*route_shape, dtype=torch.int32),
            "pathway_effect_norm": torch.empty(
                layers,
                response_tokens,
                len(PATHWAY_CONTRAST_NAMES),
                len(PATHWAY_STAGE_NAMES),
                dtype=torch.float16,
            ),
            "pathway_mlp_projection": torch.empty(
                layers,
                response_tokens,
                len(PATHWAY_CONTRAST_NAMES),
                dtype=torch.float32,
            ),
            "pathway_pre_output_cosine": torch.empty(
                layers,
                response_tokens,
                len(PATHWAY_CONTRAST_NAMES),
                dtype=torch.float32,
            ),
            "pathway_pre_output_gain": torch.empty(
                layers,
                response_tokens,
                len(PATHWAY_CONTRAST_NAMES),
                dtype=torch.float32,
            ),
            "pathway_valid": torch.empty(
                layers,
                response_tokens,
                len(PATHWAY_CONTRAST_NAMES),
                dtype=torch.bool,
            ),
            "pathway_cosine_valid": torch.empty(
                layers,
                response_tokens,
                len(PATHWAY_CONTRAST_NAMES),
                dtype=torch.bool,
            ),
            "pathway_residual_error": torch.empty(
                layers, response_tokens, len(BRANCH_NAMES), dtype=torch.float16
            ),
        }
        for family in ("attention", "edge"):
            trace[f"{family}_role_source_entropy"] = torch.empty(
                *role_shape, dtype=torch.float16
            )
            trace[f"{family}_role_top1"] = torch.empty(*role_shape, dtype=torch.float16)
            trace[f"{family}_role_anchor_index"] = torch.full(
                role_shape, -1, dtype=torch.int32
            )
            trace[f"{family}_role_effective_rank"] = torch.empty(
                *role_summary_shape, dtype=torch.float16
            )
            trace[f"{family}_role_effective_routes"] = torch.empty(
                *role_summary_shape, dtype=torch.float16
            )
        return trace

    @staticmethod
    def _routing_statistics(
        mass: torch.Tensor,
        roles: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Preserve head identity while summarizing dense sources within roles."""

        eps = 1e-12
        entropies, top1s, anchors, ranks, routes = [], [], [], [], []
        for role in roles:
            selected = mass.float() * role[:, None]
            head_total = selected.sum(-1)
            valid_head = head_total > eps
            conditional = selected / head_total.clamp_min(eps)[..., None]
            entropy = -(conditional * conditional.clamp_min(eps).log()).sum(-1)
            top1, anchor = selected.max(-1)
            top1 = top1 / head_total.clamp_min(eps)
            anchor = anchor.masked_fill(~valid_head, -1)

            total = selected.sum((1, 2))
            valid_role = total > eps
            joint = selected / total.clamp_min(eps)[:, None, None]
            gram = joint @ joint.transpose(1, 2)
            squared_mass = joint.square().sum((1, 2))
            effective_rank = squared_mass.square() / gram.square().sum(
                (1, 2)
            ).clamp_min(eps)
            joint_entropy = -(joint * joint.clamp_min(eps).log()).sum((1, 2))

            entropies.append(entropy.masked_fill(~valid_head, 0))
            top1s.append(top1.masked_fill(~valid_head, 0))
            anchors.append(anchor)
            ranks.append(effective_rank.masked_fill(~valid_role, 0))
            routes.append(joint_entropy.exp().masked_fill(~valid_role, 0))
        return {
            "source_entropy": torch.stack(entropies, -1),
            "top1": torch.stack(top1s, -1),
            "anchor_index": torch.stack(anchors, -1),
            "effective_rank": torch.stack(ranks, -1),
            "effective_routes": torch.stack(routes, -1),
        }

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
    def _factorial_effects(state: torch.Tensor) -> torch.Tensor:
        """Return evidence, history, and interaction vectors from four branches."""

        full, no_evidence, no_history, no_both = state
        evidence = 0.5 * ((full - no_evidence) + (no_history - no_both))
        history = 0.5 * ((full - no_history) + (no_evidence - no_both))
        interaction = full - no_evidence - no_history + no_both
        return torch.stack((evidence, history, interaction), dim=1)

    @classmethod
    def _pathway_statistics(
        cls,
        layer_input: torch.Tensor,
        attention_write: torch.Tensor,
        mlp_write: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Measure where each causal contrast enters and how the MLP changes it."""

        pre_mlp_native = layer_input + attention_write
        reconstruction = pre_mlp_native + mlp_write
        residual_error = (layer_output - reconstruction).float().norm(dim=-1)
        x = layer_input.float()
        attention = attention_write.float()
        pre_mlp = pre_mlp_native.float()
        mlp = mlp_write.float()
        output = layer_output.float()
        stages = torch.stack((x, attention, pre_mlp, mlp, output), dim=2)
        effects = cls._factorial_effects(stages)
        pre = effects[:, :, PATHWAY_STAGE_NAMES.index("pre_mlp")]
        mlp_effect = effects[:, :, PATHWAY_STAGE_NAMES.index("mlp")]
        out = effects[:, :, PATHWAY_STAGE_NAMES.index("output")]
        pre_norm = pre.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        scale = effects.norm(dim=-1).amax(dim=-1)
        threshold = torch.maximum(torch.full_like(scale, 1e-6), scale * 1e-6)
        valid = pre_norm > threshold
        cosine_valid = valid & (out_norm > threshold)
        safe_pre = pre_norm.clamp_min(threshold)
        safe_out = out_norm.clamp_min(threshold)
        projection = (mlp_effect * pre).sum(-1) / safe_pre.square()
        cosine = (pre * out).sum(-1) / (safe_pre * safe_out)
        gain = out_norm / safe_pre
        projection = projection.masked_fill(~valid, 0)
        cosine = cosine.masked_fill(~cosine_valid, 0)
        gain = gain.masked_fill(~valid, 0)
        return {
            "effect_norm": effects.norm(dim=-1),
            "mlp_projection": projection,
            "pre_output_cosine": cosine,
            "pre_output_gain": gain,
            "valid": valid,
            "cosine_valid": cosine_valid,
            "residual_error": residual_error.transpose(0, 1),
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
        k = min(top_k, source_tokens)
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
                    full_attention = attention[0, :, local:, :query_stop].permute(
                        1, 0, 2
                    )
                    edge = (
                        full_attention.float()
                        * source_norms[index][:query_stop].T[None]
                    )
                    full_value = values[index][0, :query_stop, self.q_to_kv]
                    source = torch.arange(query_stop, device=self.device)
                    query = torch.arange(
                        query_start + local, query_stop, device=self.device
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

                    role_attention = torch.stack(
                        [
                            (full_attention.float() * role[:, None]).sum(-1)
                            for role in roles
                        ],
                        dim=-1,
                    )
                    edge_role = torch.stack(
                        [(edge * role[:, None]).sum(-1) for role in roles], dim=-1
                    )
                    contexts = [
                        torch.einsum(
                            "rhs,shd->rhd",
                            full_attention.to(full_value.dtype) * role[:, None],
                            full_value,
                        )
                        for role in roles
                    ]
                    head_write_norm = torch.stack(
                        [
                            torch.einsum(
                                "rhd,hde,rhe->rh",
                                context.float(),
                                self.output_grams[index],
                                context.float(),
                            )
                            .clamp_min(0)
                            .sqrt()
                            for context in contexts
                        ],
                        dim=-1,
                    )
                    net_write_norm = torch.stack(
                        [
                            self._removed_write(index, context).float().norm(dim=-1)
                            for context in contexts
                        ],
                        dim=-1,
                    )
                    coherence = net_write_norm / head_write_norm.sum(1).clamp_min(1e-12)
                    core_role_edge = torch.stack(
                        (
                            edge * roles[EVIDENCE][:, None],
                            edge * roles[HISTORY][:, None],
                        ),
                        dim=2,
                    )
                    route = self._mass_cover(
                        core_role_edge, top_k=k, cover_mass=route_cover_mass
                    )

                    trace["attention_role_mass"][index, row_start:row_stop] = (
                        role_attention.detach().half().cpu()
                    )
                    trace["edge_role_mass"][index, row_start:row_stop] = (
                        edge_role.detach().half().cpu()
                    )
                    trace["head_role_write_norm"][index, row_start:row_stop] = (
                        head_write_norm.detach().half().cpu()
                    )
                    trace["role_head_coherence"][index, row_start:row_stop] = (
                        coherence.detach().half().cpu()
                    )
                    for name, value in zip(
                        (
                            "route_source_index",
                            "route_source_magnitude",
                            "route_source_remainder",
                            "route_source_cover_size",
                        ),
                        route,
                        strict=True,
                    ):
                        target = trace[name][index, row_start:row_stop]
                        if name in {
                            "route_source_index",
                            "route_source_cover_size",
                        }:
                            target.copy_(value.detach().int().cpu())
                        else:
                            target.copy_(value.detach().half().cpu())
                    for family, mass in (
                        ("attention", full_attention),
                        ("edge", edge),
                    ):
                        statistics = self._routing_statistics(mass, roles)
                        for name, value in statistics.items():
                            target = trace[f"{family}_role_{name}"][
                                index, row_start:row_stop
                            ]
                            if name == "anchor_index":
                                target.copy_(value.detach().int().cpu())
                            else:
                                target.copy_(value.detach().half().cpu())

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
                        "decoder hooks did not capture the pathway state"
                    )
                if window is not None:
                    local, row_start, row_stop = window
                    layer_output = output[0] if isinstance(output, tuple) else output
                    statistics = self._pathway_statistics(
                        layer_input[:, local:],
                        attention_write[:, local:],
                        mlp_write[:, local:],
                        layer_output[:, local:],
                    )
                    for name, value in statistics.items():
                        target = trace[f"pathway_{name}"][index, row_start:row_stop]
                        target.copy_(
                            value.detach().to(dtype=target.dtype, device="cpu")
                        )
                layer_inputs[index] = None
                attention_writes[index] = None
                mlp_writes[index] = None

            return hook

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
        top_k: int = 8,
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
