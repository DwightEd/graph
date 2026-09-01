"""Chunked teacher-forced message traces from a frozen Llama observer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROLE_NAMES = ("evidence", "other_prompt", "response_history", "predictor_self")
EVIDENCE, OTHER_PROMPT, HISTORY, SELF = range(len(ROLE_NAMES))


@dataclass
class BranchScores:
    logprob: torch.Tensor
    margin: torch.Tensor


class FunctionalTraceReplay:
    """Aggregate dynamic A/V/W_O messages and replay causal deletions."""

    def __init__(self, model: Any) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.layers = tuple(model.model.layers)
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(getattr(config, "num_key_value_heads", self.heads))
        self.head_dim = self.layers[0].self_attn.q_proj.out_features // self.heads
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
        """Return ``||W_O[h] V[s, kv(h)]||`` for every source and query head."""

        value = value[:, self.q_to_kv].float()
        return (
            torch.einsum("shd,hde,she->sh", value, self.output_grams[index], value)
            .clamp_min(0)
            .sqrt()
        )

    def _removed_write(
        self,
        index: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Project selected per-head context through this layer's own W_O."""

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
    ) -> torch.Tensor | None:
        if not any(removals):
            return None
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
        return {
            "role_attention_mass": torch.empty(*role_shape, dtype=torch.float16),
            "edge_role_energy": torch.empty(*role_shape, dtype=torch.float16),
            "head_role_write_norm": torch.empty(*role_shape, dtype=torch.float16),
            "head_source_entropy": torch.empty(*head_shape, dtype=torch.float16),
            "role_head_coherence": torch.empty(
                layers, response_tokens, len(ROLE_NAMES), dtype=torch.float16
            ),
            "top_source_index": torch.full(
                (layers, response_tokens, top_k), -1, dtype=torch.int32
            ),
            "top_source_magnitude": torch.zeros(
                layers, response_tokens, top_k, dtype=torch.float16
            ),
        }

    def _forward(
        self,
        token_ids: torch.Tensor,
        response_start: int,
        evidence_mask: torch.Tensor,
        *,
        removals: tuple[str | None, ...],
        capture: bool,
        predictor_chunk: int,
        top_k: int,
        logit_chunk: int,
    ) -> tuple[list[BranchScores], dict[str, torch.Tensor] | None]:
        ids = token_ids.to(self.device)
        source_tokens = len(ids) - 1
        response_tokens = len(ids) - response_start
        batch = len(removals)
        dtype = self.model.get_input_embeddings().weight.dtype
        intervene = any(removals)
        evidence_device = evidence_mask.to(self.device)
        k = min(top_k, source_tokens)
        trace = self._empty_trace(response_tokens, k) if capture else None
        values = (
            [
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
            if intervene or capture
            else None
        )
        source_norms = (
            [
                torch.empty(
                    source_tokens,
                    self.heads,
                    dtype=torch.float32,
                    device=self.device,
                )
                for _ in self.layers
            ]
            if capture
            else None
        )
        logprob = torch.empty(batch, response_tokens)
        margin = torch.empty(batch, response_tokens)
        query_start = query_stop = 0
        remove_mask = None
        handles = []

        def response_window() -> tuple[int, int, int] | None:
            local = max(response_start - 1 - query_start, 0)
            if query_start + local >= query_stop:
                return None
            row_start = query_start + local - (response_start - 1)
            return local, row_start, row_start + query_stop - query_start - local

        def v_hook(index: int):
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                current = output.reshape(
                    batch, query_stop - query_start, self.kv_heads, self.head_dim
                )
                if intervene or capture:
                    values[index][:, query_start:query_stop].copy_(current)
                if capture:
                    source_norms[index][query_start:query_stop] = self._source_norm(
                        index, current[0]
                    )

            return hook

        def attention_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any):
                parts = list(output)
                message, attention = parts[0], parts[1]
                window = response_window()

                if capture and window is not None:
                    local, row_start, row_stop = window
                    a = attention[0, :, local:, :query_stop].permute(1, 0, 2)
                    source_norm = source_norms[index][:query_stop]
                    edge_magnitude = a.float() * source_norm.T[None]
                    value_by_head = values[index][
                        0, :query_stop, self.q_to_kv, :
                    ]
                    evidence = torch.zeros(
                        query_stop, dtype=torch.bool, device=self.device
                    )
                    prompt_stop = min(response_start, query_stop)
                    evidence[:prompt_stop] = evidence_device[:prompt_stop]
                    source = torch.arange(query_stop, device=self.device)
                    query = torch.arange(
                        query_start + local, query_stop, device=self.device
                    )
                    self_source = source[None] == query[:, None]
                    evidence_role = evidence[None] & ~self_source
                    other_prompt = (
                        (source[None] < response_start)
                        & ~evidence[None]
                        & ~self_source
                    )
                    history = (
                        (source[None] >= response_start)
                        & (source[None] < query[:, None])
                    )
                    roles = (evidence_role, other_prompt, history, self_source)

                    role_attention = torch.stack(
                        [(a.float() * role[:, None]).sum(-1) for role in roles],
                        dim=-1,
                    )
                    edge_role = torch.stack(
                        [
                            (edge_magnitude * role[:, None]).sum(-1)
                            for role in roles
                        ],
                        dim=-1,
                    )
                    head_total = edge_magnitude.sum(-1).clamp_min(1e-12)
                    head_probability = edge_magnitude / head_total[..., None]
                    head_entropy = -(
                        head_probability
                        * head_probability.clamp_min(1e-12).log()
                    ).sum(-1)
                    visible_sources = (query + 1).clamp_min(2).float().log()
                    head_entropy = head_entropy / visible_sources[:, None]

                    contexts = []
                    for role in roles:
                        contexts.append(
                            torch.einsum(
                                "rhs,shd->rhd",
                                a.to(value_by_head.dtype) * role[:, None],
                                value_by_head,
                            )
                        )
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
                    coherence = net_write_norm / head_write_norm.sum(1).clamp_min(
                        1e-12
                    )

                    source_magnitude = edge_magnitude.sum(1)
                    current_k = min(k, query_stop)
                    visible = source[None] <= query[:, None]
                    top_magnitude, top_index = source_magnitude.masked_fill(
                        ~visible, -torch.inf
                    ).topk(current_k, dim=-1)
                    valid_rank = torch.arange(
                        current_k, device=self.device
                    )[None] < (query + 1).clamp_max(current_k)[:, None]
                    top_magnitude = top_magnitude.masked_fill(~valid_rank, 0)
                    top_index = top_index.masked_fill(~valid_rank, -1)

                    trace["edge_role_energy"][index, row_start:row_stop] = (
                        edge_role.detach().half().cpu()
                    )
                    trace["role_attention_mass"][index, row_start:row_stop] = (
                        role_attention.detach().half().cpu()
                    )
                    trace["head_role_write_norm"][index, row_start:row_stop] = (
                        head_write_norm.detach().half().cpu()
                    )
                    trace["head_source_entropy"][index, row_start:row_stop] = (
                        head_entropy.detach().half().cpu()
                    )
                    trace["role_head_coherence"][index, row_start:row_stop] = (
                        coherence.detach().half().cpu()
                    )
                    trace["top_source_index"][index, row_start:row_stop, :current_k] = (
                        top_index.int().cpu()
                    )
                    trace["top_source_magnitude"][
                        index, row_start:row_stop, :current_k
                    ] = top_magnitude.detach().cpu()

                if remove_mask is not None:
                    value_by_head = values[index][:, :query_stop, self.q_to_kv, :]
                    removed_context = torch.einsum(
                        "bhqs,bshd->bqhd",
                        attention * remove_mask[:, None],
                        value_by_head,
                    )
                    parts[0] = message - self._removed_write(index, removed_context)
                parts[1] = None
                return tuple(parts)

            return hook

        for index, layer in enumerate(self.layers):
            handles.append(layer.self_attn.v_proj.register_forward_hook(v_hook(index)))
            handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))

        try:
            past = None
            with torch.inference_mode():
                for query_start in range(0, source_tokens, predictor_chunk):
                    query_stop = min(query_start + predictor_chunk, source_tokens)
                    remove_mask = self._removal_mask(
                        removals,
                        evidence_mask,
                        response_start,
                        query_start,
                        query_stop,
                    )
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

        scores = [BranchScores(logprob[i], margin[i]) for i in range(batch)]
        if not capture:
            return scores, None
        return scores, trace

    def capture(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        response_start: int,
        evidence_mask: Sequence[bool] | torch.Tensor,
        *,
        predictor_chunk: int = 64,
        top_k: int = 8,
        logit_chunk: int = 64,
        intervention_batch: int = 3,
    ) -> dict[str, Any]:
        """Save mechanism state and symmetric direct evidence/history deletions."""

        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
        evidence_mask = torch.as_tensor(evidence_mask, dtype=torch.bool, device="cpu")
        if token_ids.ndim != 1 or not 0 < response_start < len(token_ids):
            raise ValueError("token_ids/response_start do not define a response")
        if evidence_mask.shape != (response_start,):
            raise ValueError("evidence_mask must align exactly with the prompt")
        if min(predictor_chunk, top_k, logit_chunk, intervention_batch) <= 0:
            raise ValueError("capture chunk sizes and top_k must be positive")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        full_scores, trace = self._forward(
            token_ids,
            response_start,
            evidence_mask,
            removals=(None,),
            capture=True,
            predictor_chunk=predictor_chunk,
            top_k=top_k,
            logit_chunk=logit_chunk,
        )
        removals = ("evidence", "history", "both")
        intervened = []
        for start in range(0, len(removals), intervention_batch):
            scores, _ = self._forward(
                token_ids,
                response_start,
                evidence_mask,
                removals=removals[start : start + intervention_batch],
                capture=False,
                predictor_chunk=predictor_chunk,
                top_k=top_k,
                logit_chunk=logit_chunk,
            )
            intervened.extend(scores)
        no_evidence, no_history, no_evidence_history = intervened
        peak = (
            int(torch.cuda.max_memory_reserved(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return {
            "token_ids": token_ids,
            "response_start": int(response_start),
            "trace": trace,
            "score_inputs": {
                "full_logprob": full_scores[0].logprob,
                "full_margin": full_scores[0].margin,
                "no_evidence_logprob": no_evidence.logprob,
                "no_evidence_margin": no_evidence.margin,
                "no_history_logprob": no_history.logprob,
                "no_history_margin": no_history.margin,
                "no_evidence_history_logprob": no_evidence_history.logprob,
                "no_evidence_history_margin": no_evidence_history.margin,
            },
            "peak_cuda_reserved_bytes": peak,
        }
