"""Chunked teacher-forced message traces from a frozen Llama observer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .data import EVIDENCE, ROLE_NAMES as PROMPT_ROLE_NAMES


HISTORY = len(PROMPT_ROLE_NAMES)
SELF = HISTORY + 1
ROLE_NAMES = (*PROMPT_ROLE_NAMES, "history", "self")


def predictor_positions(response_start: int, token_count: int) -> torch.Tensor:
    """Position ``q`` predicts response token ``token_ids[q + 1]``."""

    return torch.arange(response_start - 1, token_count - 1, dtype=torch.long)


def source_roles(
    prompt_roles: torch.Tensor,
    response_start: int,
    token_count: int,
) -> torch.Tensor:
    """Role of every causal source for every response-token predictor."""

    predictors = predictor_positions(response_start, token_count)
    source_count = token_count - 1
    roles = torch.full((len(predictors), source_count), -1, dtype=torch.int8)
    roles[:, :response_start] = prompt_roles.to(torch.int8)
    for row, query in enumerate(predictors.tolist()):
        if query >= response_start:
            roles[row, response_start:query] = HISTORY
        roles[row, query] = SELF
    return roles


@dataclass
class BranchScores:
    target_logit: torch.Tensor
    target_logprob: torch.Tensor
    target_margin: torch.Tensor
    top1_token_id: torch.Tensor


class FunctionalTraceReplay:
    """Save dynamic A/V messages and delete selected source writes causally."""

    def __init__(self, model: Any, *, checkpoint: str = "<in-memory>") -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.checkpoint = str(checkpoint)
        self.backbone = model.model
        self.layers = tuple(self.backbone.layers)
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(getattr(config, "num_key_value_heads", self.heads))
        self.hidden = int(config.hidden_size)
        self.head_dim = int(getattr(config, "head_dim", self.hidden // self.heads))
        repeats = self.heads // self.kv_heads
        self.q_to_kv = torch.arange(self.heads) // repeats
        self.output_grams = tuple(self._output_gram(layer) for layer in self.layers)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FunctionalTraceReplay":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        return cls(model, checkpoint=str(Path(checkpoint).resolve()))

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _output_gram(self, layer: Any) -> torch.Tensor:
        weight = layer.self_attn.o_proj.weight.detach().float()
        blocks = weight.reshape(self.hidden, self.heads, self.head_dim).permute(1, 2, 0)
        return blocks @ blocks.transpose(1, 2)

    def _target_scores(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        chunk: int,
    ) -> BranchScores:
        target_logits, target_logprobs, margins, top1 = [], [], [], []
        for start in range(0, len(hidden), chunk):
            stop = min(start + chunk, len(hidden))
            logits = self.model.lm_head(hidden[start:stop]).float()
            targets = target_ids[start:stop]
            selected = logits.gather(1, targets[:, None]).squeeze(1)
            competitor = logits.scatter(1, targets[:, None], -torch.inf).max(1).values
            target_logits.append(selected.cpu())
            target_logprobs.append((selected - logits.logsumexp(1)).cpu())
            margins.append((selected - competitor).cpu())
            top1.append(logits.argmax(1).cpu())
        return BranchScores(
            torch.cat(target_logits),
            torch.cat(target_logprobs),
            torch.cat(margins),
            torch.cat(top1),
        )

    @staticmethod
    def _tensor(output: Any) -> torch.Tensor:
        return output if torch.is_tensor(output) else output[0]

    def _removal_mask(
        self,
        removals: tuple[str | None, ...],
        prompt_roles: torch.Tensor,
        response_start: int,
        query_start: int,
        query_stop: int,
    ) -> torch.Tensor | None:
        if removals[0] is None:
            return None
        source = torch.arange(query_stop, device=self.device)
        query = torch.arange(query_start, query_stop, device=self.device)
        causal = source[None] <= query[:, None]
        evidence = torch.zeros(query_stop, dtype=torch.bool, device=self.device)
        prompt_stop = min(response_start, query_stop)
        evidence[:prompt_stop] = prompt_roles[:prompt_stop].to(self.device) == EVIDENCE
        response = source >= response_start
        masks = []
        for removal in removals:
            selected = torch.zeros_like(causal)
            if removal in {"evidence", "both"}:
                selected |= causal & evidence[None]
            if removal in {"response", "both"}:
                selected |= causal & response[None]
            masks.append(selected)
        return torch.stack(masks)

    def _empty_trace(
        self,
        response_tokens: int,
        source_tokens: int,
        top_k: int,
        dtype: torch.dtype,
        *,
        retain_raw: bool,
    ) -> dict[str, torch.Tensor]:
        layers = len(self.layers)
        roles = len(ROLE_NAMES)
        trace = {
            "role_attention": torch.empty(
                layers, response_tokens, self.heads, roles, dtype=torch.float32
            ),
            "role_edge_magnitude": torch.empty(
                layers, response_tokens, self.heads, roles, dtype=torch.float32
            ),
            "source_message_entropy": torch.empty(
                layers, response_tokens, dtype=torch.float32
            ),
            "message_coherence": torch.empty(
                layers, response_tokens, dtype=torch.float32
            ),
            "top_source_index": torch.full(
                (layers, response_tokens, top_k), -1, dtype=torch.int32
            ),
            "top_source_magnitude": torch.zeros(
                layers, response_tokens, top_k, dtype=torch.float32
            ),
        }
        if retain_raw:
            trace.update(
                attention=torch.zeros(
                    layers, self.heads, response_tokens, source_tokens, dtype=dtype
                ),
                o_proj_input=torch.empty(
                    layers, response_tokens, self.hidden, dtype=dtype
                ),
                residual_input=torch.empty(
                    layers, response_tokens, self.hidden, dtype=dtype
                ),
                attention_update=torch.empty(
                    layers, response_tokens, self.hidden, dtype=dtype
                ),
                mlp_update=torch.empty(
                    layers, response_tokens, self.hidden, dtype=dtype
                ),
                final_hidden=torch.empty(response_tokens, self.hidden, dtype=dtype),
            )
        return trace

    def _forward(
        self,
        token_ids: torch.Tensor,
        response_start: int,
        prompt_roles: torch.Tensor,
        *,
        removals: tuple[str | None, ...],
        capture: bool,
        retain_raw: bool,
        predictor_chunk: int,
        top_k: int,
        logit_chunk: int,
    ) -> tuple[list[BranchScores], dict[str, torch.Tensor] | None]:
        ids = token_ids.to(self.device)
        source_tokens = len(ids) - 1
        response_tokens = len(ids) - response_start
        batch = len(removals)
        dtype = self.model.get_input_embeddings().weight.dtype
        roles = source_roles(prompt_roles, response_start, len(ids))
        roles_device = roles.to(self.device)
        k = min(top_k, source_tokens)
        trace = (
            self._empty_trace(
                response_tokens, source_tokens, k, dtype, retain_raw=retain_raw
            )
            if capture
            else None
        )
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
        score_fields = {
            name: torch.empty(batch, response_tokens, dtype=field_dtype)
            for name, field_dtype in (
                ("target_logit", torch.float32),
                ("target_logprob", torch.float32),
                ("target_margin", torch.float32),
                ("top1_token_id", torch.long),
            )
        }
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
                values[index][:, query_start:query_stop].copy_(current)
                if capture:
                    current_by_head = current[
                        0, :, self.q_to_kv.to(self.device), :
                    ].float()
                    source_norms[index][query_start:query_stop] = torch.einsum(
                        "shd,hde,she->sh",
                        current_by_head,
                        self.output_grams[index],
                        current_by_head,
                    ).clamp_min(0).sqrt()

            return hook

        def layer_input_hook(index: int):
            def hook(_module: Any, args: tuple[Any, ...]) -> None:
                window = response_window()
                if capture and retain_raw and window is not None:
                    local, row_start, row_stop = window
                    trace["residual_input"][index, row_start:row_stop] = (
                        args[0][0, local:].detach().cpu()
                    )

            return hook

        def o_proj_input_hook(index: int):
            def hook(_module: Any, args: tuple[Any, ...]) -> None:
                window = response_window()
                if capture and retain_raw and window is not None:
                    local, row_start, row_stop = window
                    trace["o_proj_input"][index, row_start:row_stop] = (
                        args[0][0, local:].detach().cpu()
                    )

            return hook

        def mlp_hook(index: int):
            def hook(_module: Any, _args: Any, output: Any) -> None:
                window = response_window()
                if capture and retain_raw and window is not None:
                    local, row_start, row_stop = window
                    trace["mlp_update"][index, row_start:row_stop] = (
                        self._tensor(output)[0, local:].detach().cpu()
                    )

            return hook

        def attention_hook(index: int):
            layer = self.layers[index]

            def hook(_module: Any, _args: Any, output: Any):
                parts = list(output)
                message, attention = parts[0], parts[1]
                value_by_head = values[index][
                    :, :query_stop, self.q_to_kv.to(self.device), :
                ]
                window = response_window()

                if capture and window is not None:
                    local, row_start, row_stop = window
                    a = attention[0, :, local:, :query_stop].permute(1, 0, 2)
                    current_roles = roles_device[row_start:row_stop, :query_stop]
                    source_norm = source_norms[index][:query_stop]
                    edge_magnitude = a.float() * source_norm.T[None]
                    role_attention, role_edge = [], []
                    for role in range(len(ROLE_NAMES)):
                        mask = (current_roles == role)[:, None, :]
                        role_attention.append((a.float() * mask).sum(-1))
                        role_edge.append((edge_magnitude * mask).sum(-1))
                    source_magnitude = edge_magnitude.sum(1)
                    probability = source_magnitude / source_magnitude.sum(
                        -1, keepdim=True
                    ).clamp_min(1e-12)
                    entropy = -(
                        probability * probability.clamp_min(1e-12).log()
                    ).sum(-1)
                    total_write = message[0, local:].float()
                    if layer.self_attn.o_proj.bias is not None:
                        total_write = total_write - layer.self_attn.o_proj.bias.float()
                    current_k = min(k, query_stop)
                    top_magnitude, top_index = source_magnitude.topk(
                        current_k, dim=-1
                    )

                    if retain_raw:
                        trace["attention"][
                            index, :, row_start:row_stop, :query_stop
                        ] = attention[0, :, local:, :query_stop].detach().cpu()
                        trace["attention_update"][index, row_start:row_stop] = (
                            message[0, local:].detach().cpu()
                        )
                    trace["role_attention"][index, row_start:row_stop] = (
                        torch.stack(role_attention, -1).detach().cpu()
                    )
                    trace["role_edge_magnitude"][index, row_start:row_stop] = (
                        torch.stack(role_edge, -1).detach().cpu()
                    )
                    trace["source_message_entropy"][index, row_start:row_stop] = (
                        entropy.detach().cpu()
                    )
                    trace["message_coherence"][index, row_start:row_stop] = (
                        (
                            total_write.norm(dim=-1)
                            / source_magnitude.sum(-1).clamp_min(1e-12)
                        )
                        .detach()
                        .cpu()
                    )
                    trace["top_source_index"][
                        index, row_start:row_stop, :current_k
                    ] = (
                        top_index.int().cpu()
                    )
                    trace["top_source_magnitude"][
                        index, row_start:row_stop, :current_k
                    ] = (
                        top_magnitude.detach().cpu()
                    )

                if remove_mask is not None:
                    removed_context = torch.einsum(
                        "bhqs,bshd->bqhd",
                        attention * remove_mask[:, None],
                        value_by_head,
                    )
                    removed_write = F.linear(
                        removed_context.reshape(
                            batch, query_stop - query_start, self.hidden
                        ),
                        layer.self_attn.o_proj.weight,
                        bias=None,
                    )
                    parts[0] = message - removed_write
                parts[1] = None
                return tuple(parts)

            return hook

        for index, layer in enumerate(self.layers):
            handles.append(layer.self_attn.v_proj.register_forward_hook(v_hook(index)))
            handles.append(layer.self_attn.register_forward_hook(attention_hook(index)))
            if retain_raw:
                handles.append(layer.register_forward_pre_hook(layer_input_hook(index)))
                handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        o_proj_input_hook(index)
                    )
                )
                handles.append(layer.mlp.register_forward_hook(mlp_hook(index)))

        try:
            past = None
            with torch.inference_mode():
                for query_start in range(0, source_tokens, predictor_chunk):
                    query_stop = min(query_start + predictor_chunk, source_tokens)
                    remove_mask = self._removal_mask(
                        removals,
                        prompt_roles,
                        response_start,
                        query_start,
                        query_stop,
                    )
                    output = self.backbone(
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
                        current = self._target_scores(
                            hidden[branch], targets, chunk=logit_chunk
                        )
                        for name in score_fields:
                            score_fields[name][branch, row_start:row_stop] = getattr(
                                current, name
                            )
                    if capture and retain_raw:
                        trace["final_hidden"][row_start:row_stop] = (
                            hidden[0].detach().cpu()
                        )
        finally:
            for handle in handles:
                handle.remove()

        scores = [
            BranchScores(**{name: value[branch] for name, value in score_fields.items()})
            for branch in range(batch)
        ]
        if not capture:
            return scores, None
        if retain_raw:
            trace["value_states"] = torch.stack([value[0].cpu() for value in values])
        trace["source_role"] = roles
        return scores, trace

    def capture(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        response_start: int,
        prompt_roles: Sequence[int] | torch.Tensor,
        *,
        predictor_chunk: int = 64,
        top_k: int = 8,
        logit_chunk: int = 64,
        intervention_batch: int = 3,
        retain_raw: bool = True,
    ) -> dict[str, Any]:
        """Save one mechanism trace and three same-sample message-deletion branches."""

        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
        prompt_roles = torch.as_tensor(prompt_roles, dtype=torch.int8, device="cpu")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        full_scores, trace = self._forward(
            token_ids,
            response_start,
            prompt_roles,
            removals=(None,),
            capture=True,
            retain_raw=retain_raw,
            predictor_chunk=predictor_chunk,
            top_k=top_k,
            logit_chunk=logit_chunk,
        )
        branches = {"full": full_scores[0]}
        branch_specs = (
            ("evidence_removed", "evidence"),
            ("response_removed", "response"),
            ("evidence_response_removed", "both"),
        )
        for start in range(0, len(branch_specs), intervention_batch):
            current_specs = branch_specs[start : start + intervention_batch]
            current_scores, _ = self._forward(
                token_ids,
                response_start,
                prompt_roles,
                removals=tuple(removal for _name, removal in current_specs),
                capture=False,
                retain_raw=False,
                predictor_chunk=predictor_chunk,
                top_k=top_k,
                logit_chunk=logit_chunk,
            )
            branches.update(
                (name, score)
                for (name, _removal), score in zip(current_specs, current_scores)
            )
        peak = (
            int(torch.cuda.max_memory_reserved(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return {
            "token_ids": token_ids,
            "response_start": int(response_start),
            "predictor_positions": predictor_positions(response_start, len(token_ids)),
            "target_ids": token_ids[response_start:],
            "prompt_role": prompt_roles,
            "role_names": ROLE_NAMES,
            "trace": trace,
            "scores": {
                name: {
                    field: getattr(score, field)
                    for field in BranchScores.__dataclass_fields__
                }
                for name, score in branches.items()
            },
            "peak_cuda_reserved_bytes": peak,
        }


__all__ = [
    "FunctionalTraceReplay",
    "HISTORY",
    "ROLE_NAMES",
    "SELF",
    "predictor_positions",
    "source_roles",
]
