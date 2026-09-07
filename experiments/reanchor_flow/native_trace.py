"""Stream native vector writes and signed readout accounting, without labels.

The final observed-token/runner direction is frozen by a first forward.  A second
forward records every head and module in one common residual-space projection.
Signed margins are direct-logit accounting, not causal or factual attribution.
Only display edges are truncated; all sources participate in each head output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.common.llama_message_intervention import (
    READOUT_CHUNK,
    VALIDATED_ATTRIBUTE,
    forward_layers,
    validate_manual_forward,
)

from .message_norm import HEAD_CHUNK, model_gram_cache, output_gram


def residual_projection(hidden_size: int, sketch_dim: int, seed: int) -> np.ndarray:
    """One deterministic orthonormal basis, shared across layers and samples."""

    if not 1 <= sketch_dim <= hidden_size:
        raise ValueError("sketch_dim must be between 1 and the residual width")
    matrix = np.random.default_rng(seed).standard_normal((hidden_size, sketch_dim))
    basis, triangular = np.linalg.qr(matrix, mode="reduced")
    basis *= np.where(np.diag(triangular) < 0, -1, 1)
    return basis.astype(np.float32)


def _numpy(value: Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


@dataclass(frozen=True)
class FrozenReadout:
    direction: Tensor
    arrays: dict[str, np.ndarray]

    @classmethod
    def capture(cls, model, raw_hidden: Tensor, ids: Tensor, rows: Tensor):
        raw = raw_hidden[0].index_select(0, rows).float()
        normalized = model.model.norm(raw_hidden[0].index_select(0, rows))
        observed = ids[rows + 1]
        runner = torch.empty_like(observed)
        margin = torch.empty(len(rows), device=raw.device)
        logprob = torch.empty_like(margin)
        entropy = torch.empty_like(margin)
        for begin in range(0, len(rows), READOUT_CHUNK):
            end = min(begin + READOUT_CHUNK, len(rows))
            logits = F.linear(
                normalized[begin:end], model.lm_head.weight, model.lm_head.bias
            ).float()
            target_logit = logits.gather(1, observed[begin:end, None]).flatten()
            normalizer = logits.logsumexp(-1)
            logprob[begin:end] = target_logit - normalizer
            entropy[begin:end] = normalizer - (logits.softmax(-1) * logits).sum(-1)
            logits.scatter_(1, observed[begin:end, None], -torch.inf)
            runner[begin:end] = logits.argmax(-1)
            margin[begin:end] = target_logit - logits.gather(
                1, runner[begin:end, None]
            ).flatten()

        norm = model.model.norm
        denominator = (raw.square().mean(-1) + norm.variance_epsilon).sqrt()
        direction = model.lm_head.weight[observed].float()
        direction -= model.lm_head.weight[runner].float()
        direction *= norm.weight.float()[None] / denominator[:, None]
        bias = torch.zeros_like(margin)
        if model.lm_head.bias is not None:
            bias = model.lm_head.bias[observed].float() - model.lm_head.bias[runner].float()
        return cls(
            direction,
            {
                "runner_token_ids": runner.cpu().numpy(),
                "observed_token_ids": observed.cpu().numpy(),
                "final_margin": _numpy(margin),
                "observed_logprob": _numpy(logprob),
                "entropy": _numpy(entropy),
                "readout_bias": _numpy(bias),
                "final_rms_denominator": _numpy(denominator),
            },
        )


class NativeTraceObserver:
    """Current-layer observer; no full source-edge vectors or autograd graph."""

    def __init__(self, model, rows, direction, projection, display_edges, save_head_codes):
        self.model = model
        self.rows = rows
        self.start = int(rows[0])
        self.direction = direction
        self.projection = projection
        self.display_edges = display_edges
        layers = len(model.model.layers)
        attention = model.model.layers[0].self_attn
        self.head_dim = attention.head_dim
        self.heads = attention.q_proj.out_features // self.head_dim
        count, rank = len(rows), projection.shape[1]
        lhq = (layers, self.heads, count)
        lq = (layers, count)
        self.arrays = {
            "head_sketch": np.empty((*lhq, rank), np.float32),
            "head_margin": np.empty(lhq, np.float32),
            "head_norm": np.empty(lhq, np.float32),
            "edge_sum_margin": np.empty(lhq, np.float32),
            "edge_total_absolute_margin": np.empty(lhq, np.float32),
            "edge_source_position": np.full((*lhq, display_edges), -1, np.int32),
            "edge_attention": np.zeros((*lhq, display_edges), np.float32),
            "edge_margin": np.zeros((*lhq, display_edges), np.float32),
            "residual_sketch": np.empty((layers + 1, count, rank), np.float32),
            "residual_norm": np.empty((layers + 1, count), np.float32),
            "stage_margin": np.empty((layers + 1, count), np.float32),
        }
        for name in ("attention", "post_attention", "mlp"):
            self.arrays[f"{name}_sketch"] = np.empty((*lq, rank), np.float32)
            self.arrays[f"{name}_margin"] = np.empty(lq, np.float32)
            self.arrays[f"{name}_norm"] = np.empty(lq, np.float32)
        if save_head_codes:
            self.arrays["head_code"] = np.empty((*lhq, self.head_dim), np.float32)
        self.gram_cache = model_gram_cache(model)

    def _record_stage(self, name: str, layer: int, value: Tensor):
        value = value.float()
        self.arrays[f"{name}_sketch"][layer] = _numpy(value @ self.projection)
        self.arrays[f"{name}_norm"][layer] = _numpy(value.norm(dim=-1))
        margin_name = "stage_margin" if name == "residual" else f"{name}_margin"
        self.arrays[margin_name][layer] = _numpy((value * self.direction).sum(-1))

    def observe_layer_input(self, layer: int, hidden: Tensor):
        self.input_rows = hidden[0].index_select(0, self.rows)
        self._record_stage("residual", layer, self.input_rows)
        weight = self.model.model.layers[layer].self_attn.o_proj.weight
        blocks = weight.view(weight.shape[0], self.heads, self.head_dim).permute(1, 2, 0)
        self.projected_output = torch.empty(
            (self.heads, self.head_dim, self.projection.shape[1]),
            device=weight.device, dtype=torch.float32,
        )
        self.readout_output = torch.empty(
            (self.heads, len(self.rows), self.head_dim), device=weight.device,
            dtype=torch.float32,
        )
        for begin in range(0, self.heads, HEAD_CHUNK):
            end = min(begin + HEAD_CHUNK, self.heads)
            block = blocks[begin:end].float()
            self.projected_output[begin:end] = block @ self.projection
            self.readout_output[begin:end] = torch.einsum("hdk,qk->hqd", block, self.direction)
        if layer not in self.gram_cache:
            self.gram_cache[layer] = output_gram(weight, self.heads, self.head_dim)
        self.gram = self.gram_cache[layer].to(weight.device)

    def _query_slice(self, begin: int, count: int):
        first, stop = max(begin, self.start), min(begin + count, int(self.rows[-1]) + 1)
        return first, stop, slice(first - self.start, stop - self.start)

    def observe_chunk(self, layer, begin, probability, value, output_weight):
        first, stop, saved = self._query_slice(begin, probability.shape[2])
        if stop <= first:
            return
        selected = probability[0, :, first - begin:stop - begin]
        sources = value.shape[2]
        count = min(self.display_edges, sources)
        edge_sum = torch.empty((self.heads, stop - first), device=value.device)
        edge_absolute = torch.empty_like(edge_sum)
        kept_margin = torch.empty((*edge_sum.shape, count), device=value.device)
        kept_attention = torch.empty_like(kept_margin)
        kept_source = torch.empty_like(kept_margin, dtype=torch.long)
        causal = torch.arange(sources, device=value.device)[None] <= torch.arange(
            first, stop, device=value.device
        )[:, None]
        for head_begin in range(0, self.heads, HEAD_CHUNK):
            head_end = min(head_begin + HEAD_CHUNK, self.heads)
            heads = slice(head_begin, head_end)
            slope = self.readout_output[heads, saved]
            edge = torch.einsum("hsd,hqd->hqs", value[0, heads].float(), slope)
            edge *= selected[heads].float()
            edge_sum[heads] = edge.sum(-1)
            edge_absolute[heads] = edge.abs().sum(-1)
            if count:
                scores, indices = edge.abs().masked_fill(~causal[None], -torch.inf).topk(
                    count, dim=-1
                )
                valid = scores.isfinite()
                margin = edge.gather(-1, indices).masked_fill(~valid, 0)
                attention = selected[heads].gather(-1, indices).masked_fill(~valid, 0)
                indices.masked_fill_(~valid, -1)
                kept_source[heads] = indices
                kept_margin[heads] = margin
                kept_attention[heads] = attention
        self.arrays["edge_sum_margin"][layer, :, saved] = _numpy(edge_sum)
        self.arrays["edge_total_absolute_margin"][layer, :, saved] = _numpy(edge_absolute)
        self.arrays["edge_source_position"][layer, :, saved, :count] = kept_source.cpu().numpy()
        self.arrays["edge_margin"][layer, :, saved, :count] = _numpy(kept_margin)
        self.arrays["edge_attention"][layer, :, saved, :count] = _numpy(kept_attention)

    def observe_head_output(self, layer, begin, head_output):
        first, stop, saved = self._query_slice(begin, head_output.shape[2])
        if stop <= first:
            return
        code = head_output[0, :, first - begin:stop - begin].float()
        self.arrays["head_sketch"][layer, :, saved] = _numpy(code @ self.projected_output)
        margin = torch.empty(code.shape[:2], device=code.device)
        norm = torch.empty_like(margin)
        for head_begin in range(0, self.heads, HEAD_CHUNK):
            head_end = min(head_begin + HEAD_CHUNK, self.heads)
            heads = slice(head_begin, head_end)
            slope = self.readout_output[heads, saved]
            margin[heads] = (code[heads] * slope).sum(-1)
            squared = torch.einsum("hqd,hde,hqe->hq", code[heads], self.gram[heads], code[heads])
            norm[heads] = squared.clamp_min(0).sqrt()
        self.arrays["head_margin"][layer, :, saved] = _numpy(margin)
        self.arrays["head_norm"][layer, :, saved] = _numpy(norm)
        if "head_code" in self.arrays:
            self.arrays["head_code"][layer, :, saved] = _numpy(code)

    def observe_attention_write(self, layer: int, value: Tensor):
        selected = value[0].index_select(0, self.rows)
        self._record_stage("attention", layer, selected)
        # Match the native dtype addition before projecting; retain its remainder.
        self.post_attention = self.input_rows + selected
        self._record_stage("post_attention", layer, self.post_attention)

    def observe_mlp_write(self, layer: int, value: Tensor):
        selected = value[0].index_select(0, self.rows)
        self._record_stage("mlp", layer, selected)
        self._record_stage("residual", layer + 1, self.post_attention + selected)

    def finish(self, readout: FrozenReadout) -> dict[str, np.ndarray]:
        arrays = self.arrays
        arrays.update(readout.arrays)
        arrays["edge_rounding_remainder"] = arrays["head_margin"] - arrays["edge_sum_margin"]
        arrays["omitted_margin"] = arrays["head_margin"] - arrays["edge_margin"].sum(-1)
        arrays["edge_omitted_absolute_margin"] = (
            arrays["edge_total_absolute_margin"] - np.abs(arrays["edge_margin"]).sum(-1)
        ).clip(min=0)
        for suffix in ("sketch", "margin"):
            residual = arrays["residual_sketch" if suffix == "sketch" else "stage_margin"]
            attention = arrays[f"attention_{suffix}"]
            post = arrays[f"post_attention_{suffix}"]
            arrays[f"head_remainder_{suffix}"] = attention - arrays[f"head_{suffix}"].sum(1)
            arrays[f"attention_add_remainder_{suffix}"] = post - residual[:-1] - attention
            arrays[f"mlp_add_remainder_{suffix}"] = residual[1:] - post - arrays[f"mlp_{suffix}"]
        arrays["readout_remainder"] = (
            arrays["final_margin"] - arrays["stage_margin"][-1] - arrays["readout_bias"]
        )
        return arrays


@torch.inference_mode()
def capture_native_trace(
    model,
    token_ids,
    response_start: int,
    *,
    sketch_dim: int = 16,
    seed: int = 2026,
    query_chunk: int = 8,
    display_edges: int = 2,
    save_head_codes: bool = False,
) -> dict[str, np.ndarray]:
    """Capture all response predictors with two forwards and bounded edge chunks.

    RMSNorm denominators and non-observed runners are frozen in the first pass.
    Sketches are lossy common-coordinate observations; margins/norms are computed
    from full vectors.  Positive margin supports the observed token over its
    runner, which need not correspond to factual support.
    """

    device = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).flatten()
    if not 1 <= response_start < len(ids):
        raise ValueError("response_start must leave prompt and response tokens")
    if query_chunk < 1 or display_edges < 0:
        raise ValueError("query_chunk must be positive and display_edges nonnegative")
    model.eval()
    if not getattr(model, VALIDATED_ATTRIBUTE, False):
        validate_manual_forward(model, ids)
    rows = torch.arange(response_start - 1, len(ids) - 1, device=device)
    hidden = model.model.embed_tokens(ids[:-1][None])
    raw_final = forward_layers(
        model, hidden, 0, attention_query_chunk=query_chunk, apply_final_norm=False
    )
    readout = FrozenReadout.capture(model, raw_final, ids, rows)
    first_raw_rows = raw_final[0].index_select(0, rows).float()
    del raw_final
    basis = residual_projection(hidden.shape[-1], sketch_dim, seed)
    observer = NativeTraceObserver(
        model, rows, readout.direction, torch.from_numpy(basis).to(device),
        display_edges, save_head_codes,
    )
    final = forward_layers(
        model, hidden, 0, observer=observer,
        attention_query_chunk=query_chunk, apply_final_norm=False,
    )
    observer._record_stage("residual", len(model.model.layers), final[0].index_select(0, rows))
    arrays = observer.finish(readout)
    arrays["forward_repeat_max_abs_error"] = _numpy(
        (final[0].index_select(0, rows).float() - first_raw_rows).abs().amax(-1)
    )
    arrays.update({
        "native_trace_schema": np.asarray(1),
        "token_ids": ids.cpu().numpy(),
        "response_start": np.asarray(response_start),
        "row_position": rows.cpu().numpy(),
        "sketch_seed": np.asarray(seed),
        "sketch_dim": np.asarray(sketch_dim),
        "sketch_projection": basis,
        "query_chunk": np.asarray(query_chunk),
        "num_heads": np.asarray(observer.heads),
        "num_kv_heads": np.asarray(
            model.model.layers[0].self_attn.k_proj.out_features // observer.head_dim
        ),
        "labels_used_for_capture": np.asarray(False),
        "head_averaging": np.asarray(False),
        "display_edges_only": np.asarray(True),
        "margin_semantics": np.asarray("frozen_observed_minus_runner_direct_logit_accounting"),
        "omitted_margin_includes_head_rounding": np.asarray(True),
        "head_remainder_includes_output_bias": np.asarray(True),
    })
    return arrays
