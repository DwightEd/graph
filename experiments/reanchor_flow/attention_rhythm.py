"""Head-resolved attention phenomenology; no classifier, PCA or causal claims.

Paper: Attention Illuminates LLM Reasoning, arXiv:2510.13554v2, Eqs. 7,9,10.
Rows include P-1 (first response predictor) AND P..N-1 (paper response rows).
The same full source rows produce distance, clipped distance, FAI and buckets.
Only a bounded, explicitly indexed selection of raw maps is retained for plots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RhythmConfig:
    window: int = 10
    future_lo: int = 10
    future_hi: int = 100
    local_window: int = 10
    head_fraction: float = 0.30

    def __post_init__(self):
        if not (self.window > 0 and self.local_window > 0
                and 0 <= self.future_lo <= self.future_hi and self.future_hi > 0
                and 0 < self.head_fraction <= 0.5):
            raise ValueError("invalid window, FAI horizon or head fraction")


def head_groups(distance_mean: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Paper span ranking only; not semantic head roles or a trained selector."""
    order = np.argsort(np.asarray(distance_mean).ravel(), kind="stable")
    count = min(max(1, int(len(order) * fraction)), len(order) // 2)
    return order[:count], order[-count:] if count else order[:0]


class AttentionRhythmObserver:
    """Stream native attention into [layer,head,position] measurements.

    observe_rows is model-independent and testable with exact synthetic maps.
    observe_chunk adapts the repository's shared Llama forward observer.
    One current-layer Gram matrix/value norm is retained; never [L,H,N,N].
    """

    def __init__(self, layers: int, heads: int, tokens: int, response_start: int,
                 evidence_mask: np.ndarray, config: RhythmConfig):
        if not 1 <= response_start < tokens:
            raise ValueError("response_start must leave a nonempty response")
        evidence = np.asarray(evidence_mask, dtype=bool)
        if evidence.shape != (tokens,) or evidence[response_start:].any():
            raise ValueError("evidence_mask must identify prompt positions only")
        self.config, self.start, self.tokens = config, response_start, tokens
        self.rows = np.arange(response_start - 1, tokens)
        self.evidence = evidence
        shape = (layers, heads, len(self.rows))
        self.values = {name: np.zeros(shape, np.float32) for name in
                       ("distance", "waad", "message_distance", "message_waad")}
        self.buckets = {name: np.zeros((*shape, 4), np.float32)
                        for name in ("attention_buckets", "message_buckets")}
        self.winner = np.zeros(shape, np.int32)
        self.fai_sum = np.zeros(shape, np.float64)
        self.fai_message_sum = np.zeros(shape, np.float64)
        self.fai_message_count = np.zeros(shape, np.int32)
        self.seen = np.zeros((layers, len(self.rows)), bool)
        self.mass_error, self.future_leak = 0.0, 0.0
        self._layer, self._value_norm = -1, None

    def observe_chunk(self, layer, begin, probability, value, output_weight):
        if begin + probability.shape[2] <= self.start - 1:
            return
        if self._layer != layer:
            heads, dim = value.shape[1], value.shape[-1]
            blocks = output_weight.detach().float().reshape(-1, heads, dim).permute(1, 0, 2)
            gram = blocks.transpose(-1, -2) @ blocks
            v = value[0].detach().float()
            self._value_norm = torch.einsum("hsd,hde,hse->hs", v, gram, v).clamp_min(0).sqrt()
            self._layer = layer
        self.observe_rows(layer, begin, probability[0], self._value_norm)

    @torch.no_grad()
    def observe_rows(self, layer: int, begin: int, probability: torch.Tensor,
                     value_norm: torch.Tensor):
        """probability [H,chunk,N]; value_norm [H,N] is ||W_O[h] V_s||."""
        first = max(begin, self.start - 1)
        stop = min(begin + probability.shape[1], self.tokens)
        if stop <= first:
            return
        saved = slice(first - self.start + 1, stop - self.start + 1)
        if self.seen[layer, saved].any():
            raise ValueError("duplicate attention rows would double-count FAI")
        self.seen[layer, saved] = True
        a = probability[:, first - begin:stop - begin].detach().float()
        if a.shape[-1] != self.tokens or value_norm.shape != (a.shape[0], self.tokens):
            raise ValueError("full causal source row required, not a pruned cache")
        device = a.device
        q = torch.arange(first, stop, device=device)
        source = torch.arange(self.tokens, device=device)
        lag = q[:, None] - source[None]
        future = lag < 0
        self.mass_error = max(self.mass_error, float((a.sum(-1) - 1).abs().max()))
        self.future_leak = max(self.future_leak, float(a.masked_fill(~future[None], 0).abs().max()))
        distance = lag.clamp_min(0).float()
        clipped = distance.clamp_max(self.config.window)
        transport = a * value_norm.to(device)[:, None]
        total = transport.sum(-1, keepdim=True)
        weighted = torch.where(total > 0, transport / total.clamp_min(1e-30), 0)
        for prefix, weights in (("", a), ("message_", weighted)):
            for name, metric in (("distance", distance), ("waad", clipped)):
                value = (weights * metric).sum(-1)
                if prefix:
                    value = value.masked_fill(total.squeeze(-1) == 0, torch.nan)
                self.values[prefix + name][layer, :, saved] = value.cpu().numpy()
        self.winner[layer, :, saved] = a.argmax(-1).cpu().numpy()
        prompt = source < self.start
        evidence = torch.as_tensor(self.evidence, device=device)
        masks = torch.stack((
            (lag >= 0) & evidence,
            (lag >= 0) & prompt & ~evidence,
            (lag > self.config.local_window) & ~prompt,
            (lag >= 0) & (lag <= self.config.local_window) & ~prompt,
        )).float()
        for name, weights in (("attention_buckets", a), ("message_buckets", transport)):
            self.buckets[name][layer, :, saved] = torch.einsum(
                "hqs,bqs->hqb", weights, masks).cpu().numpy()
        # Eq.10 is inclusive at BOTH horizons. Query rows must be response rows.
        eligible = ((q[:, None] >= self.start)
                    & (lag >= self.config.future_lo) & (lag <= self.config.future_hi))
        for storage, weights in ((self.fai_sum, a), (self.fai_message_sum, weighted)):
            storage[layer] += (weights * eligible[None]).sum(1)[:, self.start - 1:].cpu().numpy()
        message_eligible = eligible[None] & (total > 0)
        self.fai_message_count[layer] += message_eligible.sum(1)[:, self.start - 1:].cpu().numpy()

    def finish(self) -> dict[str, np.ndarray]:
        if not self.seen.all():
            raise ValueError("incomplete native row capture")
        if self.future_leak > 1e-7 or self.mass_error > 0.02:
            raise ValueError("attention is not causal/row-normalized")
        left = np.maximum(self.rows + self.config.future_lo, self.start)
        right = np.minimum(self.rows + self.config.future_hi, self.tokens - 1)
        count = np.maximum(right - left + 1, 0)
        def average(value):
            return np.divide(value, count[None, None],
                             out=np.full(value.shape, np.nan), where=count[None, None] > 0).astype(np.float32)
        message_fai = np.divide(self.fai_message_sum, self.fai_message_count,
                                out=np.full(self.fai_sum.shape, np.nan),
                                where=self.fai_message_count > 0).astype(np.float32)
        # P-1 is not included in the paper's head-role statistic.
        distance_mean = self.values["distance"][..., 1:].mean(-1)
        local, global_ = head_groups(distance_mean, self.config.head_fraction)
        return {
            **self.values, **self.buckets,
            "row_position": self.rows, "response_start": np.array(self.start),
            "fai_source_position": self.rows.copy(), "fai_count": count,
            "fai": average(self.fai_sum), "message_fai": message_fai,
            "message_fai_count": self.fai_message_count,
            "fai_full_horizon": count == self.config.future_hi - self.config.future_lo + 1,
            "distance_mean": distance_mean, "local_heads": local, "global_heads": global_,
            "winner_position": self.winner,
            "max_row_mass_error": np.array(self.mass_error),
            "max_future_attention": np.array(self.future_leak),
        }


class RawMapObserver:
    """Exact source-position maps for explicit heads and a bounded query window.

    Paper group means are OPTIONAL reference panels only, never primary data.
    """

    def __init__(self, selected: np.ndarray, heads: int, rows: np.ndarray, tokens: int,
                 groups: tuple[np.ndarray, np.ndarray] | None = None):
        self.selected, self.heads, self.rows = np.asarray(selected), heads, np.asarray(rows)
        self.maps = np.zeros((len(selected), len(rows), tokens), np.float32)
        self.groups = groups
        self.group_maps = np.zeros((2, len(rows), tokens), np.float32) if groups is not None else None

    def observe_chunk(self, layer, begin, probability, value, output_weight):
        mask = (self.rows >= begin) & (self.rows < begin + probability.shape[2])
        if not mask.any():
            return
        rows = self.rows[mask] - begin
        selected = np.flatnonzero(self.selected // self.heads == layer)
        for slot in selected:
            head = int(self.selected[slot] % self.heads)
            self.maps[slot, mask] = probability[0, head, rows].detach().float().cpu().numpy()
        if self.groups is not None:
            for group_index, group in enumerate(self.groups):
                ids = group[group // self.heads == layer] % self.heads
                if len(ids):
                    total = probability[0, ids][:, rows].float().sum(0) / len(group)
                    self.group_maps[group_index, mask] += total.detach().cpu().numpy()


def representative_heads(trace: dict, per_group: int = 2) -> np.ndarray:
    """Select span-rank representatives, never by labels, peaks or effect size."""
    chosen = []
    for group in (trace["local_heads"], trace["global_heads"]):
        if len(group):
            chosen.extend(group[np.linspace(0, len(group) - 1, min(per_group, len(group)), dtype=int)])
    return np.unique(chosen).astype(int)


@torch.inference_mode()
def capture_rhythm(model, token_ids, response_start: int, evidence_mask, *,
                   config: RhythmConfig = RhythmConfig(), query_chunk: int = 8,
                   map_tokens: int = 128, map_offset: int = 0,
                   explicit_heads: tuple[tuple[int, int], ...] = (), paper_groups: bool = False):
    """One no-gradient forward for all curves; one OPTIONAL forward for maps.

    Uses the existing shared Llama implementation, including GQA/RoPE. No new
    model forward, target selection, source cut or trainable encoder is defined.
    """
    from experiments.common.llama_message_intervention import (
        VALIDATED_ATTRIBUTE, forward_layers, validate_manual_forward,
    )
    if query_chunk < 1 or map_tokens < 0 or map_offset < 0:
        raise ValueError("invalid capture/map budget")
    model.eval()
    device = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
    if not getattr(model, VALIDATED_ATTRIBUTE, False):
        validate_manual_forward(model, ids)
    layers, heads = len(model.model.layers), model.config.num_attention_heads
    observer = AttentionRhythmObserver(layers, heads, len(ids), response_start, evidence_mask, config)
    hidden = model.get_input_embeddings()(ids[None])
    final = forward_layers(model, hidden, 0, observer=observer, attention_query_chunk=query_chunk)
    result = observer.finish()
    entropy = np.full(len(observer.rows), np.nan, np.float32)
    # Entropy[q] predicts token q+1; the last response row has no observed next token.
    predictors = observer.rows[:-1]
    for begin in range(0, len(predictors), 32):
        rows = predictors[begin:begin + 32]
        logits = model.lm_head(final[0, rows]).float()
        entropy[begin:begin + len(rows)] = (
            logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)
        ).cpu().numpy()
    result.update(token_ids=ids.cpu().numpy(), predictor_entropy=entropy,
                  rhythm_schema=np.array(1), labels_used_for_capture=np.array(False))
    del final, observer
    selected = (np.asarray([l * heads + h for l, h in explicit_heads], dtype=int)
                if explicit_heads else representative_heads(result))
    if any(not (0 <= l < layers and 0 <= h < heads) for l, h in explicit_heads):
        raise ValueError("--head must name an existing layer:head")
    if map_tokens and len(selected):
        first = response_start - 1 + map_offset
        rows = np.arange(first, min(first + map_tokens, len(ids)))
        if not len(rows):
            raise ValueError("map offset is outside captured response")
        groups = (result["local_heads"], result["global_heads"]) if paper_groups else None
        maps = RawMapObserver(selected, heads, rows, len(ids), groups)
        forward_layers(model, hidden, 0, observer=maps, attention_query_chunk=query_chunk)
        result.update(map_heads=selected, map_query_position=rows,
                      map_source_position=np.arange(len(ids)), attention_maps=maps.maps)
        if paper_groups:
            result["paper_group_maps"] = maps.group_maps
    return result
