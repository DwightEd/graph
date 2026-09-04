"""One-pass capture for internal prompt-revisit and anchor rhythm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.common.llama_message_intervention import baseline_forward

from .claims import sentence_boundaries
from .rhythm import build_rhythm
from .routes import RouteAccumulator

CAPTURE_SCHEMA = 4


@dataclass(frozen=True)
class SampleCapture:
    arrays: dict[str, object]


def decode_tokens(tokenizer, token_ids) -> np.ndarray:
    return np.asarray(
        [
            tokenizer.decode(
                [int(token)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for token in np.asarray(token_ids).reshape(-1)
        ],
        dtype="U32",
    )


def capture_sample(
    model,
    tokenizer,
    token_ids,
    response_start: int,
    prompt_evidence_mask,
    *,
    sample_id: str,
    source_id: str,
    task_type: str,
    model_id: str,
    query_chunk: int = 64,
    route_window: int = 4,
    future_horizon: int = 16,
    far_lag: int = 32,
    peak_quantile: float = 0.9,
    max_lag: int = 3,
    detail: bool = False,
) -> SampleCapture:
    ids = torch.as_tensor(token_ids, dtype=torch.long).cpu()
    observer = RouteAccumulator(
        model,
        response_start,
        prompt_evidence_mask,
        route_window=route_window,
        future_horizon=future_horizon,
        far_lag=far_lag,
        detail=detail,
    )
    cache = baseline_forward(
        model,
        ids,
        response_start,
        observer=observer,
        checkpoint_layers=(0,),
        attention_query_chunk=query_chunk,
    )
    trace = observer.finish()
    rhythm = build_rhythm(
        trace,
        revisit_window=route_window,
        peak_quantile=peak_quantile,
        max_lag=max_lag,
    )

    arrays: dict[str, object] = {
        "capture_schema": CAPTURE_SCHEMA,
        "sample_id": sample_id,
        "source_id": source_id,
        "task_type": task_type,
        "model_id": model_id,
        "response_start": response_start,
        "query_position": cache.query,
        "prediction_position": cache.query + 1,
        "target_token_id": cache.target,
        "baseline_target_logprob": cache.baseline_target_logprob,
        "baseline_entropy": cache.baseline_entropy,
        "prompt_share_layer": trace.prompt_share,
        "evidence_share_layer": trace.evidence_share,
        "history_share_layer": trace.history_share,
        "far_prompt_share_layer": trace.far_prompt_share,
        "prompt_breadth_layer": trace.prompt_breadth,
        "route_change_layer": trace.route_change,
        "future_influence_layer": trace.future_influence,
        "route_change": rhythm.route_change,
        "prompt_revisit": rhythm.prompt_revisit,
        "evidence_revisit": rhythm.evidence_revisit,
        "history_share": rhythm.history_share,
        "prompt_breadth": rhythm.prompt_breadth,
        "future_influence": rhythm.future_influence,
        "revisit_delta": rhythm.revisit_delta,
        "evidence_delta": rhythm.evidence_delta,
        "revisit_peak": rhythm.revisit_peaks,
        "anchor_peak": rhythm.anchor_peaks,
        "revisit_peak_kind": rhythm.peak_kind,
        "paired_anchor": rhythm.paired_anchor,
        "coupling_rate": rhythm.coupling_rate,
        "coupling_null_rate": rhythm.null_rate,
        "median_anchor_lag": rhythm.median_lag,
        "sentence_boundary_position": sentence_boundaries(
            tokenizer, ids.numpy(), response_start
        ),
        "detail": int(detail),
    }
    if detail and trace.detail is not None:
        arrays.update(
            detail_edge_map=trace.detail["edge_map"],
            detail_prompt_head=trace.detail["prompt_head"],
            detail_route_change_head=trace.detail["route_change_head"],
            detail_future_head=trace.detail["future_head"],
            token_text=decode_tokens(tokenizer, ids.numpy()),
            detail_prompt_evidence_mask=np.asarray(prompt_evidence_mask, dtype=bool),
        )
    return SampleCapture(arrays)
