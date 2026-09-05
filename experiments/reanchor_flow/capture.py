"""One-pass rhythm capture plus optional grouped causal mechanism audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.common.llama_message_intervention import baseline_forward

from .artifacts import CAPTURE_SCHEMA
from .claims import sentence_boundaries
from .mechanism import capture_mechanism
from .rhythm import build_rhythm
from .routes import RouteAccumulator

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
    distance_scale: int = 16,
    peak_quantile: float = 0.9,
    max_lag: int = 3,
    detail: bool = False,
    mechanism: bool = False,
) -> SampleCapture:
    ids = torch.as_tensor(token_ids, dtype=torch.long).cpu()
    observer = RouteAccumulator(
        model,
        response_start,
        prompt_evidence_mask,
        route_window=route_window,
        future_horizon=future_horizon,
        distance_scale=distance_scale,
        detail=detail,
    )
    checkpoints = range(len(model.model.layers)) if mechanism else (0,)
    cache = baseline_forward(
        model,
        ids,
        response_start,
        observer=observer,
        checkpoint_layers=checkpoints,
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
        "predictor_position": cache.query,
        "prediction_position": cache.query + 1,
        "emitted_position": cache.query + 1,
        "target_token_id": cache.target,
        "baseline_target_logprob": cache.baseline_target_logprob,
        "baseline_entropy": cache.baseline_entropy,
        "prompt_share_layer": trace.prompt_share,
        "evidence_share_layer": trace.evidence_share,
        "history_share_layer": trace.history_share,
        "prompt_lift_layer": trace.prompt_lift,
        "evidence_lift_layer": trace.evidence_lift,
        "history_lift_layer": trace.history_lift,
        "nonlocality_layer": trace.nonlocality,
        "prompt_breadth_layer": trace.prompt_breadth,
        "route_change_layer": trace.route_change,
        "predictor_reuse_layer": trace.predictor_reuse,
        "future_influence_layer": trace.future_influence,
        "emitted_token_anchor_layer": trace.future_influence,
        "head_attention_prompt_mass": trace.head["attention_prompt_mass"],
        "head_attention_evidence_mass": trace.head["attention_evidence_mass"],
        "head_attention_history_mass": trace.head["attention_history_mass"],
        "head_prompt_transport_share": trace.head["prompt_share"],
        "head_evidence_transport_share": trace.head["evidence_share"],
        "head_history_transport_share": trace.head["history_share"],
        "head_nonlocality": trace.head["nonlocality"],
        "head_route_change": trace.head["route_change"],
        "head_predictor_reuse": trace.head["predictor_reuse"],
        "head_emitted_token_anchor": trace.head["future_influence"],
        "route_change": rhythm.route_change,
        "prompt_share": rhythm.prompt_share,
        "evidence_share": rhythm.evidence_share,
        "history_share": rhythm.history_share,
        "prompt_lift": rhythm.prompt_lift,
        "evidence_lift": rhythm.evidence_lift,
        "history_lift": rhythm.history_lift,
        "nonlocality": rhythm.nonlocality,
        "prompt_breadth": rhythm.prompt_breadth,
        "predictor_reuse": rhythm.predictor_reuse,
        "future_influence": rhythm.future_influence,
        "emitted_token_anchor": rhythm.future_influence,
        "prompt_delta": rhythm.prompt_delta,
        "evidence_delta": rhythm.evidence_delta,
        "nonlocal_delta": rhythm.nonlocal_delta,
        "transition_peak": rhythm.transition_peaks,
        "prompt_peak": rhythm.prompt_peaks,
        "review_peak": rhythm.review_peaks,
        "anchor_peak": rhythm.anchor_peaks,
        "prompt_paired_anchor": rhythm.prompt_paired_anchor,
        "review_paired_anchor": rhythm.review_paired_anchor,
        "prompt_coupling_rate": rhythm.prompt_coupling_rate,
        "prompt_coupling_null_rate": rhythm.prompt_null_rate,
        "prompt_median_anchor_lag": rhythm.prompt_median_lag,
        "review_coupling_rate": rhythm.review_coupling_rate,
        "review_coupling_null_rate": rhythm.review_null_rate,
        "review_median_anchor_lag": rhythm.review_median_lag,
        "sentence_boundary_position": sentence_boundaries(
            tokenizer, ids.numpy(), response_start
        ),
        "detail": int(detail),
        "functional": 0,
        "mechanism": 0,
    }
    arrays.update(
        capture_mechanism(
            model,
            cache,
            response_start,
            prompt_evidence_mask,
            grouped=mechanism,
        )
    )
    if detail and trace.detail is not None:
        arrays.update(
            detail_edge_map=trace.detail["edge_map"],
            detail_prompt_head=trace.detail["prompt_head"],
            detail_evidence_head=trace.detail["evidence_head"],
            detail_nonlocal_head=trace.detail["nonlocal_head"],
            detail_route_change_head=trace.detail["route_change_head"],
            detail_predictor_reuse_head=trace.detail["predictor_reuse_head"],
            detail_future_head=trace.detail["future_head"],
            token_text=decode_tokens(tokenizer, ids.numpy()),
            detail_prompt_evidence_mask=np.asarray(prompt_evidence_mask, dtype=bool),
        )
    return SampleCapture(arrays)
