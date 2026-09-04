"""One-pass, layer-resolved capture for the re-anchor phenomenon audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.common.llama_message_intervention import (
    MessageGate,
    baseline_forward,
    rerun_gate,
)

from .claims import split_claims
from .routes import RouteAccumulator

CAPTURE_SCHEMA = 3


@dataclass(frozen=True)
class SampleCapture:
    arrays: dict[str, object]


def evidence_source_mask(prompt_mask, source_count: int, response_start: int) -> torch.Tensor:
    prompt = torch.as_tensor(prompt_mask, dtype=torch.bool).flatten()
    if len(prompt) != response_start:
        raise ValueError("evidence mask must cover the complete prompt")
    evidence = torch.zeros(source_count, dtype=torch.bool)
    evidence[:response_start] = prompt
    return evidence


def evidence_gate(
    evidence: torch.Tensor,
    response_start: int,
    layer_count: int,
    *,
    direct_response_only: bool,
) -> MessageGate:
    targets = None
    if direct_response_only:
        targets = torch.arange(len(evidence)) >= response_start - 1
    return MessageGate(
        split_layer=layer_count,
        source_mask=evidence,
        source_targets=targets,
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
    causal_cuts: bool = False,
    query_chunk: int = 128,
    min_claim_tokens: int = 2,
    max_claim_tokens: int = 96,
) -> SampleCapture:
    """Capture observed routing once and optionally run two evidence cuts.

    The functional trace is message magnitude, not causal attribution. Optional
    cuts separately test direct response reads and all attention-mediated paths;
    MLP parametric knowledge remains active in both reruns.
    """

    ids = torch.as_tensor(token_ids, dtype=torch.long).cpu()
    source_count = len(ids) - 1
    evidence = evidence_source_mask(
        prompt_evidence_mask, source_count, response_start
    )
    if not bool(evidence.any()):
        raise ValueError("the declared evidence span contains no tokens")

    accumulator = RouteAccumulator(model, response_start, prompt_evidence_mask)
    cache = baseline_forward(
        model,
        ids,
        response_start,
        observer=accumulator,
        checkpoint_layers=(0,),
        attention_query_chunk=query_chunk,
    )
    routes = accumulator.finish()
    del accumulator
    functional = routes.functional_share.numpy()
    attention = routes.attention_share.numpy()
    claims = split_claims(
        tokenizer,
        ids.numpy(),
        response_start,
        min_tokens=min_claim_tokens,
        max_tokens=max_claim_tokens,
    )

    if causal_cuts:
        direct_delta = rerun_gate(
            model,
            cache,
            evidence_gate(
                evidence,
                response_start,
                cache.layer_count,
                direct_response_only=True,
            ),
        )
        global_delta = rerun_gate(
            model,
            cache,
            evidence_gate(
                evidence,
                response_start,
                cache.layer_count,
                direct_response_only=False,
            ),
        )

    arrays: dict[str, object] = {
        "capture_schema": CAPTURE_SCHEMA,
        "sample_id": sample_id,
        "source_id": source_id,
        "task_type": task_type,
        "model_id": model_id,
        "response_start": response_start,
        "evidence_tokens": int(evidence.sum()),
        "query_position": cache.query,
        "prediction_position": cache.query + 1,
        "target_token_id": cache.target,
        "runner_token_id": cache.runner,
        "baseline_margin": cache.full_margin,
        "baseline_target_logprob": cache.baseline_target_logprob,
        "baseline_entropy": cache.baseline_entropy,
        "functional_role_share": functional,
        "attention_role_share": attention,
        "functional_availability_null": routes.functional_null.numpy(),
        "attention_availability_null": routes.attention_null.numpy(),
        "functional_message_mass": routes.functional_mass,
        "causal_cuts": causal_cuts,
        "claim_start": np.asarray([claim.start for claim in claims], dtype=np.int64),
        "claim_stop": np.asarray([claim.stop for claim in claims], dtype=np.int64),
        "claim_boundary_kind": np.asarray(
            [claim.boundary_kind for claim in claims], dtype=np.int8
        ),
    }
    if causal_cuts:
        arrays["direct_evidence_cut_delta"] = direct_delta
        arrays["global_evidence_cut_delta"] = global_delta
    return SampleCapture(arrays=arrays)
