"""Capture one functional rhythm and its evidence-cut output effect."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .intervene import ForwardCache, RelayGate, baseline_forward, rerun_gate
from .rhythm import Rhythm, build_rhythm, relay_diagnostics
from .routes import FunctionalRouteAccumulator, FunctionalRoutes


@dataclass(frozen=True)
class SampleCapture:
    """Small persisted arrays plus per-sample maps used only for plotting."""

    arrays: dict[str, object]
    routes: FunctionalRoutes
    rhythm: Rhythm


def source_evidence_mask(
    prompt_evidence_mask: Tensor,
    source_count: int,
    response_start: int,
) -> Tensor:
    """Place the prompt evidence span in full teacher-forcing coordinates."""

    prompt = torch.as_tensor(prompt_evidence_mask, dtype=torch.bool).flatten()
    if len(prompt) != response_start:
        raise ValueError("evidence mask must cover the complete prompt")
    evidence = torch.zeros(source_count, dtype=torch.bool)
    evidence[:response_start] = prompt
    return evidence


def matched_non_evidence_mask(
    routes: FunctionalRoutes,
    evidence: Tensor,
    response_start: int,
) -> Tensor | None:
    """Greedily match evidence sources by prompt position and route mass."""

    roots = torch.nonzero(evidence[:response_start], as_tuple=False).flatten()
    candidates = torch.nonzero(~evidence[:response_start], as_tuple=False).flatten()
    if len(candidates) < len(roots):
        return None

    mass = routes.absolute_map.sum(dim=0)
    mass_scale = mass[:response_start].std(unbiased=False).clamp_min(1e-6)
    order = roots[mass.index_select(0, roots).argsort(descending=True)]
    available = candidates.clone()
    matched = torch.zeros_like(evidence)
    for root in order:
        cost = (available - root).abs() / max(response_start, 1)
        cost = cost + (mass.index_select(0, available) - mass[root]).abs() / mass_scale
        choice = cost.argmin()
        matched[available[choice]] = True
        available = torch.cat((available[:choice], available[choice + 1 :]))
    return matched


def relay_gate(
    rhythm: Rhythm,
    evidence: Tensor,
    split_layer: int,
    *,
    cut_evidence: bool = False,
    cut_upstream: bool = False,
    cut_downstream: bool = False,
    direct_response_only: bool = False,
) -> RelayGate:
    evidence_targets = None
    if direct_response_only:
        evidence_targets = torch.arange(len(evidence)) >= rhythm.query_position[0]
    return RelayGate(
        upstream_edges=rhythm.upstream_edges,
        downstream_edges=rhythm.downstream_edges,
        split_layer=split_layer,
        cut_evidence=cut_evidence,
        cut_upstream=cut_upstream,
        cut_downstream=cut_downstream,
        evidence_mask=evidence,
        evidence_targets=evidence_targets,
    )


def run_relay_audit(
    model,
    cache: ForwardCache,
    rhythm: Rhythm,
    evidence: Tensor,
    split_layer: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the three non-baseline cells of the U/D diagnostic."""

    upstream = rerun_gate(
        model,
        cache,
        relay_gate(rhythm, evidence, split_layer, cut_upstream=True),
    )
    downstream = rerun_gate(
        model,
        cache,
        relay_gate(rhythm, evidence, split_layer, cut_downstream=True),
    )
    joint = rerun_gate(
        model,
        cache,
        relay_gate(
            rhythm,
            evidence,
            split_layer,
            cut_upstream=True,
            cut_downstream=True,
        ),
    )
    interaction = relay_diagnostics(upstream, downstream, joint)[:, 2]
    return upstream, downstream, joint, interaction


def capture_sample(
    model,
    full_token_ids,
    response_start: int,
    prompt_evidence_mask,
    *,
    sample_id: str,
    source_id: str,
    task_type: str,
    model_id: str,
    audit_relay: bool = False,
    head_quantile: float = 0.3,
    query_chunk: int = 128,
    window: int = 10,
    horizon_low: int = 10,
    horizon_high: int = 100,
    carrier_quantile: float = 0.75,
    mass_floor: float = 1e-6,
    max_carriers: int = 8,
    split_layer: int | None = None,
) -> SampleCapture:
    """Run one baseline, one evidence cut, and an optional three-rerun audit."""

    token_ids = torch.as_tensor(full_token_ids, dtype=torch.long, device="cpu")
    source_count = len(token_ids) - 1
    evidence = source_evidence_mask(
        torch.as_tensor(prompt_evidence_mask), source_count, response_start
    )
    if not evidence.any():
        raise ValueError("the declared evidence span contains no tokens")

    layer_count = int(model.config.num_hidden_layers)
    if split_layer is None:
        split_layer = layer_count // 2
    if not 0 < split_layer < layer_count:
        raise ValueError("split_layer must leave nonempty early and late bands")

    accumulator = FunctionalRouteAccumulator(
        model,
        response_start,
        head_quantile=head_quantile,
        query_chunk=query_chunk,
        split_layer=split_layer,
    )
    cache = baseline_forward(
        model,
        token_ids,
        response_start,
        observer=accumulator.observe,
        checkpoint_layers=(0, split_layer) if audit_relay else (0,),
    )
    routes = accumulator.finish()
    del accumulator
    rhythm = build_rhythm(
        routes,
        response_start,
        evidence,
        window=window,
        horizon_low=horizon_low,
        horizon_high=horizon_high,
        carrier_quantile=carrier_quantile,
        mass_floor=mass_floor,
        max_carriers=max_carriers,
        split_layer=split_layer,
        build_endpoints=audit_relay,
    )

    # This signed, unnormalized rerun effect is the only detection score.
    deficit = rerun_gate(
        model,
        cache,
        relay_gate(rhythm, evidence, split_layer, cut_evidence=True),
    )
    valid = torch.isfinite(cache.full_margin) & torch.isfinite(deficit)
    deficit = deficit.masked_fill(~valid, torch.nan)
    cut_margin = cache.full_margin + deficit

    empty = torch.full_like(deficit, torch.nan)
    direct = empty
    matched_delta = empty
    upstream = downstream = joint = interaction = empty
    audited = bool(audit_relay and rhythm.carrier_mask.any())
    matched = None
    if audit_relay:
        matched = matched_non_evidence_mask(routes, evidence, response_start)
        direct = rerun_gate(
            model,
            cache,
            relay_gate(
                rhythm,
                evidence,
                split_layer,
                cut_evidence=True,
                direct_response_only=True,
            ),
        )
        if matched is not None:
            matched_delta = rerun_gate(
                model,
                cache,
                relay_gate(rhythm, matched, split_layer, cut_evidence=True),
            )
    if audited:
        upstream, downstream, joint, interaction = run_relay_audit(
            model, cache, rhythm, evidence, split_layer
        )

    arrays: dict[str, object] = {
        "sample_id": sample_id,
        "source_id": source_id,
        "task_type": task_type,
        "model_id": model_id,
        "response_start": response_start,
        "evidence_tokens": int(evidence.sum()),
        "split_layer": split_layer,
        "control_audited": audit_relay,
        "matched_control_available": matched is not None,
        "relay_audited": audited,
        "query_position": cache.query,
        "prediction_position": cache.query + 1,
        "target_token_id": cache.target,
        "runner_token_id": cache.runner,
        "baseline_margin": cache.full_margin,
        "baseline_target_logprob": cache.baseline_target_logprob,
        "baseline_entropy": cache.baseline_entropy,
        "cut_margin": cut_margin,
        "constraint_deficit": deficit,
        "valid": valid,
        "functional_mass": routes.absolute_map.sum(dim=1),
        "functional_reach": rhythm.functional_reach,
        "future_influence": rhythm.future_influence,
        "future_delivery": rhythm.future_delivery,
        "evidence_uptake": rhythm.evidence_uptake,
        "evidence_binding": rhythm.evidence_binding,
        "relay_capacity": rhythm.relay_capacity,
        "relay_mass": rhythm.relay_mass,
        "carrier_mask": rhythm.carrier_mask,
        "direct_response_cut_delta": direct,
        "matched_non_evidence_cut_delta": matched_delta,
        "upstream_cut_delta": upstream,
        "downstream_cut_delta": downstream,
        "joint_cut_delta": joint,
        "relay_interaction": interaction,
    }
    return SampleCapture(arrays=arrays, routes=routes, rhythm=rhythm)
