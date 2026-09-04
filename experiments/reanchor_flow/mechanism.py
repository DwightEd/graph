"""Layerwise evidence-entry, integration, persistence and readout audit.

The broad rhythm capture is single-pass.  This module is the optional deep
validation pass: one baseline plus four grouped message cuts per sample.  It
never deletes individual edges and stores only layer/token scalars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.common.llama_message_intervention import (
    ForwardCache,
    MessageGate,
    forward_layers,
    gate_to,
    rerun_gate,
)


@dataclass(frozen=True)
class CutTrace:
    layer_input: dict[int, Tensor]
    final_hidden: Tensor
    margin_change: Tensor


VOCAB_CHUNK = 4096
EVIDENCE_CANDIDATES = 5


def _response_targets(sources: int, response_start: int) -> Tensor:
    target = torch.zeros(sources, dtype=torch.bool)
    target[max(response_start - 1, 0) :] = True
    return target


def _role_masks(
    sources: int,
    response_start: int,
    evidence_mask,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    evidence = torch.zeros(sources, dtype=torch.bool)
    evidence[:response_start] = torch.as_tensor(evidence_mask, dtype=torch.bool)
    other_prompt = torch.zeros_like(evidence)
    other_prompt[:response_start] = ~evidence[:response_start]
    prompt = evidence | other_prompt
    history = torch.zeros_like(evidence)
    history[response_start:] = True
    return evidence, other_prompt, prompt, history


def _gate(source: Tensor, targets: Tensor) -> MessageGate:
    return MessageGate(
        split_layer=0,
        source_mask=source,
        source_targets=targets,
    )


def _fixed_margin(model, cache: ForwardCache, hidden: Tensor) -> Tensor:
    device = hidden.device
    response = hidden.index_select(0, cache.query.to(device))
    margin = torch.einsum(
        "td,td->t",
        response.float(),
        cache.readout_direction.to(device),
    )
    return margin + cache.readout_bias.to(device)


def _run_trace(model, cache: ForwardCache, gate: MessageGate) -> CutTrace:
    device = model.get_input_embeddings().weight.device
    saved: dict[int, Tensor] = {}
    with torch.inference_mode():
        final = forward_layers(
            model,
            cache.layer_input[0].to(device)[None],
            0,
            gate=gate_to(gate, device),
            save_inputs=saved,
            save_layers=set(range(cache.layer_count)),
            attention_query_chunk=cache.attention_query_chunk,
        )[0]
        change = _fixed_margin(model, cache, final).cpu() - cache.full_margin
    return CutTrace(saved, final.detach().cpu(), change)


def _baseline_final(model, cache: ForwardCache) -> Tensor:
    """Recover the baseline final state with only the last decoder layer."""

    last = cache.layer_count - 1
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        return forward_layers(
            model,
            cache.layer_input[last].to(device)[None],
            last,
            attention_query_chunk=cache.attention_query_chunk,
        )[0].detach().cpu()


def _log_normalizer(model, hidden: Tensor, chunk: int) -> Tensor:
    """Compute per-event logsumexp without materializing token x vocabulary."""

    weight = model.lm_head.weight
    bias = getattr(model.lm_head, "bias", None)
    projected = hidden.to(dtype=weight.dtype)
    normalizer = torch.full(
        (len(hidden),),
        -torch.inf,
        dtype=torch.float32,
        device=hidden.device,
    )
    for begin in range(0, len(weight), chunk):
        end = min(begin + chunk, len(weight))
        current_bias = None if bias is None else bias[begin:end]
        logits = F.linear(projected, weight[begin:end], current_bias).float()
        normalizer = torch.logaddexp(normalizer, torch.logsumexp(logits, dim=1))
    return normalizer


def vocabulary_effect(
    model,
    cache: ForwardCache,
    baseline_final: Tensor,
    cut_final: Tensor,
    *,
    top_k: int = EVIDENCE_CANDIDATES,
    chunk: int = VOCAB_CHUNK,
) -> dict[str, np.ndarray]:
    """Map a context-path cut to supported vocabulary candidates and adoption.

    The evidence mask currently covers the external-context region, not an
    exact claim-support span, so exported fields intentionally use ``context``.
    """

    if top_k < 1 or chunk < 1:
        raise ValueError("top_k and chunk must be positive")
    device = model.get_input_embeddings().weight.device
    query = cache.query.to(device)
    target = cache.target.to(device)
    baseline_model = baseline_final.to(device).index_select(0, query)
    cut_model = cut_final.to(device).index_select(0, query)
    weight = model.lm_head.weight
    bias = getattr(model.lm_head, "bias", None)
    top_k = min(int(top_k), len(weight))

    with torch.inference_mode():
        baseline_log_z = _log_normalizer(model, baseline_model, chunk)
        cut_log_z = _log_normalizer(model, cut_model, chunk)
        target_weight_model = weight.index_select(0, target)
        target_bias = (
            torch.zeros(len(target), device=device)
            if bias is None
            else bias.index_select(0, target).float()
        )
        baseline_target_logprob = (
            torch.einsum("td,td->t", baseline_model, target_weight_model).float()
            + target_bias
            - baseline_log_z
        )
        recorded = cache.baseline_target_logprob.to(device)
        tolerance = 5e-2 if weight.dtype == torch.bfloat16 else 1e-2
        if not torch.allclose(
            baseline_target_logprob,
            recorded,
            rtol=tolerance,
            atol=tolerance,
        ):
            error = float((baseline_target_logprob - recorded).abs().max())
            raise RuntimeError(
                f"baseline vocabulary reconstruction mismatch: max_abs={error:.6g}"
            )

        cut_target_logprob = (
            torch.einsum("td,td->t", cut_model, target_weight_model).float()
            + target_bias
            - cut_log_z
        )
        target_logprob_gain = baseline_target_logprob - cut_target_logprob
        normalizer_gain = baseline_log_z - cut_log_z
        # The normalizer is common to every vocabulary candidate, so adding it
        # back gives the exact target-logit intervention effect used for ranks.
        target_logit_gain = target_logprob_gain + normalizer_gain

        candidate_gain = torch.empty(
            (len(target), 0), dtype=torch.float32, device=device
        )
        candidate_id = torch.empty(
            (len(target), 0), dtype=torch.long, device=device
        )
        target_rank = torch.ones(len(target), dtype=torch.long, device=device)
        best_other = torch.full_like(target_logit_gain, -torch.inf)
        distribution_js = torch.zeros_like(target_logit_gain)

        for begin in range(0, len(weight), chunk):
            end = min(begin + chunk, len(weight))
            current_bias = None if bias is None else bias[begin:end]
            baseline_logits = F.linear(
                baseline_model, weight[begin:end], current_bias
            ).float()
            cut_logits = F.linear(
                cut_model, weight[begin:end], current_bias
            ).float()
            log_p = baseline_logits - baseline_log_z[:, None]
            log_q = cut_logits - cut_log_z[:, None]
            log_middle = torch.logaddexp(log_p, log_q) - np.log(2.0)
            p = log_p.exp()
            q = log_q.exp()
            distribution_js += 0.5 * (
                torch.where(p > 0, p * (log_p - log_middle), 0).sum(1)
                + torch.where(q > 0, q * (log_q - log_middle), 0).sum(1)
            )

            gain = baseline_logits - cut_logits
            target_rank += (gain > target_logit_gain[:, None]).sum(1)
            target_inside = (target >= begin) & (target < end)
            if bool(target_inside.any()):
                rows = torch.flatnonzero(target_inside)
                gain[rows, target[rows] - begin] = -torch.inf
            best_other = torch.maximum(best_other, gain.max(1).values)

            current_k = min(top_k, end - begin)
            values, indices = torch.topk(
                baseline_logits - cut_logits,
                k=current_k,
                dim=1,
            )
            indices += begin
            values = values - normalizer_gain[:, None]
            combined_gain = torch.cat((candidate_gain, values), dim=1)
            combined_id = torch.cat((candidate_id, indices), dim=1)
            keep = min(top_k, combined_gain.shape[1])
            candidate_gain, order = torch.topk(combined_gain, k=keep, dim=1)
            candidate_id = torch.gather(combined_id, 1, order)

    return {
        "context_distribution_js": distribution_js.cpu().numpy(),
        "context_target_logprob_gain": target_logprob_gain.cpu().numpy(),
        "context_candidate_id": candidate_id.to(torch.int32).cpu().numpy(),
        "context_candidate_logprob_gain": candidate_gain.cpu().numpy(),
        "context_target_rank": target_rank.to(torch.int32).cpu().numpy(),
        "context_target_log_rank": target_rank.float().log().cpu().numpy(),
        "context_adoption_margin": (
            target_logit_gain - best_other
        ).cpu().numpy(),
    }


def _logit_lens(model, state: Tensor, direction: Tensor, final: bool) -> Tensor:
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        hidden = state.to(device)
        if not final:
            hidden = model.model.norm(hidden)
        return torch.einsum(
            "td,td->t", hidden.float(), direction.to(device)
        ).cpu()


def _state_trace(
    model,
    cache: ForwardCache,
    cut: CutTrace,
    baseline_final: Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return relative state presence, logit-lens control and final readout gain."""

    query = cache.query.long()
    layers = cache.layer_count
    events = len(query)
    presence = torch.empty((layers + 1, events), dtype=torch.float32)
    control = torch.empty_like(presence)
    direction = cache.readout_direction.float()

    for layer in range(layers):
        baseline = cache.layer_input[layer].index_select(0, query)
        intervened = cut.layer_input[layer].index_select(0, query)
        difference = baseline.float() - intervened.float()
        presence[layer] = torch.linalg.vector_norm(difference, dim=-1) / (
            torch.linalg.vector_norm(baseline.float(), dim=-1) + 1e-12
        )
        control[layer] = _logit_lens(
            model, baseline, direction, final=False
        ) - _logit_lens(model, intervened, direction, final=False)

    baseline = baseline_final.index_select(0, query)
    intervened = cut.final_hidden.index_select(0, query)
    difference = baseline.float() - intervened.float()
    presence[-1] = torch.linalg.vector_norm(difference, dim=-1) / (
        torch.linalg.vector_norm(baseline.float(), dim=-1) + 1e-12
    )
    control[-1] = torch.einsum("td,td->t", difference, direction)
    gain = control[-1].abs() / (
        torch.linalg.vector_norm(difference, dim=-1)
        * torch.linalg.vector_norm(direction, dim=-1)
        + 1e-12
    )
    return presence.numpy(), control.numpy(), gain.numpy()


def capture_mechanism(
    model,
    cache: ForwardCache,
    response_start: int,
    evidence_mask,
) -> dict[str, object]:
    """Measure the four registered failure stages for every response token.

    Evidence and other-prompt cuts affect only response-query rows.  Prompt
    encoding therefore remains fixed; the estimand is direct re-entry into the
    response computation, not removal of all semantic ancestry.
    """

    sources = cache.layer_input[0].shape[0]
    targets = _response_targets(sources, response_start)
    evidence, other, prompt, history = _role_masks(
        sources, response_start, evidence_mask
    )

    evidence_trace = _run_trace(model, cache, _gate(evidence, targets))
    baseline_final = _baseline_final(model, cache)
    presence, control, readout_gain = _state_trace(
        model, cache, evidence_trace, baseline_final
    )

    other_change = rerun_gate(model, cache, _gate(other, targets))
    prompt_change = rerun_gate(model, cache, _gate(prompt, targets))
    history_change = rerun_gate(model, cache, _gate(history, targets))

    evidence_effect = -evidence_trace.margin_change.numpy()
    other_effect = -other_change.numpy()
    prompt_effect = -prompt_change.numpy()
    history_effect = -history_change.numpy()
    interaction = -evidence_trace.margin_change.numpy() - other_change.numpy()
    interaction += prompt_change.numpy()

    middle = max(1, cache.layer_count // 3)
    peak_control = np.nanmax(np.abs(control[middle:]), axis=0)
    final_control = control[-1]
    result = {
        "mechanism": 1,
        "mechanism_layer": np.arange(cache.layer_count + 1, dtype=np.int16),
        "evidence_state_presence": presence,
        "evidence_state_control": control,
        "evidence_readout_gain": readout_gain,
        "evidence_effect": evidence_effect,
        "other_prompt_effect": other_effect,
        "prompt_effect": prompt_effect,
        "evidence_prompt_interaction": interaction,
        "history_effect": history_effect,
        "evidence_peak_control": peak_control,
        "evidence_late_control_loss": peak_control - np.abs(final_control),
    }
    result.update(
        vocabulary_effect(
            model,
            cache,
            baseline_final,
            evidence_trace.final_hidden,
        )
    )
    return result
