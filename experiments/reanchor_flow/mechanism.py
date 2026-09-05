"""Functional context cut plus optional grouped mechanism audit.

Every sample receives one context-path cut.  The source-diverse deep subset
adds three grouped cuts and a layerwise state trace.  Interventions remove
source messages only on response-query rows; no individual edge search is run.
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


VOCAB_EVENT_CHUNK = 16
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


def _run_trace(
    model,
    cache: ForwardCache,
    gate: MessageGate,
    *,
    layerwise: bool,
) -> CutTrace:
    device = model.get_input_embeddings().weight.device
    saved: dict[int, Tensor] = {}
    with torch.inference_mode():
        final = forward_layers(
            model,
            cache.layer_input[0].to(device)[None],
            0,
            gate=gate_to(gate, device),
            save_inputs=saved if layerwise else None,
            save_layers=set(range(cache.layer_count)) if layerwise else None,
            attention_query_chunk=cache.attention_query_chunk,
        )[0]
        change = _fixed_margin(model, cache, final).cpu() - cache.full_margin
    return CutTrace(saved, final.detach().cpu(), change)


def vocabulary_effect(
    model,
    cache: ForwardCache,
    baseline_final: Tensor,
    cut_final: Tensor,
    *,
    top_k: int = EVIDENCE_CANDIDATES,
    chunk: int = VOCAB_EVENT_CHUNK,
) -> dict[str, np.ndarray]:
    """Map a context-path cut to supported vocabulary candidates and adoption.

    The evidence mask currently covers the external-context region, not an
    exact claim-support span, so exported fields intentionally use ``context``.
    """

    device = model.get_input_embeddings().weight.device
    query = cache.query.long()
    target = cache.target.to(device)
    baseline_model = baseline_final.index_select(0, query).to(device)
    cut_model = cut_final.index_select(0, query).to(device)
    weight = model.lm_head.weight
    bias = getattr(model.lm_head, "bias", None)
    top_k = min(int(top_k), len(weight))
    result = {
        "context_distribution_js": [],
        "context_target_logprob_gain": [],
        "context_candidate_id": [],
        "context_candidate_logprob_gain": [],
        "context_target_rank": [],
        "context_adoption_margin": [],
    }
    with torch.inference_mode():
        for begin in range(0, len(target), chunk):
            end = min(begin + chunk, len(target))
            current_target = target[begin:end]
            rows = torch.arange(end - begin, device=device)
            baseline_logits = F.linear(
                baseline_model[begin:end].to(weight.dtype), weight, bias
            ).float()
            cut_logits = F.linear(
                cut_model[begin:end].to(weight.dtype), weight, bias
            ).float()
            log_p = baseline_logits.log_softmax(dim=1)
            log_q = cut_logits.log_softmax(dim=1)
            del baseline_logits, cut_logits
            log_middle = torch.logaddexp(log_p, log_q) - np.log(2.0)
            p = log_p.exp()
            q = log_q.exp()
            distribution_js = 0.5 * (
                (p * (log_p - log_middle)).sum(1)
                + (q * (log_q - log_middle)).sum(1)
            )
            del log_middle, p, q

            gain = log_p - log_q
            target_logprob_gain = gain[rows, current_target]
            target_rank = 1 + (gain > target_logprob_gain[:, None]).sum(1)
            gain[rows, current_target] = -torch.inf
            candidate_gain, candidate_id = torch.topk(gain, k=top_k, dim=1)

            result["context_distribution_js"].append(distribution_js.cpu())
            result["context_target_logprob_gain"].append(target_logprob_gain.cpu())
            result["context_candidate_id"].append(candidate_id.to(torch.int32).cpu())
            result["context_candidate_logprob_gain"].append(
                candidate_gain.cpu()
            )
            result["context_target_rank"].append(target_rank.to(torch.int32).cpu())
            result["context_adoption_margin"].append(
                (target_logprob_gain - candidate_gain[:, 0]).cpu()
            )

    arrays = {name: torch.cat(values).numpy() for name, values in result.items()}
    arrays["context_target_log_rank"] = np.log(
        arrays["context_target_rank"].astype(np.float32)
    )
    return arrays


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
    *,
    grouped: bool = True,
) -> dict[str, object]:
    """Measure context adoption, plus grouped state cuts when requested.

    Evidence and other-prompt cuts affect only response-query rows.  Prompt
    encoding therefore remains fixed; the estimand is direct re-entry into the
    response computation, not removal of all semantic ancestry.
    """

    sources = cache.layer_input[0].shape[0]
    targets = _response_targets(sources, response_start)
    evidence, other, prompt, history = _role_masks(
        sources, response_start, evidence_mask
    )

    evidence_trace = _run_trace(
        model,
        cache,
        _gate(evidence, targets),
        layerwise=grouped,
    )
    baseline_final = cache.final_hidden
    evidence_effect = -evidence_trace.margin_change.numpy()
    result = {
        "functional": 1,
        "mechanism": int(grouped),
        "evidence_effect": evidence_effect,
    }
    result.update(
        vocabulary_effect(
            model,
            cache,
            baseline_final,
            evidence_trace.final_hidden,
        )
    )
    if grouped:
        presence, control, readout_gain = _state_trace(
            model, cache, evidence_trace, baseline_final
        )
        other_change = rerun_gate(model, cache, _gate(other, targets))
        prompt_change = rerun_gate(model, cache, _gate(prompt, targets))
        history_change = rerun_gate(model, cache, _gate(history, targets))
        middle = max(1, cache.layer_count // 3)
        peak_control = np.nanmax(np.abs(control[middle:]), axis=0)
        result.update(
            mechanism_layer=np.arange(cache.layer_count + 1, dtype=np.int16),
            evidence_state_presence=presence,
            evidence_state_control=control,
            evidence_readout_gain=readout_gain,
            other_prompt_effect=-other_change.numpy(),
            prompt_effect=-prompt_change.numpy(),
            evidence_prompt_interaction=(
                evidence_effect - other_change.numpy() + prompt_change.numpy()
            ),
            history_effect=-history_change.numpy(),
            evidence_peak_control=peak_control,
            evidence_late_control_loss=peak_control - np.abs(control[-1]),
        )
    return result
