"""Exact response-relay geometry and fixed shortcut-route measurements.

The capture side keeps residual-space Gram tensors rather than deciding from a
hand-written scalar which route is truthful.  The analysis side derives a small
set of preregistered geometric contrasts from those Grams.  No hallucination
label is read here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .schema import (
    AUTONOMOUS_HISTORY_WRITE,
    DIRECT_EVIDENCE_WRITE,
    EVIDENCE,
    EVIDENCE_RELAY_CARRIER,
    EVIDENCE_RELAY_GATE,
    FULL,
    FULL_HISTORY_WRITE,
    HISTORY,
    NO_EVIDENCE,
    NO_EVIDENCE_HISTORY,
    REWIRED_EVIDENCE_RELAY_CARRIER,
    REWIRED_EVIDENCE_RELAY_GATE,
    SHORTCUT_VECTOR_NAMES,
)

_EPS = 1e-12

SHORTCUT_SCORE_NAMES = (
    "shortcut_route_incompleteness_mean",
    "shortcut_endpoint_rewire_gap_mean",
    "shortcut_autonomous_support_mean",
    "shortcut_route_candidate_mean",
    "shortcut_route_rewired_control_mean",
)

SHORTCUT_SCORE_DEFINITIONS = {
    "shortcut_route_incompleteness_mean": (
        "one minus the residual-energy projection of the full response-history "
        "write onto direct-evidence and evidence-conditioned relay writes"
    ),
    "shortcut_endpoint_rewire_gap_mean": (
        "route completion after adjacent response-endpoint rewiring minus route "
        "completion with the observed endpoints"
    ),
    "shortcut_autonomous_support_mean": (
        "signed contribution of the no-evidence history write to the full "
        "history-write direction"
    ),
    "shortcut_route_candidate_mean": (
        "observed route incompleteness times positive autonomous residual alignment"
    ),
    "shortcut_route_rewired_control_mean": (
        "the same candidate after adjacent response-endpoint rewiring"
    ),
}


def _adjacent_swap(index: torch.Tensor) -> torch.Tensor:
    """Return a deterministic near-lag derangement for a response prefix."""

    result = index.clone()
    paired = len(index) - len(index) % 2
    if paired:
        left = torch.arange(0, paired, 2, device=index.device)
        right = left + 1
        result[left], result[right] = index[right], index[left]
    return result


def capture_shortcut_geometry(
    attention: torch.Tensor,
    value: torch.Tensor,
    roles: Sequence[torch.Tensor],
    *,
    q_to_kv: torch.Tensor,
    output_weight: torch.Tensor,
    output_gram: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Capture route-completion vectors as exact residual-space Gram tensors.

    Parameters
    ----------
    attention:
        Actual post-intervention coefficients ``[branch,row,head,source]``.
    value:
        Branch-specific value states ``[branch,source,kv_head,head_dim]``.
    roles:
        Four disjoint ``[row,source]`` masks in the shared schema order.

    The observed vectors are

    ``full_history``
        The complete full-branch write from strict response history.
    ``direct_evidence``
        The full-branch write from direct evidence endpoints.
    ``evidence_relay_carrier``
        ``mean(A) delta(V)`` over response endpoints for ``full - noE``.
    ``evidence_relay_gate``
        ``delta(A) mean(V)`` over response endpoints for ``full - noE``.
    ``autonomous_history``
        The exact history-root write for ``noE - noEH``.

    The rewire control swaps adjacent response value endpoints while preserving
    every target row, head-specific coefficient, and the value multiset.  It
    therefore breaks endpoint identity without replacing the observed route by
    random noise.
    """

    if attention.ndim != 4 or value.ndim != 4:
        raise ValueError("attention/value must be branch-aligned rank-four tensors")
    branches, rows, heads, sources = attention.shape
    if branches < 4 or value.shape[:2] != (branches, sources):
        raise ValueError("shortcut geometry requires four aligned replay branches")
    if len(roles) != 4 or any(role.shape != (rows, sources) for role in roles):
        raise ValueError("source roles must partition every target/source pair")
    q_to_kv = q_to_kv.to(device=value.device, dtype=torch.long)
    if q_to_kv.shape != (heads,):
        raise ValueError("q_to_kv must contain one KV index per query head")
    value_by_head = value[:, :, q_to_kv].float()
    if value_by_head.shape[2] != heads:
        raise ValueError("value states do not match the query-head mapping")
    head_dim = value_by_head.shape[-1]
    if output_weight.shape[1] != heads * head_dim:
        raise ValueError("output projection does not match head geometry")
    if output_gram.shape != (heads, head_dim, head_dim):
        raise ValueError("per-head output Gram has incompatible geometry")

    attention = attention.float()
    evidence = roles[EVIDENCE].to(device=attention.device, dtype=torch.bool)
    history = roles[HISTORY].to(device=attention.device, dtype=torch.bool)

    def context(
        coefficient: torch.Tensor,
        source_value: torch.Tensor,
        role: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "rhs,shd->rhd", coefficient * role[:, None], source_value
        )

    a_full = attention[FULL]
    a_no_evidence = attention[NO_EVIDENCE]
    a_no_both = attention[NO_EVIDENCE_HISTORY]
    v_full = value_by_head[FULL]
    v_no_evidence = value_by_head[NO_EVIDENCE]
    v_no_both = value_by_head[NO_EVIDENCE_HISTORY]

    full_history = context(a_full, v_full, history)
    direct_evidence = context(a_full, v_full, evidence)
    mean_attention = 0.5 * (a_full + a_no_evidence)
    delta_attention = a_full - a_no_evidence
    delta_value = v_full - v_no_evidence
    mean_value = 0.5 * (v_full + v_no_evidence)
    evidence_carrier = context(mean_attention, delta_value, history)
    evidence_gate = context(delta_attention, mean_value, history)
    no_evidence_history = context(a_no_evidence, v_no_evidence, history)
    evidence_conditioned_history = full_history - no_evidence_history
    relay_reconstruction = evidence_carrier + evidence_gate
    relay_closure_error = (
        evidence_conditioned_history - relay_reconstruction
    ).flatten(1).norm(dim=-1)
    relay_scale = torch.maximum(
        evidence_conditioned_history.flatten(1).norm(dim=-1),
        relay_reconstruction.flatten(1).norm(dim=-1),
    )
    if not torch.all(
        relay_closure_error <= 5e-5 + 5e-5 * relay_scale
    ):
        raise ValueError("evidence relay midpoint decomposition does not close")
    autonomous_history = no_evidence_history - context(
        a_no_both, v_no_both, history
    )

    rewired_carrier = torch.zeros_like(evidence_carrier)
    rewired_gate = torch.zeros_like(evidence_gate)
    rewire_valid = torch.zeros(rows, dtype=torch.bool, device=attention.device)
    for row in range(rows):
        endpoint = torch.nonzero(history[row], as_tuple=False).flatten()
        if len(endpoint) < 2:
            continue
        rewired = _adjacent_swap(endpoint)
        carrier_weight = mean_attention[row, :, endpoint]
        gate_weight = delta_attention[row, :, endpoint]
        rewired_carrier[row] = torch.einsum(
            "hk,khd->hd", carrier_weight, delta_value[rewired]
        )
        rewired_gate[row] = torch.einsum(
            "hk,khd->hd", gate_weight, mean_value[rewired]
        )
        rewire_valid[row] = bool(torch.any(rewired != endpoint))

    contexts = torch.stack(
        (
            full_history,
            direct_evidence,
            evidence_carrier,
            evidence_gate,
            autonomous_history,
            rewired_carrier,
            rewired_gate,
        ),
        dim=1,
    )
    vector_count = len(SHORTCUT_VECTOR_NAMES)
    projected = F.linear(
        contexts.reshape(rows * vector_count, heads * head_dim),
        output_weight.float(),
        bias=None,
    ).reshape(rows, vector_count, output_weight.shape[0])
    route_gram = torch.einsum("rkd,rmd->rkm", projected, projected)
    head_gram = torch.einsum(
        "rkhd,hde,rmhe->rhkm",
        contexts,
        output_gram.float(),
        contexts,
    )
    route_gram = 0.5 * (route_gram + route_gram.transpose(-1, -2))
    head_gram = 0.5 * (head_gram + head_gram.transpose(-1, -2))
    if not torch.isfinite(route_gram).all() or not torch.isfinite(head_gram).all():
        raise ValueError("shortcut route Gram contains non-finite values")
    return {
        "route_gram": route_gram,
        "head_gram": head_gram,
        "rewire_valid": rewire_valid,
        "relay_closure_error": relay_closure_error,
    }


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _support_matrix(gram: np.ndarray, support: Sequence[int]) -> np.ndarray:
    return np.take(np.take(gram, support, axis=-2), support, axis=-1)


def _support_inverse(gram: np.ndarray, support: Sequence[int]) -> np.ndarray:
    return np.linalg.pinv(_support_matrix(gram, support), rcond=1e-8)


def _projection_fraction(
    gram: np.ndarray, target: int, support: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    inverse = _support_inverse(gram, support)
    cross = np.take(gram[..., target, :], support, axis=-1)
    projected = np.einsum("...i,...ij,...j->...", cross, inverse, cross)
    energy = np.maximum(gram[..., target, target], 0.0)
    fraction = np.clip(projected / np.maximum(energy, _EPS), 0.0, 1.0)
    valid = energy > _EPS
    return fraction, valid


def _signed_support(
    gram: np.ndarray, source: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project additive source writes onto the full-history direction.

    For an exact decomposition ``h = r + a``, the returned signed supports
    ``<h,r>/||h||²`` and ``<h,a>/||h||²`` sum to one.  Unlike residualizing
    both ``h`` and ``a`` against ``r``, this quantity is not an algebraic
    identity equal to one; it retains reinforcement and cancellation.
    """

    energy = np.maximum(gram[..., FULL_HISTORY_WRITE, FULL_HISTORY_WRITE], 0.0)
    cross = np.take(gram[..., FULL_HISTORY_WRITE, :], source, axis=-1).sum(-1)
    valid = energy > _EPS
    support = np.zeros_like(energy)
    support[valid] = cross[valid] / energy[valid]
    return support, valid


def _combined_norm(gram: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    block = _support_matrix(gram, indices)
    return np.sqrt(np.maximum(block.sum(axis=(-1, -2)), 0.0))


def shortcut_layer_metrics(trace: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return fixed ``[layer,token]`` measurements from the stored raw Grams."""

    gram = _array(trace["shortcut_route_gram"]).astype(np.float64)
    vector_count = len(SHORTCUT_VECTOR_NAMES)
    if gram.ndim != 4 or gram.shape[-2:] != (vector_count, vector_count):
        raise ValueError("shortcut_route_gram must be [layer,token,vector,vector]")
    if not np.isfinite(gram).all():
        raise ValueError("shortcut_route_gram must be finite")
    gram = 0.5 * (gram + gram.swapaxes(-1, -2))
    rewire_valid = _array(trace["shortcut_rewire_valid"]).astype(bool)
    if rewire_valid.shape != gram.shape[:2]:
        raise ValueError("shortcut_rewire_valid must be [layer,token]")

    relay_support = (EVIDENCE_RELAY_CARRIER, EVIDENCE_RELAY_GATE)
    evidence_support = (
        DIRECT_EVIDENCE_WRITE,
        EVIDENCE_RELAY_CARRIER,
        EVIDENCE_RELAY_GATE,
    )
    rewired_support = (
        DIRECT_EVIDENCE_WRITE,
        REWIRED_EVIDENCE_RELAY_CARRIER,
        REWIRED_EVIDENCE_RELAY_GATE,
    )
    relay_completion, history_valid = _projection_fraction(
        gram, FULL_HISTORY_WRITE, relay_support
    )
    route_completion, _ = _projection_fraction(
        gram, FULL_HISTORY_WRITE, evidence_support
    )
    rewired_completion, _ = _projection_fraction(
        gram, FULL_HISTORY_WRITE, rewired_support
    )
    evidence_relay_support, support_valid = _signed_support(
        gram, (EVIDENCE_RELAY_CARRIER, EVIDENCE_RELAY_GATE)
    )
    autonomous_support, autonomous_valid = _signed_support(
        gram, (AUTONOMOUS_HISTORY_WRITE,)
    )
    additive_support_error = np.abs(
        evidence_relay_support + autonomous_support - 1.0
    )

    route_incompleteness = 1.0 - route_completion
    endpoint_rewire_gap = rewired_completion - route_completion
    shortcut_candidate = route_incompleteness * np.maximum(autonomous_support, 0.0)
    rewired_candidate = (1.0 - rewired_completion) * np.maximum(
        autonomous_support, 0.0
    )
    valid_rewire = history_valid & rewire_valid

    values = {
        "shortcut_history_write_norm": np.sqrt(
            np.maximum(gram[..., FULL_HISTORY_WRITE, FULL_HISTORY_WRITE], 0.0)
        ),
        "shortcut_direct_evidence_write_norm": np.sqrt(
            np.maximum(gram[..., DIRECT_EVIDENCE_WRITE, DIRECT_EVIDENCE_WRITE], 0.0)
        ),
        "shortcut_evidence_relay_write_norm": _combined_norm(gram, relay_support),
        "shortcut_autonomous_history_write_norm": np.sqrt(
            np.maximum(
                gram[..., AUTONOMOUS_HISTORY_WRITE, AUTONOMOUS_HISTORY_WRITE], 0.0
            )
        ),
        "shortcut_relay_completion": relay_completion,
        "shortcut_route_completion": route_completion,
        "shortcut_route_incompleteness": route_incompleteness,
        "shortcut_rewired_route_completion": rewired_completion,
        "shortcut_endpoint_rewire_gap": endpoint_rewire_gap,
        "shortcut_evidence_relay_support": evidence_relay_support,
        "shortcut_autonomous_support": autonomous_support,
        "shortcut_additive_support_error": additive_support_error,
        "shortcut_route_candidate": shortcut_candidate,
        "shortcut_route_rewired_control": rewired_candidate,
    }
    validity = {
        "shortcut_history_write_norm": history_valid,
        "shortcut_direct_evidence_write_norm": history_valid,
        "shortcut_evidence_relay_write_norm": history_valid,
        "shortcut_autonomous_history_write_norm": history_valid,
        "shortcut_relay_completion": history_valid,
        "shortcut_route_completion": history_valid,
        "shortcut_route_incompleteness": history_valid,
        "shortcut_rewired_route_completion": valid_rewire,
        "shortcut_endpoint_rewire_gap": valid_rewire,
        "shortcut_evidence_relay_support": history_valid & support_valid,
        "shortcut_autonomous_support": history_valid & autonomous_valid,
        "shortcut_additive_support_error": history_valid & support_valid & autonomous_valid,
        "shortcut_route_candidate": history_valid & autonomous_valid,
        "shortcut_route_rewired_control": valid_rewire & autonomous_valid,
    }
    return {
        **{name: np.asarray(value, dtype=np.float32) for name, value in values.items()},
        **{f"{name}__valid": valid for name, valid in validity.items()},
    }


def _masked_mean(
    value: np.ndarray, valid: np.ndarray, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    selected = valid[start:stop]
    count = selected.sum(axis=0)
    total = np.where(selected, value[start:stop], 0.0).sum(axis=0)
    return total / np.maximum(count, 1), count > 0


def shortcut_token_metrics(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Reduce the raw layer geometry without selecting layers from labels."""

    layer = shortcut_layer_metrics(artifact["trace"])
    result: dict[str, np.ndarray] = {}
    names = tuple(name for name in layer if not name.endswith("__valid"))
    layers = next(iter(layer.values())).shape[0]
    width = max(layers // 3, 1)
    for name in names:
        value = np.asarray(layer[name], dtype=np.float64)
        valid = np.asarray(layer[f"{name}__valid"], dtype=bool)
        mean, mean_valid = _masked_mean(value, valid, 0, layers)
        early, early_valid = _masked_mean(value, valid, 0, width)
        late, late_valid = _masked_mean(value, valid, layers - width, layers)
        shift_valid = early_valid & late_valid
        shift = np.where(shift_valid, late - early, 0.0)
        result[f"{name}_mean"] = mean.astype(np.float32)
        result[f"{name}_mean__valid"] = mean_valid
        result[f"{name}_layer_shift"] = shift.astype(np.float32)
        result[f"{name}_layer_shift__valid"] = shift_valid
    return result
