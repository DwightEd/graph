"""Candidate source-context compatibility on two strictly ordered layers.

This is a diagnostic for source-copyable candidates, not a truth probability.
Attention is a routing proxy; values, output projections and MLPs are absent.
"""

from __future__ import annotations

import numpy as np


def source_context_readout(
    attention, token_ids, roles, query, candidates, window=16, seed=20260912
):
    """Return graph/direct/shuffled/count readouts and explicit coverage.

    Inputs contain exactly two adjacent layers, receiver-first attention,
    and source=0 / instruction=1 / history=2 / special=3 roles. No entity
    positions, correct candidates, target token or future tokens are read.
    """
    ids, role = np.asarray(token_ids), np.asarray(roles)
    candidates = np.asarray(candidates)
    if (
        attention.ndim != 4
        or len(attention) != 2
        or attention.shape[-2:] != (len(ids), len(ids))
    ):
        raise ValueError("expected two attention layers aligned with token IDs")
    if role.shape != ids.shape or not np.isin(role, range(4)).all():
        raise ValueError("invalid token roles")
    if (
        not 0 <= query < len(ids)
        or window < 1
        or candidates.ndim != 1
        or len(candidates) < 2
    ):
        raise ValueError("invalid query, window or candidate set")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("duplicate candidates")
    a = np.asarray(attention[:, :, : query + 1, : query + 1], dtype=np.float64)
    if not np.isfinite(a).all() or (a < 0).any() or np.triu(a, 1).any():
        raise ValueError("attention must be finite, nonnegative and causal")
    if not np.allclose(a.sum(-1), 1, atol=0.005):
        raise ValueError("attention rows must sum to one")
    # Captured bf16 weights can have small row-sum error; normalize explicitly.
    a = a / a.sum(-1, keepdims=True)
    previous, current = a.mean(axis=1)
    source = role[: query + 1] == 0
    context_keys = source & ~np.isin(ids[: query + 1], candidates)
    positions = np.arange(query + 1)
    history = (
        (role[: query + 1] == 2) & (positions < query) & (positions >= query - window)
    )
    endpoints = np.flatnonzero(source & np.isin(ids[: query + 1], candidates))
    membership = ids[endpoints, None] == candidates[None, :]
    counts = membership.sum(0)
    direct = current[query, endpoints] @ membership
    target = current[query, history] @ previous[history][:, context_keys]
    contexts = previous[endpoints][:, context_keys].copy()
    # No self-edge context: an endpoint cannot establish its own ownership.
    contexts *= positions[context_keys][None, :] < endpoints[:, None]
    norms = np.linalg.norm(contexts, axis=1)
    target_norm = np.linalg.norm(target)
    target_mass = target.sum()
    normalized = np.divide(
        contexts, norms[:, None], out=np.zeros_like(contexts), where=norms[:, None] > 0
    )
    target = target / target_norm if target_norm else target
    compatible = normalized @ target
    permutation = np.random.default_rng(seed).permutation(len(endpoints))
    shuffled = normalized[permutation] @ target
    graph = (current[query, endpoints] * compatible) @ membership
    null = (current[query, endpoints] * shuffled) @ membership
    supports = {
        "context": graph,
        "direct": direct,
        "shuffled": null,
        "count": counts.astype(float),
    }
    coverage = bool(counts[0] > 0 and (counts > 0).sum() >= 2)
    # Unknown endpoint context is not negative evidence for its candidate.
    known = (norms > 0).astype(int) @ membership
    context_complete = bool(np.all(known[counts > 0] == counts[counts > 0]))
    validity = {
        "context": bool(
            coverage and context_complete and target_norm > 0 and graph.sum() > 0
        ),
        "shuffled": bool(
            coverage and context_complete and target_norm > 0 and null.sum() > 0
        ),
        "direct": bool(coverage and direct.sum() > 0),
        "count": coverage,
    }
    result = {
        "valid": all(validity.values()),
        "permutation_seed": seed,
        "endpoint_permutation": permutation.tolist(),
        "permuted_endpoint_fraction": float(
            np.mean(permutation != np.arange(len(endpoints)))
        )
        if len(endpoints)
        else None,
        "permuted_candidate_fraction": float(
            np.mean(ids[endpoints][permutation] != ids[endpoints])
        )
        if len(endpoints)
        else None,
        "candidate_context_coverage": np.divide(
            known, counts, out=np.zeros(len(counts)), where=counts > 0
        ).tolist(),
        "covered_candidates": int((counts > 0).sum()),
        "candidate_count": len(candidates),
        "top_candidate_covered": bool(counts[0] > 0),
        "history_source_mass": float(target_mass),
        "candidate_source_mass": float(direct.sum()),
        "endpoint_context_coverage": float(np.mean(norms > 0)) if len(norms) else 0.0,
    }
    for name, values in supports.items():
        mass = float(values.sum())
        distribution = values / mass if mass else np.zeros_like(values)
        result[f"{name}_support"] = distribution.tolist()
        result[f"{name}_valid"] = validity[name]
        result[f"{name}_risk"] = float(1 - distribution[0]) if validity[name] else None
    return result
