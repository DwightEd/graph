"""Independent hallucination-mechanism trajectories.

The audit reports grounding drift, routing/functional dispersion and
counterfactual evidence bypass as separate axes.  This module never constructs
a hand-weighted total hallucination score.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .counterfactuals import SWAPPED_EVIDENCE_NAMES


EPSILON = 1e-12


def _finite_mean(value: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(value)
    count = finite.sum(axis=axis)
    total = np.where(finite, value, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=np.float64),
        where=count > 0,
    )


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _variant(counterfactual: Any, *names: str) -> Any | None:
    variants = _member(counterfactual, "variants")
    if variants is not None:
        counterfactual = variants
    for name in names:
        found = _member(counterfactual, name)
        if found is not None:
            return found
    return None


def _metric(variant: Any | None, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if variant is None:
        return np.full(shape, np.nan, dtype=np.float64)
    if not bool(_member(variant, "available", True)):
        return np.full(shape, np.nan, dtype=np.float64)
    aliases = {
        "margin": ("margin", "chosen_vs_best_other_margin"),
        "jsd_from_full": ("jsd_from_full", "vocab_jsd_from_full"),
    }.get(name, (name,))
    value = None
    for alias in aliases:
        value = _member(variant, alias)
        if value is not None:
            break
    if value is None:
        return np.full(shape, np.nan, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(
            f"counterfactual {name} must have shape {shape}, got {array.shape}"
        )
    return array


def _ensemble_metric(
    counterfactual: Any,
    name: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite-only donor mean, population std and available count."""

    values = np.stack(
        [
            _metric(_variant(counterfactual, variant_name), name, shape)
            for variant_name in SWAPPED_EVIDENCE_NAMES
        ],
        axis=0,
    )
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        count,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    square_error = np.where(finite, (values - mean[None]) ** 2, 0.0).sum(axis=0)
    variance = np.divide(
        square_error,
        count,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    return mean, np.sqrt(variance), count.astype(np.int64)


def _counterfactual_trajectories(counterfactual: Any) -> dict[str, np.ndarray]:
    full = _variant(counterfactual, "full", "baseline")
    if full is None:
        raise ValueError("counterfactual audit requires a full baseline replay")
    full_logp_value = _member(full, "chosen_logprob")
    if full_logp_value is None:
        raise ValueError("full replay does not provide chosen_logprob")
    full_logp = np.asarray(full_logp_value, dtype=np.float64)
    if full_logp.ndim != 1:
        raise ValueError("counterfactual chosen_logprob must be a token trajectory")
    shape = full_logp.shape

    no_evidence = _variant(counterfactual, "no_evidence", "evidence_removed")
    no_history = _variant(counterfactual, "no_history", "history_removed")
    joint = _variant(
        counterfactual,
        "joint_no_evidence_no_history",
        "no_evidence_no_history",
        "joint_removed",
    )

    no_evidence_logp = _metric(no_evidence, "chosen_logprob", shape)
    swapped_logp, swapped_donor_std, swapped_available_count = _ensemble_metric(
        counterfactual, "chosen_logprob", shape
    )
    no_history_logp = _metric(no_history, "chosen_logprob", shape)
    joint_logp = _metric(joint, "chosen_logprob", shape)
    no_evidence_delta = no_evidence_logp - full_logp
    swapped_delta = swapped_logp - full_logp
    no_history_delta = no_history_logp - full_logp
    evidence_bypass = np.where(
        np.isfinite(no_evidence_delta) & np.isfinite(swapped_delta),
        0.5 * (no_evidence_delta + swapped_delta),
        np.nan,
    )

    full_margin = _metric(full, "margin", shape)
    no_evidence_margin = _metric(no_evidence, "margin", shape)
    swapped_margin, _, _ = _ensemble_metric(counterfactual, "margin", shape)
    swapped_jsd, _, _ = _ensemble_metric(
        counterfactual, "jsd_from_full", shape
    )
    no_history_margin = _metric(no_history, "margin", shape)

    return {
        "counterfactual_full_chosen_logprob": full_logp,
        "counterfactual_no_evidence_delta": no_evidence_delta,
        "counterfactual_swapped_evidence_delta": swapped_delta,
        "counterfactual_swapped_evidence_donor_std": swapped_donor_std,
        "counterfactual_swapped_evidence_available_count": swapped_available_count,
        "counterfactual_evidence_bypass": evidence_bypass,
        "counterfactual_no_history_delta": no_history_delta,
        "counterfactual_history_necessity": full_logp - no_history_logp,
        "counterfactual_evidence_history_interaction": (
            joint_logp - no_evidence_logp - no_history_logp + full_logp
        ),
        "counterfactual_no_evidence_margin_delta": no_evidence_margin - full_margin,
        "counterfactual_swapped_evidence_margin_delta": swapped_margin - full_margin,
        "counterfactual_no_history_margin_delta": no_history_margin - full_margin,
        "counterfactual_no_evidence_jsd_from_full": _metric(
            no_evidence, "jsd_from_full", shape
        ),
        "counterfactual_swapped_evidence_jsd_from_full": swapped_jsd,
        "counterfactual_no_history_jsd_from_full": _metric(
            no_history, "jsd_from_full", shape
        ),
    }


def combine_mechanisms(
    functional: Mapping[str, np.ndarray],
    routing: Mapping[str, np.ndarray],
    counterfactual: Any | None = None,
) -> dict[str, np.ndarray]:
    """Create preregistered raw trajectories without a weighted total score."""

    role_energy = np.asarray(
        functional["functional_absolute_layer_role"], dtype=np.float64
    )
    if role_energy.ndim != 3 or role_energy.shape[-1] < 2:
        raise ValueError("functional role energy must have shape [R,L,roles]")
    prompt_energy = role_energy[..., :-1].sum(axis=-1)
    grounding_energy = role_energy[..., :3].sum(axis=-1)
    other_prompt_energy = role_energy[..., 3]
    history_energy = role_energy[..., -1]
    total_energy = prompt_energy + history_energy
    evidence_energy = role_energy[..., 0]

    entropy = np.asarray(functional["functional_entropy_observed"], dtype=np.float64)
    hhi = np.asarray(functional["functional_hhi_observed"], dtype=np.float64)
    known_coverage = np.asarray(
        functional["functional_known_attention_coverage"], dtype=np.float64
    )
    grounding_drift = np.log(
        (history_energy + EPSILON) / (grounding_energy + EPSILON)
    )
    grounding_drift = np.where(total_energy > 0, grounding_drift, np.nan)
    output: dict[str, np.ndarray] = {
        "drift_functional_history_to_grounding_log_ratio": grounding_drift,
        "drift_functional_prompt_fraction": np.divide(
            prompt_energy,
            total_energy,
            out=np.full_like(total_energy, np.nan),
            where=total_energy > 0,
        ),
        "drift_functional_evidence_fraction": np.divide(
            evidence_energy,
            total_energy,
            out=np.full_like(total_energy, np.nan),
            where=total_energy > 0,
        ),
        "drift_functional_other_prompt_fraction": np.divide(
            other_prompt_energy,
            total_energy,
            out=np.full_like(total_energy, np.nan),
            where=total_energy > 0,
        ),
        "dispersion_functional_entropy_observed": _finite_mean(entropy, axis=2),
        "dispersion_functional_hhi_observed": _finite_mean(hhi, axis=2),
        "dispersion_functional_head_role_js": np.asarray(
            functional["functional_head_role_js"], dtype=np.float64
        ),
        "dispersion_functional_cancellation": np.asarray(
            functional["functional_cancellation"], dtype=np.float64
        ),
        "functional_known_attention_coverage": _finite_mean(
            known_coverage, axis=2
        ),
        "routing_entropy_lower": np.asarray(
            routing["routing_entropy_bounds"], dtype=np.float64
        )[..., 0],
        "routing_entropy_upper": np.asarray(
            routing["routing_entropy_bounds"], dtype=np.float64
        )[..., 1],
        "routing_concentration_lower": np.asarray(
            routing["routing_concentration_bounds"], dtype=np.float64
        )[..., 0],
        "routing_concentration_upper": np.asarray(
            routing["routing_concentration_bounds"], dtype=np.float64
        )[..., 1],
        "routing_head_role_js": np.asarray(
            routing["routing_head_role_js"], dtype=np.float64
        ),
        "routing_direct_evidence_ancestry": np.asarray(
            routing["routing_direct_role_ancestry"], dtype=np.float64
        )[..., 0],
        "routing_relayed_evidence_ancestry": np.asarray(
            routing["routing_relayed_role_ancestry"], dtype=np.float64
        )[..., 0],
        "routing_total_evidence_ancestry": np.asarray(
            routing["routing_grounded_role_ancestry"], dtype=np.float64
        )[..., 0],
        "routing_ungrounded_history_ancestry": np.asarray(
            routing["routing_ungrounded_history_ancestry"], dtype=np.float64
        ),
        "routing_unresolved_ancestry": np.asarray(
            routing["routing_unresolved_ancestry"], dtype=np.float64
        ),
    }
    signed_role = np.asarray(
        functional["functional_signed_layer_role"], dtype=np.float64
    )
    signed_role_se = np.asarray(
        functional.get(
            "functional_signed_layer_role_estimator_se",
            np.full_like(signed_role, np.nan),
        ),
        dtype=np.float64,
    )
    functional_names = tuple(
        np.asarray(functional["functional_role_names"]).astype(str).tolist()
    )
    if (
        signed_role.shape != role_energy.shape
        or signed_role_se.shape != role_energy.shape
        or len(functional_names) != role_energy.shape[-1]
    ):
        raise ValueError("functional role names and layer trajectories are misaligned")
    for index, name in enumerate(functional_names):
        output[f"functional_signed_{name}"] = signed_role[..., index]
        output[f"functional_signed_{name}_estimator_se"] = signed_role_se[..., index]
        output[f"functional_absolute_{name}"] = role_energy[..., index]

    routing_role_mass = np.asarray(
        routing["routing_mean_fine_role_mass"], dtype=np.float64
    )
    routing_names = tuple(
        np.asarray(routing["routing_role_names"]).astype(str).tolist()
    )
    if routing_role_mass.ndim != 3 or len(routing_names) != routing_role_mass.shape[-1]:
        raise ValueError("routing role names and mass trajectories are misaligned")
    for index, name in enumerate(routing_names):
        output[f"routing_mean_mass_{name}"] = routing_role_mass[..., index]
    if counterfactual is not None:
        output.update(_counterfactual_trajectories(counterfactual))

    forbidden = {"total_score", "hallucination_score", "weighted_total"}
    if forbidden.intersection(output):
        raise AssertionError("mechanism axes must not be collapsed into a weighted total")
    return output


__all__ = ["combine_mechanisms"]
