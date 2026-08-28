"""Position-preserving attention interventions for mechanism auditing.

The seven variants are deliberately fixed.  Evidence removal blocks evidence
key columns for every strictly later query, including prompt-internal relays.
History removal blocks only earlier response keys from response queries.
Three independently selected evidence donors make swap sensitivity less
dependent on one arbitrary source.  Swapping changes token IDs only at the
target's evidence positions and never pads, repeats, or silently resamples.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Sequence
from typing import Mapping

import numpy as np

from .roles import EVIDENCE, ROLE_NAMES, PromptRoleMap


SWAPPED_EVIDENCE_NAMES = tuple(f"swapped_evidence_{index}" for index in range(3))
COUNTERFACTUAL_NAMES = (
    "full",
    "no_evidence",
    "no_history",
    "no_evidence_no_history",
    *SWAPPED_EVIDENCE_NAMES,
)


@dataclass(frozen=True)
class CounterfactualVariant:
    """One frozen-sequence intervention consumed by model replay."""

    name: str
    prompt_length: int
    token_ids: np.ndarray | None
    allowed_attention: np.ndarray | None
    available: bool = True
    unavailable_reason: str | None = None
    donor_source_id: str | None = None

    @property
    def attention_mask(self) -> np.ndarray | None:
        """Alias with explicit ``True = allowed`` semantics."""

        return self.allowed_attention

    @property
    def response_length(self) -> int | None:
        if self.token_ids is None:
            return None
        return int(np.asarray(self.token_ids).size - self.prompt_length)

    def validate(self) -> "CounterfactualVariant":
        if self.name not in COUNTERFACTUAL_NAMES:
            raise ValueError(f"unknown counterfactual variant: {self.name}")
        if not self.available:
            if self.token_ids is not None or self.allowed_attention is not None:
                raise ValueError("an unavailable variant must not expose runnable arrays")
            if not self.unavailable_reason:
                raise ValueError("an unavailable variant must state a reason")
            return self
        tokens = np.asarray(self.token_ids)
        allowed = np.asarray(self.allowed_attention)
        if (
            tokens.ndim != 1
            or tokens.dtype.kind not in "iu"
            or not 0 < self.prompt_length < tokens.size
        ):
            raise ValueError("variant token_ids must contain prompt and response")
        if allowed.shape != (tokens.size, tokens.size) or allowed.dtype != np.bool_:
            raise ValueError("allowed_attention must be a boolean [N,N] matrix")
        causal = np.arange(tokens.size)[:, None] >= np.arange(tokens.size)[None, :]
        if bool((allowed & ~causal).any()):
            raise ValueError("counterfactual attention cannot expose future keys")
        if bool((~allowed.any(axis=1)).any()):
            raise ValueError("every query must retain at least one causal source")
        return self


def causal_attention(token_count: int) -> np.ndarray:
    """Return the inclusive causal support, with ``True`` meaning allowed."""

    token_count = int(token_count)
    if token_count < 2:
        raise ValueError("token_count must include prompt and response")
    position = np.arange(token_count)
    return position[:, None] >= position[None, :]


def intervention_attention_masks(
    prompt_roles,
    response_length: int,
) -> dict[str, np.ndarray]:
    """Construct the four attention masks used by all runnable branches."""

    roles = np.asarray(prompt_roles)
    response_length = int(response_length)
    if (
        roles.ndim != 1
        or roles.dtype.kind not in "iu"
        or roles.size < 1
        or bool(((roles < 0) | (roles >= len(ROLE_NAMES))).any())
        or response_length < 1
    ):
        raise ValueError("prompt_roles and response_length are invalid")
    prompt_length = roles.size
    token_count = prompt_length + response_length
    full = causal_attention(token_count)

    no_evidence = full.copy()
    evidence = np.flatnonzero(roles == EVIDENCE)
    for source in evidence:
        # The evidence token may form its own state, but no later query may use
        # it as a key.  Keeping the diagonal also guarantees a valid causal row
        # when evidence begins at absolute position zero.
        no_evidence[source + 1 :, source] = False

    no_history = full.copy()
    for query in range(prompt_length, token_count):
        no_history[query, prompt_length:query] = False

    joint = no_evidence & no_history
    for name, allowed in {
        "full": full,
        "no_evidence": no_evidence,
        "no_history": no_history,
        "no_evidence_no_history": joint,
    }.items():
        if bool((~allowed.any(axis=1)).any()):
            raise ValueError(f"{name} leaves a query without any causal source")
    return {
        "full": full,
        "no_evidence": no_evidence,
        "no_history": no_history,
        "no_evidence_no_history": joint,
    }


build_attention_masks = intervention_attention_masks


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(indices[1:] != indices[:-1] + 1) + 1
    groups = np.split(indices, breaks)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def swap_evidence_tokens(
    target_token_ids,
    target_roles: PromptRoleMap,
    donor_roles: PromptRoleMap,
) -> np.ndarray:
    """Replace one target evidence span with an exact-length donor prefix.

    The donor must have a distinct source, the same task, and a contiguous
    evidence run at least as long as the target run.  The first eligible donor
    run in prompt order is cropped at its right edge to the exact target length.
    No token outside the target evidence run is changed.
    """

    target_roles.validate(target_token_ids)
    donor_roles.validate()
    if donor_roles.prompt_token_ids is None:
        raise ValueError("donor prompt token IDs are unavailable")
    if target_roles.source_id == donor_roles.source_id:
        raise ValueError("evidence donor must have a different source_id")
    if target_roles.task_type.casefold() != donor_roles.task_type.casefold():
        raise ValueError("evidence donor must have the same task type")

    target = np.asarray(target_token_ids)
    if target.ndim != 1 or target.dtype.kind not in "iu":
        raise ValueError("target_token_ids must be a one-dimensional integer array")
    target_positions = np.flatnonzero(target_roles.role_mask(EVIDENCE))
    target_runs = _contiguous_runs(target_positions)
    if len(target_runs) != 1:
        raise ValueError("target evidence must form exactly one contiguous span")
    target_start, target_end = target_runs[0]
    required = target_end - target_start

    donor_positions = np.flatnonzero(donor_roles.role_mask(EVIDENCE))
    donor_runs = _contiguous_runs(donor_positions)
    eligible = [run for run in donor_runs if run[1] - run[0] >= required]
    if not eligible:
        raise ValueError("donor has no evidence span long enough for the target")
    donor_start, _ = eligible[0]
    donor = np.asarray(donor_roles.prompt_token_ids)
    replacement = donor[donor_start : donor_start + required]
    if replacement.size != required:
        raise ValueError("donor evidence crop is not the exact target length")

    swapped = target.copy()
    swapped[target_start:target_end] = replacement.astype(target.dtype, copy=False)
    if swapped.shape != target.shape:
        raise AssertionError("evidence swap changed sequence length")
    if not np.array_equal(
        swapped[target_roles.response_idx :], target[target_roles.response_idx :]
    ):
        raise AssertionError("evidence swap changed factual response targets")
    unchanged = np.ones(target.size, dtype=np.bool_)
    unchanged[target_start:target_end] = False
    if not np.array_equal(swapped[unchanged], target[unchanged]):
        raise AssertionError("evidence swap changed a non-evidence position")
    return swapped


build_swapped_token_ids = swap_evidence_tokens


def build_counterfactual_variants(
    token_ids,
    prompt_roles: PromptRoleMap,
    donor_roles: PromptRoleMap | Sequence[PromptRoleMap] | None,
) -> dict[str, CounterfactualVariant]:
    """Build the exact seven registered variants for a factual response.

    A single donor remains accepted for caller compatibility and occupies slot
    zero.  Missing, duplicate, incompatible, or too-short donor slots are
    explicitly unavailable.  They never fall back to padding, repetition, or
    reuse of another donor.
    """

    tokens = np.asarray(token_ids)
    prompt_roles.validate(tokens)
    response_length = tokens.size - prompt_roles.response_idx
    masks = intervention_attention_masks(prompt_roles.role_ids, response_length)
    result: dict[str, CounterfactualVariant] = {}
    for name in COUNTERFACTUAL_NAMES[:4]:
        result[name] = CounterfactualVariant(
            name=name,
            prompt_length=prompt_roles.response_idx,
            token_ids=tokens.copy(),
            allowed_attention=masks[name],
        ).validate()

    if donor_roles is None:
        donors: tuple[PromptRoleMap, ...] = ()
    elif isinstance(donor_roles, PromptRoleMap):
        donors = (donor_roles,)
    else:
        donors = tuple(donor_roles)
        if len(donors) > len(SWAPPED_EVIDENCE_NAMES):
            raise ValueError("at most three evidence donors may be supplied")
        if not all(isinstance(donor, PromptRoleMap) for donor in donors):
            raise TypeError("evidence donors must be PromptRoleMap instances")

    seen_sources: set[str] = set()
    for slot, name in enumerate(SWAPPED_EVIDENCE_NAMES):
        donor = donors[slot] if slot < len(donors) else None
        if donor is None:
            swapped = CounterfactualVariant(
                name=name,
                prompt_length=prompt_roles.response_idx,
                token_ids=None,
                allowed_attention=None,
                available=False,
                unavailable_reason=f"no evidence donor was provided for slot {slot}",
            )
        elif donor.source_id in seen_sources:
            swapped = CounterfactualVariant(
                name=name,
                prompt_length=prompt_roles.response_idx,
                token_ids=None,
                allowed_attention=None,
                available=False,
                unavailable_reason="duplicate evidence donor source_id",
                donor_source_id=donor.source_id,
            )
        else:
            seen_sources.add(donor.source_id)
            try:
                swapped_tokens = swap_evidence_tokens(tokens, prompt_roles, donor)
            except ValueError as error:
                swapped = CounterfactualVariant(
                    name=name,
                    prompt_length=prompt_roles.response_idx,
                    token_ids=None,
                    allowed_attention=None,
                    available=False,
                    unavailable_reason=str(error),
                    donor_source_id=donor.source_id,
                )
            else:
                swapped = CounterfactualVariant(
                    name=name,
                    prompt_length=prompt_roles.response_idx,
                    token_ids=swapped_tokens,
                    allowed_attention=masks["full"].copy(),
                    donor_source_id=donor.source_id,
                )
        result[name] = swapped.validate()
    if tuple(result) != COUNTERFACTUAL_NAMES:
        raise AssertionError("counterfactual registry order changed")
    return result


def _donor_rank(target: PromptRoleMap, candidate: PromptRoleMap) -> bytes:
    payload = (
        target.source_id
        + candidate.source_id
        + candidate.prompt_token_sha256
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def choose_donors(
    target: PromptRoleMap,
    candidates: Mapping[str, PromptRoleMap],
    count: int = 3,
) -> tuple[PromptRoleMap, ...]:
    """Choose up to ``count`` compatible donors by a stable target-specific hash."""

    target.validate()
    count = int(count)
    if count < 1:
        raise ValueError("donor count must be positive")
    target_count = int(target.role_mask(EVIDENCE).sum())
    if target_count == 0:
        return ()
    eligible: dict[str, PromptRoleMap] = {}
    for candidate in candidates.values():
        candidate.validate()
        if candidate.source_id == target.source_id:
            continue
        if candidate.task_type.casefold() != target.task_type.casefold():
            continue
        if candidate.prompt_token_ids is None:
            continue
        if any(
            end - start >= target_count
            for start, end in _contiguous_runs(
                np.flatnonzero(candidate.role_mask(EVIDENCE))
            )
        ):
            existing = eligible.get(candidate.source_id)
            if existing is not None and (
                existing.prompt_token_sha256 != candidate.prompt_token_sha256
            ):
                raise ValueError("candidate source_id has conflicting prompt hashes")
            eligible[candidate.source_id] = candidate
    ordered = sorted(
        eligible.values(),
        key=lambda candidate: (
            _donor_rank(target, candidate),
            candidate.source_id,
        ),
    )
    return tuple(ordered[:count])


def choose_donor(
    target: PromptRoleMap,
    candidates: Mapping[str, PromptRoleMap],
) -> PromptRoleMap | None:
    """Compatibility wrapper; new code should persist the full donor ensemble."""

    selected = choose_donors(target, candidates, count=1)
    return selected[0] if selected else None


__all__ = [
    "COUNTERFACTUAL_NAMES",
    "SWAPPED_EVIDENCE_NAMES",
    "CounterfactualVariant",
    "build_counterfactual_variants",
    "choose_donor",
    "choose_donors",
    "intervention_attention_masks",
    "swap_evidence_tokens",
]
