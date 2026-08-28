import hashlib

import numpy as np
import pytest

from experiments.attention_mechanism_audit.counterfactuals import (
    COUNTERFACTUAL_NAMES,
    SWAPPED_EVIDENCE_NAMES,
    build_counterfactual_variants,
    choose_donors,
    intervention_attention_masks,
    swap_evidence_tokens,
)
from experiments.attention_mechanism_audit.roles import (
    CONSTRAINT,
    EVIDENCE,
    OTHER_PROMPT,
    QUESTION,
)
from experiments.attention_mechanism_audit.tests.helpers import make_role_map


def test_counterfactual_masks_preserve_causality_and_exact_scope():
    roles = np.asarray([OTHER_PROMPT, EVIDENCE, CONSTRAINT, QUESTION])
    masks = intervention_attention_masks(roles, response_length=3)
    full = masks["full"]
    no_evidence = masks["no_evidence"]
    no_history = masks["no_history"]

    position = np.arange(7)
    np.testing.assert_array_equal(full, position[:, None] >= position[None, :])
    assert no_evidence[1, 1]
    assert not no_evidence[2:, 1].any()
    np.testing.assert_array_equal(no_evidence[:, [0, 2, 3, 4, 5, 6]], full[:, [0, 2, 3, 4, 5, 6]])
    assert no_history[:4].tolist() == full[:4].tolist()
    assert not no_history[5, 4]
    assert not no_history[6, 4:6].any()
    assert no_history[4, 4] and no_history[5, 5] and no_history[6, 6]
    np.testing.assert_array_equal(
        masks["no_evidence_no_history"], no_evidence & no_history
    )
    assert all(mask.any(axis=1).all() for mask in masks.values())


def test_counterfactual_masks_reject_unknown_prompt_roles():
    with pytest.raises(ValueError, match="prompt_roles"):
        intervention_attention_masks([OTHER_PROMPT, 9], response_length=2)


def test_swap_replaces_only_evidence_and_preserves_factual_response():
    target_prompt = np.asarray([10, 11, 12, 13, 14, 15])
    target_roles = make_role_map(
        target_prompt,
        [OTHER_PROMPT, CONSTRAINT, EVIDENCE, EVIDENCE, QUESTION, CONSTRAINT],
        source_id="target",
    )
    donor_prompt = np.asarray([20, 21, 22, 23, 24, 25, 26, 27])
    donor_roles = make_role_map(
        donor_prompt,
        [OTHER_PROMPT, CONSTRAINT, EVIDENCE, EVIDENCE, EVIDENCE, EVIDENCE, QUESTION, CONSTRAINT],
        source_id="donor",
    )
    factual = np.concatenate((target_prompt, [91, 92, 93]))

    swapped = swap_evidence_tokens(factual, target_roles, donor_roles)

    np.testing.assert_array_equal(swapped, [10, 11, 22, 23, 14, 15, 91, 92, 93])
    np.testing.assert_array_equal(swapped[target_roles.response_idx :], [91, 92, 93])
    assert swapped.shape == factual.shape


def test_build_variants_has_exact_registry_and_shared_factual_targets():
    prompt = np.asarray([10, 11, 12, 13])
    target = make_role_map(
        prompt,
        [OTHER_PROMPT, EVIDENCE, EVIDENCE, CONSTRAINT],
        source_id="target",
    )
    donors = [
        make_role_map(
            [
                20 + 10 * index,
                21 + 10 * index,
                22 + 10 * index,
                23 + 10 * index,
                24 + 10 * index,
            ],
            [OTHER_PROMPT, EVIDENCE, EVIDENCE, EVIDENCE, CONSTRAINT],
            source_id=f"donor-{index}",
        )
        for index in range(3)
    ]
    factual = np.asarray([10, 11, 12, 13, 90, 91])

    variants = build_counterfactual_variants(factual, target, donors)

    assert tuple(variants) == COUNTERFACTUAL_NAMES
    assert tuple(variants)[4:] == SWAPPED_EVIDENCE_NAMES
    for variant in variants.values():
        assert variant.available
        np.testing.assert_array_equal(variant.token_ids[4:], [90, 91])
        assert variant.prompt_length == 4


def test_short_or_incompatible_donor_is_explicitly_unavailable():
    target = make_role_map(
        [10, 11, 12, 13],
        [OTHER_PROMPT, EVIDENCE, EVIDENCE, CONSTRAINT],
        source_id="target",
    )
    short = make_role_map(
        [20, 21, 22],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
        source_id="short",
    )
    factual = np.asarray([10, 11, 12, 13, 90])

    variants = build_counterfactual_variants(factual, target, short)
    variant = variants["swapped_evidence_0"]
    assert not variant.available
    assert variant.token_ids is None and variant.allowed_attention is None
    assert "long enough" in variant.unavailable_reason
    assert not variants["swapped_evidence_1"].available
    assert not variants["swapped_evidence_2"].available

    same_source = make_role_map(
        [20, 21, 22, 23],
        [OTHER_PROMPT, EVIDENCE, EVIDENCE, CONSTRAINT],
        source_id="target",
    )
    variant = build_counterfactual_variants(factual, target, same_source)[
        "swapped_evidence_0"
    ]
    assert not variant.available
    assert "different source_id" in variant.unavailable_reason


def test_swap_rejects_noncontiguous_target_evidence_instead_of_resampling():
    target = make_role_map(
        [10, 11, 12, 13, 14],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT, EVIDENCE, CONSTRAINT],
        source_id="target",
    )
    donor = make_role_map(
        [20, 21, 22, 23],
        [OTHER_PROMPT, EVIDENCE, EVIDENCE, CONSTRAINT],
        source_id="donor",
    )

    with pytest.raises(ValueError, match="one contiguous span"):
        swap_evidence_tokens([10, 11, 12, 13, 14, 90], target, donor)


def test_donor_ensemble_is_target_hash_ranked_and_task_matched():
    target = make_role_map(
        [1, 2, 3],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
        source_id="target",
    )
    compatible = [
        make_role_map(
            [10 * index + 4, 10 * index + 5, 10 * index + 6],
            [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
            source_id=source_id,
        )
        for index, source_id in enumerate(("zulu", "alpha", "beta", "gamma"))
    ]
    summary = make_role_map(
        [7, 8, 9],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
        source_id="aardvark",
        task_type="Summary",
    )

    candidates = {item.source_id: item for item in (*compatible, summary)}
    expected = sorted(
        compatible,
        key=lambda candidate: (
            hashlib.sha256(
                (
                    target.source_id
                    + candidate.source_id
                    + candidate.prompt_token_sha256
                ).encode("utf-8")
            ).digest(),
            candidate.source_id,
        ),
    )[:3]
    selected = choose_donors(target, candidates)
    assert [item.source_id for item in selected] == [
        item.source_id for item in expected
    ]
    assert choose_donors(target, candidates) == selected
    assert len(choose_donors(target, candidates, count=2)) == 2


def test_single_donor_is_slot_zero_and_missing_slots_are_not_reused():
    target = make_role_map(
        [10, 11, 12],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
        source_id="target",
    )
    donor = make_role_map(
        [20, 21, 22],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
        source_id="donor",
    )
    variants = build_counterfactual_variants(
        [10, 11, 12, 90], target, donor
    )
    assert variants["swapped_evidence_0"].available
    assert variants["swapped_evidence_0"].donor_source_id == "donor"
    assert not variants["swapped_evidence_1"].available
    assert not variants["swapped_evidence_2"].available
    assert variants["swapped_evidence_1"].donor_source_id is None


def test_duplicate_donor_source_is_explicitly_unavailable():
    target = make_role_map(
        [10, 11, 12],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
        source_id="target",
    )
    donor = make_role_map(
        [20, 21, 22],
        [OTHER_PROMPT, EVIDENCE, CONSTRAINT],
        source_id="donor",
    )
    variants = build_counterfactual_variants(
        [10, 11, 12, 90], target, [donor, donor]
    )
    assert variants["swapped_evidence_0"].available
    duplicate = variants["swapped_evidence_1"]
    assert not duplicate.available
    assert "duplicate" in duplicate.unavailable_reason
