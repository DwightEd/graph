import copy

import numpy as np

from experiments.attention_mechanism_audit.detect import (
    _fit_model,
    _record_weights,
    _raw_scores,
    crossfit_partitions,
    causal_auxiliary_channels,
    factorial_contrasts,
    mechanism_tensor,
    relative_positions,
    score_records,
    source_fold_assignments,
)


def _artifact(seed: int, tokens: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    layers, heads, roles = 2, 3, 4
    base = rng.lognormal(0.0, 0.25, (layers, tokens, heads, roles))
    full = rng.normal(-2.0, 0.2, tokens)
    evidence = rng.normal(0.2, 0.05, tokens)
    history = rng.normal(0.1, 0.05, tokens)
    interaction = rng.normal(0.0, 0.02, tokens)
    no_evidence = full - evidence + 0.5 * interaction
    no_history = full - history + 0.5 * interaction
    neither = full - evidence - history
    full_margin = rng.normal(0.0, 0.3, tokens)
    return {
        "response_start": 5,
        "trace": {
            "role_attention_mass": rng.dirichlet(
                np.ones(roles), size=(layers, tokens, heads)
            ),
            "edge_role_energy": base,
            "head_role_write_norm": base * rng.lognormal(0.0, 0.1, base.shape),
            "head_source_entropy": rng.uniform(0.1, 1.0, (layers, tokens, heads)),
            "role_head_coherence": rng.uniform(0.2, 0.9, (layers, tokens, roles)),
        },
        "score_inputs": {
            "full_logprob": full,
            "no_evidence_logprob": no_evidence,
            "no_history_logprob": no_history,
            "no_evidence_history_logprob": neither,
            "full_margin": full_margin,
            "no_evidence_history_margin": full_margin
            + rng.normal(0.05, 0.1, tokens),
        },
    }


def _records(count: int = 10) -> list[dict]:
    return [
        {
            "sample_id": f"sample-{index}",
            "source_id": f"source-{index}",
            "task_type": "QA",
            "artifact": _artifact(index),
        }
        for index in range(count)
    ]


def test_factorial_contrasts_cancel_common_branch_shift():
    artifact = _artifact(3)
    expected = factorial_contrasts(artifact)
    shifted = copy.deepcopy(artifact)
    for name in shifted["score_inputs"]:
        shifted["score_inputs"][name] += 17.0
    np.testing.assert_allclose(factorial_contrasts(shifted), expected, atol=1e-12)


def test_four_causal_auxiliary_channels_keep_absolute_remaining_context_margin():
    artifact = _artifact(3)
    expected = causal_auxiliary_channels(artifact)
    assert expected.shape == (7, 4)
    np.testing.assert_allclose(
        expected[:, 3],
        artifact["score_inputs"]["no_evidence_history_margin"],
    )
    shifted = copy.deepcopy(artifact)
    for name in (
        "full_logprob",
        "no_evidence_logprob",
        "no_history_logprob",
        "no_evidence_history_logprob",
    ):
        shifted["score_inputs"][name] += 17.0
    np.testing.assert_allclose(causal_auxiliary_channels(shifted), expected)

    shifted["score_inputs"]["full_margin"] -= 9.0
    shifted["score_inputs"]["no_evidence_history_margin"] -= 9.0
    shifted_channels = causal_auxiliary_channels(shifted)
    np.testing.assert_allclose(shifted_channels[:, :3], expected[:, :3])
    np.testing.assert_allclose(shifted_channels[:, 3], expected[:, 3] - 9.0)


def test_factorial_contrasts_match_symmetric_effect_equations():
    artifact = {
        "score_inputs": {
            "full_logprob": np.asarray([10.0]),
            "no_evidence_logprob": np.asarray([7.0]),
            "no_history_logprob": np.asarray([8.0]),
            "no_evidence_history_logprob": np.asarray([4.0]),
        }
    }
    np.testing.assert_allclose(factorial_contrasts(artifact), [[3.5, 2.5, -1.0]])


def test_mechanism_tensor_contains_attention_clr_without_renormalizing_entropy():
    artifact = _artifact(5)
    tensor, coherence = mechanism_tensor(artifact)
    assert tensor.shape == (7, 2, 3, 13)
    assert coherence.shape == (7, 2, 4)
    np.testing.assert_allclose(tensor[..., :4].sum(axis=-1), 0.0, atol=1e-12)
    expected_entropy = np.moveaxis(
        artifact["trace"]["head_source_entropy"], 1, 0
    )
    np.testing.assert_allclose(tensor[..., -1], expected_entropy)

    changed = copy.deepcopy(artifact)
    changed["trace"]["role_attention_mass"][..., 0] *= 2.0
    changed["trace"]["role_attention_mass"] /= changed["trace"][
        "role_attention_mass"
    ].sum(axis=-1, keepdims=True)
    changed_tensor, _ = mechanism_tensor(changed)
    assert not np.array_equal(changed_tensor, tensor)


def test_relative_position_coordinate_is_shared_and_centered_in_each_bin():
    np.testing.assert_allclose(relative_positions(1), [0.5])
    np.testing.assert_allclose(relative_positions(4), [0.125, 0.375, 0.625, 0.875])


def test_crossfit_partitions_are_source_disjoint_and_deterministic():
    sources = [f"source-{index}" for index in range(17)]
    first = source_fold_assignments(sources, folds=5, seed=19)
    second = source_fold_assignments(list(reversed(sources)), folds=5, seed=19)
    assert first == second
    seen_test_sources = []
    for partition in crossfit_partitions(sources, folds=5, seed=19):
        fit = set(partition["fit_sources"])
        calibration = set(partition["calibration_sources"])
        test = set(partition["test_sources"])
        assert not (fit & calibration or fit & test or calibration & test)
        assert fit | calibration | test == set(sources)
        seen_test_sources.extend(test)
    assert sorted(seen_test_sources) == sorted(sources)


def test_record_weights_sum_to_one_per_source_even_with_repeated_records():
    records = _records(3)
    repeated = copy.deepcopy(records[0])
    repeated["sample_id"] = "sample-0-copy"
    records.append(repeated)
    weights = _record_weights(records)
    totals = {}
    for record in records:
        source = record["source_id"]
        totals[source] = totals.get(source, 0.0) + weights[id(record)]
    assert totals == {"source-0": 1.0, "source-1": 1.0, "source-2": 1.0}


def test_scores_are_deterministic_and_labels_are_never_consumed():
    records = _records()
    kwargs = dict(seed=7, folds=5, layer_rank=2, head_rank=2, latent_rank=3)
    first, metadata = score_records(records, **kwargs)
    labeled = copy.deepcopy(records)
    for index, record in enumerate(labeled):
        record["label"] = np.asarray([(index + token) % 2 for token in range(7)])
        record["artifact"]["labels"] = np.ones(7) * index
    second, repeated_metadata = score_records(labeled, **kwargs)
    reversed_scores, reversed_metadata = score_records(
        list(reversed(records)), **kwargs
    )

    assert metadata == repeated_metadata
    assert metadata == reversed_metadata
    assert metadata["crossfit_complete"] is True
    assert metadata["nuisance_covariates"] == ["full_logprob", "full_margin"]
    assert metadata["nuisance_fit"] == "fit_sources_only"
    for sample in first:
        assert set(first[sample]) == {
            "mechanism_innovation",
            "static_state",
            "confidence",
        }
        for name in first[sample]:
            np.testing.assert_array_equal(first[sample][name], second[sample][name])
            np.testing.assert_array_equal(
                first[sample][name], reversed_scores[sample][name]
            )
            assert np.isfinite(first[sample][name]).all()
            assert len(first[sample][name]) == 7
        for name in ("mechanism_innovation", "static_state"):
            assert (first[sample][name] >= 0).all()
            assert (first[sample][name] <= 1).all()


def test_fit_only_confidence_conditioning_removes_global_common_shifts():
    records = _records()
    kwargs = dict(seed=7, folds=5, layer_rank=2, head_rank=2, latent_rank=3)
    baseline, _ = score_records(records, **kwargs)
    shifted = copy.deepcopy(records)
    for record in shifted:
        inputs = record["artifact"]["score_inputs"]
        for name in (
            "full_logprob",
            "no_evidence_logprob",
            "no_history_logprob",
            "no_evidence_history_logprob",
        ):
            inputs[name] += 17.0
        inputs["full_margin"] -= 9.0
        inputs["no_evidence_history_margin"] -= 9.0
    conditioned, _ = score_records(shifted, **kwargs)
    for sample in baseline:
        for name in ("mechanism_innovation", "static_state"):
            np.testing.assert_array_equal(
                baseline[sample][name], conditioned[sample][name]
            )


def test_first_token_uses_static_initial_state_before_ar_innovation():
    records = _records(7)
    model = _fit_model(
        records[:6],
        layer_rank=2,
        head_rank=2,
        latent_rank=3,
        max_fit_tokens_per_response=128,
    )
    innovation, static = _raw_scores(records[6], model)
    assert innovation[0] == static[0]
    assert not np.array_equal(innovation[1:], static[1:])


def test_a_test_source_cannot_change_a_partner_in_its_held_out_fold():
    records = _records()
    kwargs = dict(seed=7, folds=5, layer_rank=2, head_rank=2, latent_rank=3)
    assignment = source_fold_assignments(
        [record["source_id"] for record in records], folds=5, seed=7
    )
    same_fold = {}
    for record in records:
        same_fold.setdefault(assignment[record["source_id"]], []).append(record)
    changed_record, partner = next(
        values for values in same_fold.values() if len(values) >= 2
    )[:2]
    baseline, _ = score_records(records, **kwargs)
    changed = copy.deepcopy(records)
    changed_sample = changed_record["sample_id"]
    for record in changed:
        if record["sample_id"] == changed_sample:
            record["artifact"]["trace"]["edge_role_energy"] *= 1000.0
            record["artifact"]["trace"]["head_role_write_norm"] *= 0.001
    rescored, _ = score_records(changed, **kwargs)
    partner_sample = partner["sample_id"]
    np.testing.assert_array_equal(
        baseline[partner_sample]["mechanism_innovation"],
        rescored[partner_sample]["mechanism_innovation"],
    )
    np.testing.assert_array_equal(
        baseline[partner_sample]["static_state"],
        rescored[partner_sample]["static_state"],
    )


def test_limit_one_marks_formal_crossfit_unavailable_without_a_second_detector():
    scores, metadata = score_records(_records(1), folds=5)
    assert metadata["crossfit_complete"] is False
    assert metadata["mechanism_scores_available"] is False
    assert scores["sample-0"]["mechanism_innovation"].shape == (7,)
    assert not scores["sample-0"]["mechanism_innovation"].any()


def test_mixed_tasks_and_nonpositive_fit_cap_are_rejected():
    records = _records(3)
    records[0]["task_type"] = "Summary"
    try:
        score_records(records)
    except ValueError as error:
        assert "one task_type" in str(error)
    else:
        raise AssertionError("mixed task detector fit was accepted")
    try:
        score_records(_records(3), max_fit_tokens_per_response=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive fit cap was accepted")
    try:
        score_records(_records(3), folds=2)
    except ValueError as error:
        assert "at least three" in str(error)
    else:
        raise AssertionError("two-fold detector fit was accepted")


def test_invalid_mechanism_shape_and_nonfinite_branch_are_rejected():
    shape_error = _artifact(4)
    shape_error["trace"]["role_head_coherence"] = np.ones((2, 7, 3))
    try:
        mechanism_tensor(shape_error)
    except ValueError as error:
        assert "[L,T,4]" in str(error)
    else:
        raise AssertionError("invalid coherence shape was accepted")

    partition_error = _artifact(4)
    partition_error["trace"]["role_attention_mass"][0, 0, 0] *= 0.5
    try:
        mechanism_tensor(partition_error)
    except ValueError as error:
        assert "partition" in str(error)
    else:
        raise AssertionError("incomplete attention role partition was accepted")

    nonfinite = _artifact(4)
    nonfinite["score_inputs"]["no_history_logprob"][2] = np.nan
    try:
        factorial_contrasts(nonfinite)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite factorial branch was accepted")
