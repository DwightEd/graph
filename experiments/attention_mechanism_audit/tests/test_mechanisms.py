from types import SimpleNamespace

import numpy as np

from experiments.attention_mechanism_audit.mechanisms import combine_mechanisms


def test_mechanisms_keep_axes_roles_and_counterfactuals_separate():
    role_energy = np.asarray([[[2.0, 1.0, 1.0, 8.0, 4.0]]])
    functional = {
        "functional_role_names": np.asarray(
            ["evidence", "question", "constraint", "other_prompt", "history"]
        ),
        "functional_absolute_layer_role": role_energy,
        "functional_signed_layer_role": role_energy.copy(),
        "functional_entropy_observed": np.asarray([[[0.25]]]),
        "functional_hhi_observed": np.asarray([[[0.75]]]),
        "functional_head_role_js": np.asarray([[0.1]]),
        "functional_cancellation": np.asarray([[0.2]]),
        "functional_known_attention_coverage": np.asarray([[[0.9]]]),
    }
    routing_names = np.asarray(
        [
            "evidence",
            "question",
            "constraint",
            "other_prompt",
            "history",
            "diagonal",
            "unresolved",
        ]
    )
    routing = {
        "routing_role_names": routing_names,
        "routing_mean_fine_role_mass": np.asarray(
            [[[0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1]]]
        ),
        "routing_entropy_bounds": np.asarray([[[0.2, 0.4]]]),
        "routing_concentration_bounds": np.asarray([[[0.3, 0.6]]]),
        "routing_head_role_js": np.asarray([[0.2]]),
        "routing_direct_role_ancestry": np.asarray([[[0.2, 0.1, 0.1, 0.0]]]),
        "routing_relayed_role_ancestry": np.asarray([[[0.1, 0.0, 0.0, 0.0]]]),
        "routing_grounded_role_ancestry": np.asarray([[[0.3, 0.1, 0.1, 0.0]]]),
        "routing_ungrounded_history_ancestry": np.asarray([[0.4]]),
        "routing_unresolved_ancestry": np.asarray([[0.1]]),
    }
    variant = lambda logp, margin, jsd=0.0, available=True: SimpleNamespace(
        chosen_logprob=np.asarray([logp]),
        chosen_vs_best_other_margin=np.asarray([margin]),
        vocab_jsd_from_full=np.asarray([jsd]),
        available=available,
    )
    counterfactual = {
        "full": variant(-1.0, 2.0),
        "no_evidence": variant(-1.4, 1.5, 0.2),
        "swapped_evidence_0": variant(-1.2, 1.8, 0.1),
        "swapped_evidence_1": variant(-1.4, 1.4, 0.3),
        "swapped_evidence_2": variant(-0.8, 2.0, 0.2),
        "no_history": variant(-1.5, 1.0, 0.3),
        "no_evidence_no_history": variant(-2.1, 0.5, 0.4),
    }

    result = combine_mechanisms(functional, routing, counterfactual)
    # Grounding excludes the much larger other_prompt/template energy:
    # history / (evidence + question + constraint) = 4 / 4.
    np.testing.assert_allclose(
        result["drift_functional_history_to_grounding_log_ratio"], 0.0
    )
    assert "functional_signed_evidence" in result
    assert "functional_absolute_history" in result
    assert "routing_mean_mass_unresolved" in result
    np.testing.assert_allclose(result["counterfactual_no_evidence_delta"], -0.4)
    np.testing.assert_allclose(
        result["counterfactual_swapped_evidence_delta"], -2.0 / 15.0
    )
    np.testing.assert_allclose(
        result["counterfactual_evidence_bypass"], -4.0 / 15.0
    )
    np.testing.assert_allclose(
        result["counterfactual_swapped_evidence_donor_std"],
        np.std([-1.2, -1.4, -0.8]),
    )
    assert result["counterfactual_swapped_evidence_available_count"].item() == 3
    np.testing.assert_allclose(
        result["counterfactual_swapped_evidence_margin_delta"],
        np.mean([1.8, 1.4, 2.0]) - 2.0,
    )
    np.testing.assert_allclose(
        result["counterfactual_swapped_evidence_jsd_from_full"], 0.2
    )
    np.testing.assert_allclose(result["counterfactual_history_necessity"], 0.5)
    np.testing.assert_allclose(
        result["counterfactual_evidence_history_interaction"], -0.2
    )
    assert not any("weighted_total" in name for name in result)


def test_swap_ensemble_ignores_unavailable_donors_without_zero_imputation():
    role_energy = np.ones((2, 1, 5), dtype=np.float64)
    functional = {
        "functional_role_names": np.asarray(
            ["evidence", "question", "constraint", "other_prompt", "history"]
        ),
        "functional_absolute_layer_role": role_energy,
        "functional_signed_layer_role": role_energy,
        "functional_entropy_observed": np.ones((2, 1, 1)),
        "functional_hhi_observed": np.ones((2, 1, 1)),
        "functional_head_role_js": np.zeros((2, 1)),
        "functional_cancellation": np.zeros((2, 1)),
        "functional_known_attention_coverage": np.ones((2, 1, 1)),
    }
    routing = {
        "routing_role_names": np.asarray(
            [
                "evidence",
                "question",
                "constraint",
                "other_prompt",
                "history",
                "diagonal",
                "unresolved",
            ]
        ),
        "routing_mean_fine_role_mass": np.ones((2, 1, 7)) / 7,
        "routing_entropy_bounds": np.ones((2, 1, 2)),
        "routing_concentration_bounds": np.ones((2, 1, 2)),
        "routing_head_role_js": np.zeros((2, 1)),
        "routing_direct_role_ancestry": np.ones((2, 1, 4)),
        "routing_relayed_role_ancestry": np.ones((2, 1, 4)),
        "routing_grounded_role_ancestry": np.ones((2, 1, 4)),
        "routing_ungrounded_history_ancestry": np.zeros((2, 1)),
        "routing_unresolved_ancestry": np.zeros((2, 1)),
    }

    def scores(values, *, available=True):
        values = np.asarray(values, dtype=np.float64)
        return SimpleNamespace(
            chosen_logprob=values,
            chosen_vs_best_other_margin=values + 2,
            vocab_jsd_from_full=np.abs(values) / 10,
            available=available,
        )

    nan = [np.nan, np.nan]
    counterfactual = {
        "full": scores([-1.0, -2.0]),
        "no_evidence": scores([-1.5, -2.5]),
        "no_history": scores([-1.2, -2.2]),
        "no_evidence_no_history": scores([-1.8, -2.8]),
        "swapped_evidence_0": scores([-1.2, -2.4]),
        "swapped_evidence_1": scores(nan, available=False),
        "swapped_evidence_2": scores(nan, available=False),
    }
    result = combine_mechanisms(functional, routing, counterfactual)
    np.testing.assert_allclose(
        result["counterfactual_swapped_evidence_delta"], [-0.2, -0.4]
    )
    np.testing.assert_array_equal(
        result["counterfactual_swapped_evidence_available_count"], [1, 1]
    )
    np.testing.assert_allclose(
        result["counterfactual_swapped_evidence_donor_std"], [0.0, 0.0]
    )


def test_all_swap_donors_unavailable_yields_nan_and_zero_count():
    unavailable = SimpleNamespace(available=False)
    full = SimpleNamespace(
        chosen_logprob=np.asarray([-1.0]),
        chosen_vs_best_other_margin=np.asarray([1.0]),
        vocab_jsd_from_full=np.asarray([0.0]),
        available=True,
    )
    counterfactual = {
        "full": full,
        "no_evidence": full,
        "no_history": full,
        "no_evidence_no_history": full,
        "swapped_evidence_0": unavailable,
        "swapped_evidence_1": unavailable,
        "swapped_evidence_2": unavailable,
    }
    from experiments.attention_mechanism_audit.mechanisms import (
        _counterfactual_trajectories,
    )

    result = _counterfactual_trajectories(counterfactual)
    for name in (
        "counterfactual_swapped_evidence_delta",
        "counterfactual_swapped_evidence_donor_std",
        "counterfactual_evidence_bypass",
        "counterfactual_swapped_evidence_margin_delta",
        "counterfactual_swapped_evidence_jsd_from_full",
    ):
        assert np.isnan(result[name]).all()
    assert result["counterfactual_swapped_evidence_available_count"].item() == 0
