"""Token information must survive unit pooling; TV plateaus must preserve exact ties."""

import numpy as np
from scipy import sparse

from experiments.native_support.unified.calibration import fit_scales
from experiments.native_support.unified.graph import unit_operator
from experiments.native_support.unified.scoring import score_record
from experiments.native_support.unified.token_scoring import add_token_scores
from experiments.native_support.unified.token_solver import solve_tokens


def test_soft_prior_has_closed_form_and_does_not_force_the_unit_mean():
    units = [dict(start=0, stop=3), dict(start=3, stop=4)]
    observed = np.array([.1, .2, .6, .7])
    anchor = np.array([.8, .8, .8, .1])
    laplacian = sparse.csr_matrix((4, 4))
    score, diagnostic = solve_tokens(observed, observed, anchor, units, laplacian, 1, 0, 0)
    np.testing.assert_allclose(score, [.35, .45, .85, .4])
    np.testing.assert_allclose(unit_operator(units, 4) @ score, [.55, .4])
    assert diagnostic["unit_anchor_max_shift"] > .2
    unregularized, _ = solve_tokens(observed, observed, anchor, units, laplacian, 0, 0, 0)
    np.testing.assert_array_equal(unregularized, observed)


def test_tv_keeps_a_change_inside_one_unit_and_removes_plateau_ranking_jitter():
    # Analytical two-plateau solution: one TV boundary shifts each block by tau / length.
    observed = np.array([.1, .1, .9, .9])
    units = [dict(start=0, stop=4)]
    score, diagnostic = solve_tokens(observed, observed, np.full(4, .2), units,
                                     sparse.csr_matrix((4, 4)), 0, 0, .05)
    np.testing.assert_allclose(score, [.125, .125, .875, .875], atol=1e-12)
    assert score[0] == score[1] and score[2] == score[3]
    assert diagnostic["optimality_max_error"] < 1e-10
    constant, _ = solve_tokens(observed, observed, observed, units, sparse.csr_matrix((4, 4)), 0, 0, 1)
    np.testing.assert_array_equal(constant, np.full(4, .5))


def test_graph_couples_corrections_without_erasing_direct_source_differences():
    source = np.array([.1, .9])
    laplacian = sparse.csr_matrix([[1., -1.], [-1., 1.]])
    score, _ = solve_tokens(source, source, source, [dict(start=0, stop=2)], laplacian, 0, 100, 0)
    np.testing.assert_allclose(score, source, atol=1e-12)
    singleton, _ = solve_tokens(np.array([.8]), np.array([.8]), np.array([.2]),
        [dict(start=0, stop=1)], sparse.csr_matrix((1, 1)), 1, 1, .05)
    np.testing.assert_allclose(singleton, [.5])


def test_equal_unit_means_and_equal_routes_can_still_have_different_token_scores():
    units = [dict(start=0, stop=4)]
    edges = {name: np.array([], dtype=int) for name in ("target", "key", "sham_key")}
    edges.update(effect_with_source=np.array([]), effect_without_source=np.array([]))
    measured = []
    for source_values in (np.array([0., 0., 1., 1.]), np.array([1., 1., 0., 0.])):
        record = dict(response=dict(id="example", source_id="source"), views=dict(units=units),
            carrier_score="token_source_selected", scores=dict(raw_route=np.full(4, .5),
                source_local=source_values, token_source_selected=source_values,
                target=np.arange(4), unit_id=np.zeros(4, dtype=int)))
        scales = fit_scales([record], include_tokens=True)
        scores, components, _ = score_record(record, edges, scales, 1)
        add_token_scores(record, edges, scores, components, 1, 1, .05)
        measured.append(scores)
    first, second = measured
    np.testing.assert_array_equal(first["unified"], second["unified"])
    assert first["unified_token"][2] > first["unified_token"][0]
    np.testing.assert_allclose(first["unified_token"], second["unified_token"][::-1])
    np.testing.assert_array_equal(first["unified_token_no_token_source"], np.full(4, .5))
