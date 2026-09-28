"""Scientific invariants for source ordering, localization and calibration."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.unsupervised_graph.refinement import (
    centered, lexicographic_cutoff, lexicographic_scores, unit_groups, unit_window,
)


def test_window_respects_units_answers_and_original_token_gaps():
    pack = dict(answer_index=np.array([0, 0, 0, 0, 1]),
                unit_index=np.array([0, 0, 0, 1, 0]), target=np.array([0, 1, 8, 9, 0]))
    values = np.array([1., 3., 30., 100., 1000.])
    np.testing.assert_array_equal(unit_window(values, pack), [2., 2., 30., 100., 1000.])


def test_residual_zero_mean_per_unit_not_across_answers():
    pack = dict(answer_index=np.array([0, 0, 0, 1]), unit_index=np.array([0, 0, 1, 0]))
    residual = centered(np.array([.1, .7, .9, .2]), unit_groups(pack))
    np.testing.assert_allclose(residual, [-.3, .3, 0., 0.], atol=1e-15)
    assert np.max(np.abs(residual)) <= 1


def test_tie_preserves_distinct_unseen_anchors_even_with_opposite_details():
    # A fit-bin approach would collapse all anchors between 0 and 1.
    anchor = np.array([.25, .25000001, .8, .8])
    detail = np.array([1., -1., -.5, .5])
    cutoff = dict(anchor=.5, detail=0.)
    score, threshold = lexicographic_scores(anchor, detail, cutoff)
    assert np.all(np.diff(score) > 0)
    np.testing.assert_array_equal(score > threshold, [False, False, True, True])


def test_lexicographic_threshold_and_ranking_are_batch_invariant():
    anchor = np.array([0., 1., 1., 1., 2.])
    detail = np.array([1., -.5, 0., .5, -1.])
    cutoff = dict(anchor=1., detail=0.)
    score, threshold = lexicographic_scores(anchor, detail, cutoff)
    expected = (anchor > 1.) | ((anchor == 1.) & (detail > 0.))
    np.testing.assert_array_equal(score > threshold, expected)
    for index in range(len(anchor)):
        single, limit = lexicographic_scores(anchor[index:index+1], detail[index:index+1], cutoff)
        assert bool(single[0] > limit) == expected[index]


def test_higher_quantile_cutoff_handles_tied_anchors_and_details():
    anchor = np.array([0., 1., 1., 1., 2.])
    detail = np.array([0., -.5, 0., .5, -1.])
    cutoff = lexicographic_cutoff(anchor, detail, quantile=.5)
    assert cutoff == dict(kind="lexicographic", anchor=1., detail=0.)
    score, threshold = lexicographic_scores(anchor, detail, cutoff)
    np.testing.assert_array_equal(score > threshold, [False, False, False, True, True])


def test_selection_refuses_any_post_freeze_update(tmp_path):
    from experiments.unsupervised_graph.refine_run import select
    (tmp_path / "SCORING_FREEZE.json").touch()
    with pytest.raises(FileExistsError, match="frozen"):
        select(SimpleNamespace(output=tmp_path), "QA")


def test_cpu_scalar_entry_does_not_import_torch():
    import subprocess
    import sys
    command = "import main; main.main(['source-refine', '--help'])"
    command = "import sys\ntry:\n    " + command + "\nexcept SystemExit:\n    pass\nassert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True)
