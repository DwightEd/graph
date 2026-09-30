"""Scientific invariants: no label fitting, source isolation and causal prefix stability."""

import copy

import numpy as np
import pytest

from experiments.native_support.dual_state.scoring import (
    calibrate, fit_reference, head_scores, reference_records, window_mean,
)


def observation(identity, source, values):
    values = np.asarray(values, dtype=float)
    return dict(id=identity, source_id=source, head_route=values,
                response_energy=np.ones_like(values),
                target=np.arange(len(values)), response_length=len(values))


def test_current_survives_and_causal_prefix_does_not_read_future():
    reference = [observation("a", "a", [[0, 1]] * 25), observation("b", "b", [[1, 0]] * 25)]
    fitted = fit_reference(reference)
    query = observation("q", "q", [[.2, .3]] * 25)
    query["head_route"][12] = [2, 2]
    scores, state = head_scores(query, fitted, 6)
    changed = copy.deepcopy(query)
    changed["head_route"][13:] = -3
    later, _ = head_scores(changed, fitted, 6)
    prefix = observation("q", "q", query["head_route"][:13])
    truncated, _ = head_scores(prefix, fitted, 6)
    for name in ("head_current", "head_causal_persistent", "head_causal_dual"):
        np.testing.assert_allclose(scores[name][:13], later[name][:13])
        np.testing.assert_allclose(scores[name][:13], truncated[name])
    assert not np.allclose(scores["head_offline_persistent"][:13], later["head_offline_persistent"][:13])
    assert scores["head_causal_dual"][12] == scores["head_current"][12] == 1
    assert np.all(scores["head_causal_dual"] >= scores["head_current"])
    np.testing.assert_array_equal(state["current"], calibrate(query, fitted))


def test_temporal_accumulation_precedes_positive_head_pooling():
    reference = [observation("a", "a", [[0, 0]] * 4), observation("b", "b", [[1, 1]] * 4)]
    query = observation("q", "q", [[2, -1], [-1, 2], [2, -1], [-1, 2]])
    scores, _ = head_scores(query, fit_reference(reference), 2)
    np.testing.assert_allclose(scores["head_current"], np.sqrt(.5))
    # Alternating heads have no persistent per-head positive drift, despite constant pooled current.
    np.testing.assert_allclose(scores["head_causal_persistent"][1:], 0)
    assert window_mean(scores["head_current"], 2)[1] > 0


def test_zero_response_has_no_instantaneous_directional_risk():
    reference = [observation("a", "a", [[-1]] * 4), observation("b", "b", [[-.5]] * 4)]
    query = observation("q", "q", [[0]] * 4)
    query["response_energy"][:] = 0
    np.testing.assert_array_equal(calibrate(query, fit_reference(reference)), 0)


def test_reference_source_weights_ties_and_whole_source_exclusion():
    records = [observation("a1", "a", [[0]] * 4), observation("a2", "a", [[0]] * 4),
               observation("b", "b", [[1]] * 4), observation("c", "c", [[2]] * 4)]
    selected = reference_records(records, "a")
    assert {r["id"] for r in selected} == {"b", "c"}
    query = observation("q", "q", [[.5], [.5], [.5], [.5]])
    first = calibrate(query, fit_reference(records[:3]))
    second = calibrate(query, fit_reference([records[0], records[2]]))
    np.testing.assert_allclose(first, 0)
    np.testing.assert_allclose(first, second)
    tied = observation("q", "q", [[0]] * 4)
    np.testing.assert_allclose(calibrate(tied, fit_reference(records[:3])), -.5)
    with pytest.raises(ValueError, match="two source-disjoint"):
        reference_records(records[:3], "a")
