import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path = Path(__file__).with_name('risk_transport.py')
    if not path.exists():
        path = Path(__file__).parents[1] / 'next_iteration' / 'risk_transport.py'
    spec = importlib.util.spec_from_file_location('risk_transport', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_no_history_equals_native_entropy():
    m = module()
    scores = m.transport(np.array([2., 1., .5]), np.zeros((3, 3)), np.ones(3))
    np.testing.assert_array_equal(scores['risk_transport'], [2., 1., .5])


def test_reused_origin_carries_uncertainty():
    m = module()
    g = np.array([[0, 0, 0], [1., 0, 0], [1., 0, 0]])
    scores = m.transport(np.array([2., .1, .1]), g, np.array([1., 0, 0]))
    np.testing.assert_array_equal(scores['risk_transport'], [2., 2., 2.])


def test_graph_and_mass_matched_chain_can_differ():
    m = module()
    g = np.array([[0, 0, 0], [0, 0, 0], [.8, 0, 0]])
    scores = m.transport(np.array([2., .2, .2]), g, np.array([1., 1., .2]))
    assert scores['risk_transport'][2] == pytest.approx(1.64)
    assert scores['mass_matched_chain'][2] == pytest.approx(.2)


def test_future_edges_rejected():
    with pytest.raises(ValueError, match='future'):
        module().transport(np.ones(2), np.eye(2), np.ones(2))


def test_suffix_changes_do_not_change_earlier_scores():
    m = module()
    g = np.array([[0, 0, 0], [.4, 0, 0], [.1, .2, 0]])
    a = m.transport(np.array([1., 2., 3.]), g, np.ones(3))
    b = m.transport(np.array([1., 2., 99.]), g, np.ones(3))
    np.testing.assert_array_equal(a['risk_transport'][:2], b['risk_transport'][:2])
