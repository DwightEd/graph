import numpy as np

from .density import distances, select_neighbors
from .score import differences


def test_full_coordinate_distance_matches_explicit_pairs():
    generator = np.random.default_rng(42)
    first = generator.normal(size=(7, 11)).astype(np.float32)
    second = generator.normal(size=(9, 11)).astype(np.float32)
    expected = ((first.astype(float)[:, None] - second.astype(float)[None])**2).mean(-1)
    np.testing.assert_allclose(distances(first, second), expected, atol=1e-6)


def test_neighbor_selection_excludes_entire_source():
    reference = np.repeat(['a', 'b', 'c'], 20)
    distance = np.arange(120).reshape(2, 60).astype(float)
    neighbors = select_neighbors(distance, distance[:, ::-1], ['a', 'new'], reference, True)
    assert len(neighbors[0]) == 16
    assert 'a' not in reference[neighbors[0]]
    assert len(neighbors[1]) == 24
    assert all((reference[neighbors[1]] == source).sum() == 8 for source in ['a', 'b', 'c'])


def test_transition_starts_at_zero_without_cross_answer_predecessor():
    values = np.array([[1., 5.], [3., 1.]])
    np.testing.assert_array_equal(differences(values), [[0., 0.], [2., -4.]])


def test_head_identity_retains_difference_hidden_by_head_mean():
    first = np.array([[-1., 1.]], dtype=np.float32)
    second = np.array([[1., -1.]], dtype=np.float32)
    assert first.mean() == second.mean()
    assert distances(first, second)[0, 0] == 4.
