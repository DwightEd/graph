import numpy as np

from route_graph.operator import PathEncoder


def test_same_role_mass_and_spectrum_can_have_different_candidate_routes():
    attention = np.eye(4)[None, None]
    attention[0, 0, 3] = [0.7, 0.1, 0.1, 0.1]
    rewired = attention.copy()
    rewired[0, 0, 3, :2] = [0.1, 0.7]
    signals = np.array([1.0, -1.0, 0.0, 0.0])[None, :, None]
    roles = np.array([0, 0, 2, 3])
    encoder = PathEncoder(depths=(1,))

    first = encoder.encode(attention, signals, roles, query=3, end_layers=(0,))
    second = encoder.encode(rewired, signals, roles, query=3, end_layers=(0,))

    np.testing.assert_allclose(np.linalg.eigvals(attention), np.linalg.eigvals(rewired))
    np.testing.assert_allclose(first["null"], second["null"])
    np.testing.assert_allclose(first["signal"], second["signal"])
    assert first["residual"][0] > 0
    assert second["residual"][0] < 0


def test_null_preserves_lag_and_passage_allocation_instead_of_calling_them_alignment():
    attention = np.eye(4)[None, None]
    attention[0, 0, 3] = [0.8, 0.1, 0, 0.1]
    signals = np.array([1.0, -1.0, 0.0, 0.0])[None, :, None]
    encoder = PathEncoder(depths=(1,))
    # Distinct source passages: each endpoint is alone in its exchangeability cell.
    result = encoder.encode(
        attention,
        signals,
        np.array([0, 0, 2, 3]),
        source_units=np.array([0, 1, -1, -1]),
        query=3,
        end_layers=(0,),
    )
    np.testing.assert_allclose(result["residual"], 0, atol=1e-12)
    # Same passage, but distinct log-lag cells must also retain their masses.
    attention[0, 0, 3] = [0, 0.8, 0.1, 0.1]
    signals[0, :, 0] = [0, 1, -1, 0]
    result = encoder.encode(
        attention, signals, np.array([1, 0, 0, 3]), query=3, end_layers=(0,)
    )
    np.testing.assert_allclose(result["residual"], 0, atol=1e-12)


def test_two_step_history_path_uses_layer_order_and_excludes_query_self():
    attention = np.tile(np.eye(6), (2, 1, 1, 1))
    attention[0, 0, 3] = [0.8, 0.1, 0, 0.1, 0, 0]
    attention[0, 0, 5] = [0.6, 0.2, 0, 0, 0, 0.2]
    attention[1, 0, 5] = [0, 0, 0, 0.6, 0, 0.4]
    signals = np.tile(np.array([1.0, -1.0, 0, 0, 0, 0])[None, :, None], (2, 1, 1))
    roles = np.array([0, 0, 1, 2, 2, 2])
    encoder = PathEncoder(depths=(1, 2))
    actual = encoder.encode(attention, signals, roles, query=5, end_layers=(1,))
    index = actual["names"].index("l1/d2/source_via_history/h0/c1")
    np.testing.assert_allclose(actual["observed"][index], 0.105, atol=1e-12)
    np.testing.assert_allclose(actual["residual"][index], 0.105, atol=1e-12)
    reversed_layers = encoder.encode(
        attention[::-1], signals, roles, query=5, end_layers=(1,)
    )
    np.testing.assert_allclose(reversed_layers["residual"][index], 0, atol=1e-12)


def test_role_constant_signals_have_zero_one_step_residual():
    rng = np.random.default_rng(7)
    attention = np.tril(rng.random((2, 3, 12, 12)))
    attention /= attention.sum(-1, keepdims=True)
    roles = np.array([3, 1, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2])
    signals = np.tile((roles == 0)[None, :, None].astype(float), (2, 1, 1))
    encoded = PathEncoder(depths=(1,)).encode(
        attention, signals, roles, query=11, end_layers=(0, 1)
    )
    np.testing.assert_allclose(encoded["residual"], 0, atol=1e-12)
