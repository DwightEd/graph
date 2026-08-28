from types import SimpleNamespace

import numpy as np
import pytest

from experiments.attention_mechanism_audit.cache_binding import (
    validate_replay_attention,
)


class Edges:
    def __init__(self, source, target, head, weight):
        self.source = np.asarray(source, dtype=np.int64)
        self.target = np.asarray(target, dtype=np.int64)
        self.head = np.asarray(head, dtype=np.int64)
        self.weight = np.asarray(weight, dtype=np.float32)


class Graph:
    response_count = 2
    response_start = 2
    token_count = 4
    layer_count = 1
    head_count = 2

    def __init__(self):
        self._edges = Edges(
            source=[0, 1, 0],
            target=[2, 2, 3],
            head=[0, 1, 0],
            weight=[0.2, 0.3, 0.1],
        )
        self.diagonal = np.asarray([[[0.4, 0.5]], [[0.6, 0.7]]])
        known = np.asarray([[[0.6, 0.8]], [[0.7, 0.7]]])
        self.unresolved = 1.0 - known

    def layer_edges(self, layer):
        assert layer == 0
        return self._edges


def replay_attention():
    value = np.zeros((1, 2, 4, 4), dtype=np.float32)
    value[0, 0, 2, 0] = 0.2
    value[0, 1, 2, 1] = 0.3
    value[0, 0, 3, 0] = 0.1
    value[0, 0, 2, 2] = 0.4
    value[0, 1, 2, 2] = 0.5
    value[0, 0, 3, 3] = 0.6
    value[0, 1, 3, 3] = 0.7
    return (value,)


def test_cache_binding_checks_sparse_diagonal_and_known_mass():
    result = validate_replay_attention(Graph(), replay_attention())

    assert result.verified is True
    assert result.retained_endpoints_compared == 3
    assert result.diagonal_endpoints_compared == 4
    assert result.retained_max_abs_error < 1e-7
    assert result.known_mass_max_abs_error < 1e-7


@pytest.mark.parametrize("index", [(0, 0, 2, 0), (0, 1, 3, 3)])
def test_cache_binding_rejects_weight_or_diagonal_mismatch(index):
    attention = replay_attention()[0].copy()
    attention[index] += 0.02

    with pytest.raises(ValueError, match="does not match"):
        validate_replay_attention(Graph(), (attention,))


def test_cache_binding_rejects_wrong_geometry():
    with pytest.raises(ValueError, match="layer count"):
        validate_replay_attention(Graph(), ())
