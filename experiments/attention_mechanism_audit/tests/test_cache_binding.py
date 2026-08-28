import re

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

    with pytest.raises(ValueError, match="does not match") as caught:
        validate_replay_attention(Graph(), (attention,))

    message = str(caught.value)
    assert "worst_retained=(layer=" in message
    assert "worst_diagonal=(layer=" in message
    assert "worst_known_mass=(layer=" in message
    assert "cache=" in message
    assert "replay=" in message
    assert "abs_error=" in message
    assert "per_layer_max=[L0(retained=" in message


def test_cache_binding_reports_exact_worst_retained_endpoint():
    attention = replay_attention()[0].copy()
    attention[0, 0, 3, 0] += 0.03

    with pytest.raises(ValueError) as caught:
        validate_replay_attention(Graph(), (attention,))

    message = str(caught.value)
    endpoint = re.search(
        r"worst_retained=\(layer=0, head=0, query=3, source=0, "
        r"cache=([^,]+), replay=([^,]+), abs_error=([^\)]+)\)",
        message,
    )
    assert endpoint is not None
    cache, replay, error = map(float, endpoint.groups())
    assert cache == pytest.approx(0.1)
    assert replay == pytest.approx(0.13)
    assert error == pytest.approx(0.03)


def test_cache_binding_reports_accumulated_known_mass_endpoint():
    attention = replay_attention()[0].copy()
    attention[0, 0, 2, 0] += 0.004
    attention[0, 0, 2, 2] += 0.004

    with pytest.raises(ValueError) as caught:
        validate_replay_attention(Graph(), (attention,))

    message = str(caught.value)
    maxima = re.search(
        r"retained_max=([^,]+), diagonal_max=([^,]+), "
        r"known_mass_max=([^,]+)",
        message,
    )
    assert maxima is not None
    retained, diagonal, known_mass = map(float, maxima.groups())
    assert retained == pytest.approx(0.004, abs=2e-8)
    assert diagonal == pytest.approx(0.004, abs=2e-8)
    assert known_mass == pytest.approx(0.008, abs=2e-8)
    endpoint = re.search(
        r"worst_known_mass=\(layer=0, head=0, query=2, "
        r"cache=([^,]+), replay=([^,]+), abs_error=([^\)]+)\)",
        message,
    )
    assert endpoint is not None
    cache, replay, error = map(float, endpoint.groups())
    assert cache == pytest.approx(0.6)
    assert replay == pytest.approx(0.608)
    assert error == pytest.approx(0.008, abs=2e-8)


def test_cache_binding_rejects_wrong_geometry():
    with pytest.raises(ValueError, match="layer count"):
        validate_replay_attention(Graph(), ())
