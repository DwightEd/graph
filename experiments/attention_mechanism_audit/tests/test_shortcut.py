import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.attention_mechanism_audit.schema import SHORTCUT_VECTOR_NAMES
from experiments.attention_mechanism_audit.shortcut import (
    _adjacent_swap,
    capture_shortcut_geometry,
    shortcut_layer_metrics,
    shortcut_token_metrics,
)


def test_adjacent_rewire_changes_endpoints_without_large_lag_jumps():
    endpoint = torch.tensor([7, 8, 9, 10, 11])
    rewired = _adjacent_swap(endpoint)

    assert torch.equal(rewired, torch.tensor([8, 7, 10, 9, 11]))
    assert torch.equal(torch.sort(rewired).values, endpoint)
    assert torch.all((rewired[:4] - endpoint[:4]).abs() == 1)


def test_observed_response_endpoint_completes_route_better_than_rewire():
    branches, rows, heads, sources, kv_heads, head_dim = 4, 1, 1, 5, 1, 2
    attention = torch.zeros(branches, rows, heads, sources)
    # source 0 is evidence; 2 and 3 are strict response history; 4 is self.
    attention[0, 0, 0, [0, 2, 3, 4]] = torch.tensor([0.2, 0.7, 0.1, 0.0])
    attention[1, 0, 0, [2, 3]] = torch.tensor([0.7, 0.1])
    attention[2] = attention[0]
    # noE/noEH removes strict history at the current target.

    value = torch.zeros(branches, sources, kv_heads, head_dim)
    value[0, 2, 0] = torch.tensor([1.2, 0.0])
    value[0, 3, 0] = torch.tensor([0.0, 1.2])
    value[1, 2, 0] = torch.tensor([0.2, 0.0])
    value[1, 3, 0] = torch.tensor([0.0, 0.2])
    value[2] = value[0]
    value[3] = value[1]

    roles = (
        torch.tensor([[True, False, False, False, False]]),
        torch.tensor([[False, True, False, False, False]]),
        torch.tensor([[False, False, True, True, False]]),
        torch.tensor([[False, False, False, False, True]]),
    )
    geometry = capture_shortcut_geometry(
        attention,
        value,
        roles,
        q_to_kv=torch.tensor([0]),
        output_weight=torch.eye(2),
        output_gram=torch.eye(2)[None],
    )

    assert geometry["route_gram"].shape == (1, len(SHORTCUT_VECTOR_NAMES), 7)
    assert geometry["head_gram"].shape == (1, 1, 7, 7)
    assert geometry["rewire_valid"].item()
    torch.testing.assert_close(
        geometry["route_gram"], geometry["route_gram"].transpose(-1, -2)
    )

    trace = {
        "shortcut_route_gram": geometry["route_gram"][None],
        "shortcut_rewire_valid": geometry["rewire_valid"][None],
    }
    metrics = shortcut_layer_metrics(trace)
    assert metrics["shortcut_relay_completion"][0, 0] > 0.99
    assert metrics["shortcut_route_completion"][0, 0] > 0.99
    assert metrics["shortcut_rewired_route_completion"][0, 0] < 0.5
    assert metrics["shortcut_endpoint_rewire_gap"][0, 0] < -0.5


def test_shortcut_candidate_is_residual_history_aligned_with_autonomy():
    layers, tokens, vectors, hidden = 3, 4, 7, 3
    state = torch.zeros(layers, tokens, vectors, hidden)
    state[..., 0, 0] = 1.0  # full history
    state[..., 1, 1] = 1.0  # direct evidence, orthogonal to history
    state[..., 2, 1] = 1.0  # evidence carrier, also orthogonal
    state[..., 4, 0] = 1.0  # autonomous history matches unexplained history
    state[..., 5, 0] = 1.0  # rewired carrier spuriously explains the history
    gram = torch.einsum("ltkd,ltmd->ltkm", state, state)
    artifact = {
        "trace": {
            "shortcut_route_gram": gram,
            "shortcut_rewire_valid": torch.ones(layers, tokens, dtype=torch.bool),
        }
    }

    layer = shortcut_layer_metrics(artifact["trace"])
    token = shortcut_token_metrics(artifact)

    np.testing.assert_allclose(layer["shortcut_route_completion"], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        layer["shortcut_autonomous_residual_alignment"], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(layer["shortcut_route_candidate"], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        layer["shortcut_route_rewired_control"], 0.0, atol=1e-6
    )
    np.testing.assert_allclose(token["shortcut_route_candidate_mean"], 1.0)
    assert token["shortcut_route_candidate_mean__valid"].all()
