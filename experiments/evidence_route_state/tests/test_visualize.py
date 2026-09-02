from pathlib import Path

import numpy as np

from experiments.evidence_route_state.registers import ORIGIN_NAMES
from experiments.evidence_route_state.visualize import (
    choose_prediction,
    frame_view,
    plot_sample,
    read_capture,
    read_score,
)


class Tokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"t{token_id}" for token_id in token_ids]


def grams(vectors: np.ndarray) -> np.ndarray:
    return vectors @ np.swapaxes(vectors, -1, -2)


def sample_arrays() -> dict[str, np.ndarray]:
    tokens, layers, heads, channels, hidden = 3, 2, 2, 4, 3
    node = np.arange(tokens * channels * hidden, dtype=np.float32).reshape(
        tokens, channels, hidden
    )
    residual_vectors = np.arange(
        tokens * (layers + 1) * channels * hidden, dtype=np.float32
    ).reshape(tokens, layers + 1, channels, hidden)
    head_vectors = np.arange(
        tokens * layers * heads * channels * hidden, dtype=np.float32
    ).reshape(tokens, layers, heads, channels, hidden)
    topology = np.arange(
        tokens * layers * heads * channels * 7, dtype=np.float32
    ).reshape(tokens, layers, heads, channels, 7)
    topology[..., 1] = np.log1p(topology[..., 1])
    return {
        "token_ids": np.arange(10, 16),
        "query_position": np.array([2, 3, 4]),
        "prediction_position": np.array([3, 4, 5]),
        "valid": np.array([False, False, True]),
        "node_embedding": node,
        "residual_gram": grams(residual_vectors),
        "head_write_gram": grams(head_vectors),
        "route_topology": topology,
        "mlp_relation": np.arange(
            tokens * layers * (channels + 1), dtype=np.float32
        ).reshape(tokens, layers, channels + 1),
        "margin_contribution": np.arange(tokens * channels, dtype=np.float32).reshape(
            tokens, channels
        ),
    }


def write_capture(path: Path) -> None:
    np.savez(path, **sample_arrays())


def test_frame_view_preserves_every_graph_axis(tmp_path):
    path = tmp_path / "sample.npz"
    write_capture(path)
    capture = read_capture(path)
    view = frame_view(capture, prediction_position=5)

    assert view.query_position == 4
    assert view.prediction_position == 5
    assert view.node_norm.shape == (len(ORIGIN_NAMES),)
    assert view.residual_cosine.shape == (3, 4, 4)
    assert view.head_write_cosine.shape == (2, 2, 4, 4)
    assert view.route_topology.shape == (2, 2, 4, 7)
    assert view.mlp_relation.shape == (2, 5)
    assert view.margin_contribution.shape == (4,)
    np.testing.assert_array_equal(
        view.route_topology,
        capture["route_topology"][2],
    )
    np.testing.assert_allclose(
        np.diagonal(view.head_write_cosine, axis1=-2, axis2=-1),
        1.0,
        atol=2e-7,
    )
    assert not any(name.startswith("graph_edge") for name in capture)


def test_prediction_choice_uses_only_valid_frozen_scores():
    capture = sample_arrays()

    assert choose_prediction(capture, None) == 5
    assert choose_prediction(capture, np.array([100.0, 50.0, 1.0])) == 5


def test_read_score_restores_response_order(tmp_path):
    scores = tmp_path / "scores.npz"
    np.savez(
        scores,
        sample_id=np.array(["other", "sample", "sample", "sample"]),
        token_index=np.array([0, 2, 0, 1]),
        conditional_graph_energy=np.array([0.0, 0.95, 0.05, 0.3]),
    )

    np.testing.assert_array_equal(read_score(scores, "sample"), [0.05, 0.3, 0.95])


def test_sample_figure_renders_compact_graph_state(tmp_path):
    capture = tmp_path / "sample.npz"
    output = tmp_path / "sample.png"
    write_capture(capture)

    result = plot_sample(
        capture,
        Tokenizer(),
        output,
        graph_score=np.array([0.05, 0.3, 0.95]),
        sample_name="sample",
    )

    assert result == output
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert output.stat().st_size > 1_000
