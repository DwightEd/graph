from pathlib import Path

import numpy as np

from experiments.evidence_route_state.visualize import (
    plot_sample,
    read_capture,
    read_posterior,
    route_matrices,
    source_labels,
)


class Tokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"t{token_id}" for token_id in token_ids]


def sample_arrays() -> dict[str, np.ndarray]:
    layers, tokens, heads = 2, 3, 2
    direct = np.zeros((layers, tokens, heads), dtype=np.float32)
    relay = np.zeros_like(direct)
    feedback = np.zeros_like(direct)
    predictor = np.zeros_like(direct)
    unknown = np.zeros_like(direct)
    direct[:, 2] = [[0.30, 0.10], [0.20, 0.15]]
    relay[:, 2] = [[0.20, 0.15], [0.10, 0.20]]
    feedback[:, 2] = [[0.10, 0.35], [0.40, 0.20]]
    predictor[:, 2] = 0.05
    unknown[:, 2] = [[0.10, 0.05], [0.05, 0.10]]

    # Rows are ordered by layer, then prediction.  Only positions 5 use the
    # explicit edges below; source 2 is intentionally absent in the tail.
    unknown_capacity = np.zeros((6, heads), dtype=np.float32)
    unknown_capacity[2] = [0.10, 0.20]
    unknown_capacity[5] = [0.25, 0.35]
    return {
        "token_ids": np.arange(10, 16),
        "response_start": np.asarray(3),
        "prompt_token_unit": np.array([0, 1, 1]),
        "evidence_name": np.array(["document sentence"]),
        "prediction_position": np.array([3, 4, 5]),
        "raw_route_contraction": np.array([0.2, 0.6, 0.9]),
        "takeover": np.array([0.0, 0.2, 0.8]),
        "valid": np.array([False, False, True]),
        "prompt_evidence": direct,
        "grounded_response_relay": relay,
        "unrooted_response_feedback": feedback,
        "predictor_self": predictor,
        "unknown_route": unknown,
        "graph_row_layer": np.array([0, 0, 0, 1, 1, 1]),
        "graph_row_prediction_position": np.array([3, 4, 5, 3, 4, 5]),
        "graph_edge_start": np.array([0, 0, 0, 3, 3, 3, 5]),
        "graph_edge_head": np.array([0, 0, 1, 0, 1]),
        "graph_edge_source": np.array([0, 3, 4, 1, 4]),
        "graph_edge_capacity": np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
        "graph_unknown_capacity": unknown_capacity,
    }


def write_capture(path: Path) -> None:
    np.savez(path, **sample_arrays())


def test_route_view_keeps_exact_sources_and_endpoint_free_tail(tmp_path):
    path = tmp_path / "sample.npz"
    write_capture(path)
    capture = read_capture(path)
    view = route_matrices(capture, prediction_position=5)

    assert view["account_name"] == (
        "prompt-carried evidence",
        "grounded relay",
        "unrooted feedback",
        "predictor self",
        "unknown",
    )
    np.testing.assert_allclose(view["account"][1], [0.10, 0.15, 0.35, 0.05, 0.05])
    assert view["source_position"] == (0, 1, 3, 4, None)
    assert 2 not in view["source_position"]

    endpoint = view["endpoint_share"]
    np.testing.assert_allclose(endpoint[0, [0, 2, 4]], [0.375, 0.5, 0.125])
    np.testing.assert_allclose(endpoint[3, [3, 4]], [0.7 / 1.05, 0.35 / 1.05])
    assert endpoint[:, 1].sum() > 0  # exact evidence endpoint at source 1

    labels = source_labels(
        capture,
        view["source_position"],
        Tokenizer().convert_ids_to_tokens(capture["token_ids"].tolist()),
        query_position=4,
    )
    assert labels[0].endswith("other prompt")
    assert labels[1].endswith("document sentence")
    assert labels[2].endswith("history r0")
    assert labels[3].endswith("predictor self")
    assert labels[-1] == "unknown tail\n(no endpoint)"


def test_sample_figure_reads_capture_and_optional_frozen_posterior(tmp_path):
    capture = tmp_path / "sample.npz"
    output = tmp_path / "sample.png"
    scores = tmp_path / "scores.npz"
    write_capture(capture)
    np.savez(
        scores,
        sample_id=np.array(["other", "sample", "sample", "sample"]),
        token_index=np.array([0, 2, 0, 1]),
        captured_posterior=np.array([0.0, 0.95, 0.05, 0.3]),
    )

    posterior = read_posterior(scores, "sample")
    np.testing.assert_array_equal(posterior, [0.05, 0.3, 0.95])
    result = plot_sample(
        capture,
        Tokenizer(),
        output,
        captured_posterior=posterior,
        sample_name="sample",
    )

    assert result == output
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert output.stat().st_size > 1_000
