from pathlib import Path

import numpy as np

from experiments.attention_mechanism_audit.visualize import (
    SCORE_LABELS,
    plot_population,
    plot_sample_dashboard,
)

SCORE_ORDER = tuple(SCORE_LABELS)


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_population_figures_use_all_token_arrays(tmp_path):
    label = np.asarray([0, 0, 1, 1], dtype=bool)
    scores = {name: np.asarray([0.0, 0.2, 0.8, 1.0]) for name in SCORE_ORDER}
    report = {"detection": {name: {"auroc": 1.0} for name in SCORE_ORDER}}

    plot_population(
        label,
        scores,
        np.asarray([0, 1, 0, 1]),
        np.asarray([2, 2, 2, 2]),
        report,
        tmp_path,
    )

    _assert_png(tmp_path / "roc_pr.png")
    _assert_png(tmp_path / "mechanism_distributions.png")
    _assert_png(tmp_path / "mechanism_by_position.png")
    assert not (tmp_path / "samples").exists()


def test_one_sample_figure_is_only_an_explicit_call(tmp_path):
    layers = {
        "routing_imbalance": np.asarray([[-1.0, 0.2], [-0.5, 0.4]]),
        "source_dispersion": np.asarray([[0.1, 0.5], [0.2, 0.7]]),
        "evidence_share": np.asarray([[0.8, 0.2], [0.6, 0.1]]),
        "response_share": np.asarray([[0.0, 0.6], [0.0, 0.8]]),
    }
    record = {
        "sample_id": "11907",
        "token_text": ["A", "B"],
        "evidence_effect": np.asarray([0.5, -0.2]),
        "response_effect": np.asarray([0.1, 0.8]),
        "source_token_text": ["0:A", "1:B"],
        "source_flow": np.asarray([[0.8, 0.2], [0.1, 0.7]]),
    }
    output = tmp_path / "11907.png"

    plot_sample_dashboard(record, layers, output)

    _assert_png(output)
