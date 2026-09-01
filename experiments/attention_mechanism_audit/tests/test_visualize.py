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
    assert SCORE_ORDER == (
        "functional_route_collapse",
        "attention_route_collapse",
        "confidence",
    )
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
    _assert_png(tmp_path / "score_distributions.png")
    _assert_png(tmp_path / "scores_by_position.png")
    assert not (tmp_path / "samples").exists()


def test_unavailable_crossfit_does_not_plot_zero_placeholder_scores(tmp_path):
    for name in ("roc_pr.png", "score_distributions.png", "scores_by_position.png"):
        (tmp_path / name).write_bytes(b"stale")
    scores = {name: np.zeros(2) for name in SCORE_ORDER}
    report = {
        "detector": {
            "mechanism_scores_available": False,
            "reason": "at least three distinct sources are required",
        },
        "detection": {name: {"auroc": None} for name in SCORE_ORDER},
    }

    plot_population(
        np.asarray([False, True]),
        scores,
        np.asarray([0, 1]),
        np.asarray([2, 2]),
        report,
        tmp_path,
    )

    assert not (tmp_path / "roc_pr.png").exists()
    assert not (tmp_path / "score_distributions.png").exists()
    assert not (tmp_path / "scores_by_position.png").exists()
    marker = tmp_path / "DETECTION_UNAVAILABLE.txt"
    assert "three distinct sources" in marker.read_text(encoding="utf-8")


def test_one_sample_figure_is_only_an_explicit_call(tmp_path):
    layers = {
        "prompt_edge_effective_sources": np.asarray([[2.0, 1.2], [3.0, 1.5]]),
        "prompt_edge_effective_rank": np.asarray([[2.0, 1.2], [2.5, 1.4]]),
        "prompt_edge_anchor_turnover": np.asarray([[0.1, 0.0], [0.2, 0.1]]),
        "edge_evidence_share": np.asarray([[0.8, 0.2], [0.6, 0.1]]),
        "edge_other_prompt_share": np.asarray([[0.2, 0.1], [0.3, 0.1]]),
        "edge_history_share": np.asarray([[0.0, 0.6], [0.0, 0.7]]),
        "edge_self_share": np.asarray([[0.0, 0.1], [0.1, 0.1]]),
    }
    record = {
        "sample_id": "11907",
        "token_text": ["A", "B"],
        "evidence_support": np.asarray([0.5, -0.2]),
        "history_support": np.asarray([0.1, 0.8]),
        "route_interaction": np.asarray([0.0, -0.1]),
        "source_token_text": ["0:A", "1:B"],
        "source_flow": np.asarray([[0.8, 0.2], [0.1, 0.7]]),
    }
    output = tmp_path / "11907.png"

    plot_sample_dashboard(record, layers, output)

    _assert_png(output)
