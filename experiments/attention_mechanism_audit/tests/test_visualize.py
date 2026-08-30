from pathlib import Path

import numpy as np

from experiments.attention_mechanism_audit.reporting import KEY_METRICS, ONSET_METRICS
from experiments.attention_mechanism_audit.visualize import (
    plot_population,
    plot_sample_dashboard,
)


def _record(sample_id: str, labels: list[int]) -> dict:
    tokens = len(labels)
    return {
        "sample_id": sample_id,
        "token_text": [f"t{index}" for index in range(tokens)],
        "label": labels,
        "token_metrics": {
            "message_routing_drift_mean": np.linspace(-0.1, 0.3, tokens),
            "evidence_message_effect": np.linspace(0.5, -0.2, tokens),
            "response_message_effect": np.linspace(0.2, 0.7, tokens),
        },
    }


def _report(onset_events: int) -> dict:
    summary = {
        "correct_mean": 0.2,
        "hallucinated_mean": 0.3,
        "position_matched_source_equal_difference": 0.04,
        "ci95": [0.01, 0.07],
        "sources": 12,
        "matched_cells": 30,
    }
    onset = {
        "offset": [-1, 0, 1],
        "difference_in_difference": [-0.01, 0.03, 0.02],
        "ci95_low": [-0.03, 0.01, 0.0],
        "ci95_high": [0.01, 0.05, 0.04],
        "events": [onset_events] * 3,
        "sources": [3] * 3,
    }
    keys = {metric.key for metric in (*KEY_METRICS, *ONSET_METRICS)}
    return {
        "summaries": {key: dict(summary) for key in keys},
        "matched_onset": {key: dict(onset) for key in keys},
    }


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_sample_dashboard_creates_png(tmp_path):
    record = _record("sample-1", [0, 0, 1, 1, 0, 0])
    layers, tokens = 4, len(record["label"])
    layer_metrics = {
        "routing_imbalance": np.linspace(-0.2, 0.4, layers * tokens).reshape(layers, tokens),
        "source_dispersion": np.linspace(0.3, 0.9, layers * tokens).reshape(layers, tokens),
        "evidence_share": np.full((layers, tokens), 0.35),
        "response_share": np.full((layers, tokens), 0.55),
    }
    output = tmp_path / "sample.png"

    plot_sample_dashboard(record, layer_metrics, output)

    _assert_png(output)


def test_plot_population_creates_all_available_figures(tmp_path):
    records = [_record("a", [0, 0, 1]), _record("b", [0, 1, 1])]

    plot_population(_report(onset_events=4), records, tmp_path)

    _assert_png(tmp_path / "population_effects.png")
    _assert_png(tmp_path / "sample_map.png")
    _assert_png(tmp_path / "onset_dynamics.png")


def test_plot_population_omits_unavailable_onset(tmp_path):
    plot_population(_report(onset_events=0), [_record("a", [0, 1])], tmp_path)

    _assert_png(tmp_path / "population_effects.png")
    _assert_png(tmp_path / "sample_map.png")
    assert not (tmp_path / "onset_dynamics.png").exists()


def test_plot_population_draws_splits_and_tolerates_missing_estimates(tmp_path):
    report = _report(onset_events=3)
    report["by_split"] = {
        "train": _report(onset_events=0),
        "test": _report(onset_events=0),
    }
    report["by_split"]["train"]["summaries"][KEY_METRICS[0].key][
        "position_matched_source_equal_difference"
    ] = None
    report["by_split"]["test"]["summaries"][KEY_METRICS[1].key]["ci95"] = [
        None,
        None,
    ]
    report["summaries"][KEY_METRICS[2].key]["ci95"] = None
    report["matched_onset"][ONSET_METRICS[0].key]["ci95_low"] = None

    plot_population(report, [_record("a", [0, 1])], tmp_path)

    _assert_png(tmp_path / "population_effects.png")
    _assert_png(tmp_path / "onset_dynamics.png")
