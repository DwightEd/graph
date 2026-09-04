from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

from experiments.constraint_routing_rhythm.visualize import save_sample_figure


def sample_inputs(events: int = 9, sources: int = 14) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(13)
    return {
        "local_route": rng.random((events, sources)),
        "global_route": rng.random((events, sources)),
        "functional_reach": rng.random(events),
        "relay_capacity": rng.random(events),
        "constraint_deficit": rng.normal(size=events),
        "response_positions": np.arange(31, 31 + events),
        "response_start": 6,
        "evidence_mask": np.asarray([True] * 3 + [False] * (sources - 3)),
        "carrier_mask": np.arange(events) == 3,
    }


def test_save_sample_figure_writes_png_and_closes_figure(tmp_path: Path) -> None:
    destination = tmp_path / "sample.png"
    open_figures = set(plt.get_fignums())

    result = save_sample_figure(
        destination, title="Synthetic sample", **sample_inputs()
    )

    assert result == destination
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert destination.stat().st_size > 1_000
    assert set(plt.get_fignums()) == open_figures


def test_optional_token_labels_use_sparse_long_sequence_ticks(tmp_path: Path) -> None:
    inputs = sample_inputs(events=40)
    labels = [f"tok-{index}" for index in range(40)]

    save_sample_figure(tmp_path / "tokens.png", token_labels=labels, **inputs)

    assert (tmp_path / "tokens.png").exists()


def test_invalid_primary_events_remain_visible_as_gaps(tmp_path: Path) -> None:
    inputs = sample_inputs()
    inputs["constraint_deficit"][2] = np.nan
    inputs["relay_capacity"][-2:] = np.nan

    save_sample_figure(tmp_path / "nan-gap.png", **inputs)

    assert (tmp_path / "nan-gap.png").exists()


def test_empty_route_is_rejected(tmp_path: Path) -> None:
    inputs = sample_inputs()
    inputs["local_route"] = np.empty((0, 14))

    with pytest.raises(ValueError, match="local_route must be a non-empty 2D array"):
        save_sample_figure(tmp_path / "empty.png", **inputs)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("global_route", np.ones((8, 14)), "must have the same shape"),
        ("functional_reach", np.ones(8), "one value per response event"),
        ("relay_capacity", np.ones(8), "one value per response event"),
        ("evidence_mask", np.ones(13), "one value per source"),
        ("carrier_mask", np.ones(8), "one value per response event"),
    ],
)
def test_misaligned_shapes_are_rejected(
    tmp_path: Path, name: str, value: np.ndarray, message: str
) -> None:
    inputs = sample_inputs()
    inputs[name] = value

    with pytest.raises(ValueError, match=message):
        save_sample_figure(tmp_path / "bad.png", **inputs)
