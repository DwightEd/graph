import numpy as np

from experiments.evidence_route_state.controls import (
    RouteCollapseControl,
    prompt_log_volume,
)


def record(source: str, prompt: int, volume: np.ndarray) -> dict[str, object]:
    return {"source_id": source, "prompt_length": prompt, "volume": volume}


def test_prompt_log_volume_preserves_layers_and_temporal_head_anchors():
    sources = np.array([[2.0, 4.0]])
    rank = np.array([[2.0, 1.0]])
    anchors = np.array([[[0, 0], [0, 1]]])

    volume = prompt_log_volume(sources, rank, anchors, window=2)

    probability = np.array([0.75, 0.25])
    effective_anchors = np.exp(-(probability * np.log(probability)).sum())
    expected = np.array(
        [
            [np.log(2.0) + np.log(2.0)],
            [np.log(4.0) + np.log(effective_anchors)],
        ]
    )
    np.testing.assert_allclose(volume, expected)
    assert volume.shape == (2, 1)


def nuisance_records() -> list[dict[str, object]]:
    position = np.linspace(0.1, 0.9, 5)
    return [
        record(
            "source-a",
            8,
            np.column_stack((1.0 + position + 0.05 * position**2, 2.0 - position)),
        ),
        record(
            "source-b",
            19,
            np.column_stack((1.3 + 0.5 * position, 1.7 - 0.2 * position**2)),
        ),
        record(
            "source-c",
            31,
            np.column_stack((0.8 + 0.2 * position**2, 2.2 - 0.4 * position)),
        ),
    ]


def calibration_records() -> list[dict[str, object]]:
    base = nuisance_records()
    return [
        record(f"calibration-{index}", item["prompt_length"], item["volume"] - shift)
        for index, (item, shift) in enumerate(zip(base, (0.0, 0.2, 0.7), strict=True))
    ]


def test_duplicate_records_from_one_source_do_not_change_source_equal_fit():
    nuisance = nuisance_records()
    calibration = calibration_records()
    reference = RouteCollapseControl.fit(nuisance, calibration)
    duplicated = RouteCollapseControl.fit(
        [nuisance[0], nuisance[0], *nuisance[1:]],
        [calibration[0], calibration[0], *calibration[1:]],
    )

    np.testing.assert_allclose(duplicated.coefficients, reference.coefficients)
    np.testing.assert_allclose(duplicated.scales, reference.scales)
    probe = record("probe", 14, np.full((7, 2), 1.2))
    np.testing.assert_allclose(duplicated.score(probe), reference.score(probe))


def test_lower_than_expected_route_volume_has_the_higher_collapse_score():
    control = RouteCollapseControl.fit(nuisance_records(), calibration_records())
    tokens = 6
    prompt = 17
    position = (np.arange(tokens) + 0.5) / tokens
    design = np.column_stack(
        (
            np.ones(tokens),
            position,
            position**2,
            np.full(tokens, np.log1p(prompt + tokens)),
        )
    )
    expected = design @ control.coefficients.T
    contracted = record("contracted", prompt, expected - 0.5 * control.scales)
    expanded = record("expanded", prompt, expected + 0.5 * control.scales)

    np.testing.assert_allclose(control.raw_score(contracted), 0.5)
    np.testing.assert_allclose(control.raw_score(expanded), 0.0)
    assert np.all(control.score(contracted) >= control.score(expanded))
    assert np.any(control.score(contracted) > control.score(expanded))


def test_score_uses_the_matching_position_decile_ecdf():
    tables = tuple(
        (np.array([0.25]), np.array([(index + 1) / 10])) for index in range(10)
    )
    control = RouteCollapseControl(
        coefficients=np.zeros((1, 4)),
        scales=np.ones(1),
        position_tables=tables,
    )
    scored = control.score(record("probe", 5, -np.ones((10, 1))))

    np.testing.assert_allclose(scored, np.arange(1, 11) / 10)
