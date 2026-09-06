from __future__ import annotations

import inspect

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from experiments.reanchor_flow.mechanism_plot import (
    _display_edges,
    _draw_interventions,
    _draw_reanchor_timeline,
    _draw_reanchor_triangles,
    _draw_route,
    save_mechanism_figure,
)


def synthetic_artifact() -> dict[str, object]:
    layers, heads, rows, stages = 4, 3, 5, 4
    edge_layer = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int16)
    edge_source = np.asarray([0, 1, 3, 4, 5, 6, 7, 6], dtype=np.int32)
    edge_target = np.asarray([3, 4, 5, 6, 7, 7, 8, 8], dtype=np.int32)
    edge_origin = np.zeros((len(edge_layer), 4), dtype=np.float32)
    edge_origin[:6, 0] = np.linspace(1.0, 0.35, 6)
    edge_origin[:6, 2] = 1 - edge_origin[:6, 0]
    edge_origin[6:, 2] = 1

    rng = np.random.default_rng(7)
    transport = rng.uniform(0.01, 1.0, (layers, heads, rows, 4)).astype(np.float32)
    action = rng.normal(0, 0.2, (layers, heads, rows, 4)).astype(np.float32)
    head_integration = rng.uniform(0.1, 1.0, (layers, heads, rows, 4)).astype(
        np.float32
    )
    head_integration[..., 2] = rng.uniform(0.25, 1.0, (layers, heads, rows))
    layer_integration = rng.uniform(0.1, 1.0, (layers, rows, 4)).astype(np.float32)
    layer_integration[..., 2] = rng.uniform(0.2, 1.0, (layers, rows))
    route_position = np.asarray([3, 5, 6, 7, 8], dtype=np.int32)
    source_position = np.full((layers, heads, rows, 4), -1, dtype=np.int32)
    source_position[..., 0] = 0
    source_position[..., 1] = 1
    for slot, destination in enumerate(route_position):
        if destination - 3 > 2:
            source_position[:, :, slot, 2] = 3
        source_position[:, :, slot, 3] = max(3, destination - 1)
    source_transport = rng.uniform(0.01, 1.0, (layers, heads, rows, 4)).astype(
        np.float32
    )
    source_action = rng.normal(0, 0.2, (layers, heads, rows, 4)).astype(np.float32)
    source_attention = rng.uniform(0.01, 0.9, (layers, heads, rows, 4)).astype(
        np.float32
    )
    switch_delta = np.zeros((layers, heads, rows), dtype=np.float32)
    switch_delta[2, 1, 3] = 0.45
    switch_delta[3, 2, 4] = 0.62
    reanchor_score = np.abs(switch_delta)

    return {
        "dataset_sample_id": "synthetic-1",
        "response_start": 3,
        "layer_count": layers,
        "query_position": 8,
        "token_unit_id": np.asarray([0, 0, 1, 2, 2, 2, 2, 2, 2]),
        "unit_name": np.asarray(["support passage", "prompt", "response"]),
        "selected_root_unit_id": 0,
        "edge_layer": edge_layer,
        "edge_head": np.asarray([0, 1, 2, 0, 1, 2, 0, 2], dtype=np.int16),
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_root_throughput": np.linspace(1.0, 0.2, len(edge_layer)),
        "edge_on_backbone": np.asarray(
            [True, False, True, False, True, False, True, False]
        ),
        "edge_in_frozen_corridor": np.asarray(
            [True, True, True, False, True, True, True, False]
        ),
        "edge_native_functional_score": np.asarray(
            [0.8, 0.4, 0.5, -0.2, 0.35, 0.25, 0.6, -0.1]
        ),
        "edge_evidence_lineage_action": np.asarray(
            [0.8, 0.0, 0.35, -0.02, 0.2, 0.05, 0.12, -0.01]
        ),
        "route_edge_origin": edge_origin,
        "route_node_origin": rng.uniform(0.0, 1.0, (layers + 1, 9, 4)).astype(
            np.float32
        ),
        "route_node_throughput": rng.uniform(0.0, 1.0, (layers + 1, 9)).astype(
            np.float32
        ),
        "backbone_node_layer": np.arange(layers + 1, dtype=np.int16),
        "backbone_node_position": np.asarray([0, 3, 5, 7, 8], dtype=np.int32),
        "backbone_node_throughput": np.asarray([0.8, 0.7, 0.6, 0.5, 1.0]),
        "backbone_node_origin": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0, 0.2],
                [0.65, 0.0, 0.15, 0.2],
                [0.5, 0.0, 0.3, 0.2],
                [0.4, 0.0, 0.4, 0.2],
            ]
        ),
        "backbone_step_throughput": np.asarray([0.8, 0.7, 0.6, 0.5]),
        "backbone_step_is_residual": np.asarray([False, False, False, False]),
        "route_row_position": route_position,
        "route_head_transport": transport,
        "route_head_action": action,
        "route_head_integration": head_integration,
        "route_layer_integration": layer_integration,
        "route_stage_position": np.asarray([3, 5, 7, 8], dtype=np.int32),
        "route_stage_displacement": rng.uniform(0.01, 1.0, (layers, stages, 3)).astype(
            np.float32
        ),
        "route_stage_action": rng.normal(0, 0.3, (layers, stages, 3)).astype(
            np.float32
        ),
        "route_module_vector_cosine": rng.uniform(-1, 1, (layers, stages)),
        "route_module_functional_agreement": rng.uniform(0, 1, (layers, stages)),
        "route_state_continuity": rng.uniform(-1, 1, (layers, stages)),
        "local_window": 2,
        "reanchor_bucket_name": np.asarray(
            [
                "prompt_evidence",
                "other_prompt",
                "remote_response",
                "recent_local",
            ]
        ),
        "reanchor_source_position": source_position,
        "reanchor_source_transport": source_transport,
        "reanchor_source_attention": source_attention,
        "reanchor_source_downstream_action": source_action,
        "reanchor_switch_delta": switch_delta,
        "reanchor_score": reanchor_score,
        "reanchor_candidate_layer": np.asarray([2, 3], dtype=np.int16),
        "reanchor_candidate_head": np.asarray([1, 2], dtype=np.int16),
        "reanchor_candidate_position": np.asarray([7, 8], dtype=np.int32),
        "reanchor_candidate_source_kind": np.asarray(
            ["remote_response", "prompt_evidence"]
        ),
        "reanchor_candidate_source_position": np.asarray([3, 0], dtype=np.int32),
        "reanchor_candidate_source_unit": np.asarray([2, 0], dtype=np.int32),
        "reanchor_candidate_previous_local_source_position": np.asarray(
            [6, 7], dtype=np.int32
        ),
        "reanchor_candidate_previous_local_source_unit": np.asarray(
            [2, 2], dtype=np.int32
        ),
        "reanchor_candidate_switch_delta": np.asarray([0.45, 0.62]),
        "reanchor_candidate_relative_anchor_rise": np.asarray([0.5, 0.7]),
        "reanchor_candidate_relative_local_fall": np.asarray([0.4, 0.6]),
        "reanchor_candidate_source_evidence_fraction": np.asarray([0.8, 1.0]),
        "reanchor_candidate_source_downstream_action": np.asarray([-0.05, 0.3]),
        "reanchor_candidate_source_immediate_action": np.asarray([np.nan, 0.3]),
        "reanchor_candidate_long_range_downstream_action": np.asarray([-0.2, 0.42]),
        "reanchor_candidate_long_range_immediate_action": np.asarray([np.nan, 0.42]),
        "reanchor_candidate_current_target_match": np.asarray([False, True]),
        "reanchor_candidate_selected_root_integration_coherence": np.asarray(
            [np.nan, 0.85]
        ),
        "native_margin": 2.1,
        "root_cut_margin": 0.7,
        "root_value_effect": 1.4,
        "selected_root_causal_score": 0.55,
        "selected_root_confirmed": True,
        "corridor_necessity": 1.2,
        "corridor_conditional_rescue": 1.1,
        "corridor_mediated_rescue": 0.95,
        "corridor_restoration_valid": True,
        "corridor_confirmed": True,
        "carrier_layer": np.asarray([2], dtype=np.int16),
        "carrier_position": np.asarray([7], dtype=np.int32),
        "carrier_route_throughput": np.asarray([0.8]),
        "carrier_necessity": np.asarray([0.75]),
        "carrier_rescue": np.asarray([0.72]),
        "carrier_blocked_rescue": np.asarray([0.08]),
        "carrier_mediated_rescue": np.asarray([0.64]),
        "carrier_confirmed": np.asarray([True]),
        "full_chain_confirmed": True,
        "hub_layer": np.asarray([2], dtype=np.int16),
        "hub_position": np.asarray([7], dtype=np.int32),
    }


def test_save_mechanism_figure_is_label_free_and_writes_png(tmp_path) -> None:
    parameters = tuple(inspect.signature(save_mechanism_figure).parameters)
    assert parameters == ("path", "artifact", "token_labels")
    assert "label" not in parameters

    destination = tmp_path / "target_mechanism.png"
    labels = [f"tok-{index}" for index in range(9)]
    result = save_mechanism_figure(destination, synthetic_artifact(), labels)

    assert result == destination
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert destination.stat().st_size > 10_000


def test_schema_artifact_source_tracks_render_without_candidates(tmp_path) -> None:
    from experiments.reanchor_flow.artifact_payload import native_audit_arrays
    from experiments.reanchor_flow.tests.test_subset_artifacts import (
        _fixture,
        _metadata,
    )

    world, audit, _ = _fixture()
    artifact = native_audit_arrays(world, audit, _metadata())
    destination = tmp_path / "empty_reanchor.png"

    save_mechanism_figure(destination, artifact)

    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_reanchor_timeline_keeps_every_layer_head_track() -> None:
    from matplotlib import pyplot as plt

    artifact = synthetic_artifact()
    figure, axis = plt.subplots()
    try:
        _draw_reanchor_timeline(axis, figure, artifact, None)
        raster = np.asarray(axis.images[0].get_array())
        assert raster.shape[:2] == (4 * 3, 5)
        offsets = np.concatenate(
            [collection.get_offsets() for collection in axis.collections]
        )
        assert any(np.allclose(offset, [3, 2 * 3 + 1]) for offset in offsets)
        assert any(np.allclose(offset, [4, 3 * 3 + 2]) for offset in offsets)
        assert "no mean" in axis.get_title(loc="left")
    finally:
        plt.close(figure)


def test_reanchor_triangles_show_sources_and_separate_action_semantics() -> None:
    from matplotlib import pyplot as plt

    artifact = synthetic_artifact()
    figure, axes = plt.subplots(1, 4)
    try:
        _draw_reanchor_triangles(axes, figure, artifact, None)
        first_text = "\n".join(item.get_text() for item in axes[0].texts)
        second_text = "\n".join(item.get_text() for item in axes[1].texts)
        assert "winner-source downstream action to audited q" in first_text
        assert "all-long-range downstream action" in first_text
        assert "winner-source immediate action" in second_text
        assert "all-long-range immediate action" in second_text
        assert "support passage" in second_text
        assert "selected-root integration=not applicable" in first_text
        first_offsets = np.concatenate(
            [collection.get_offsets() for collection in axes[0].collections]
        )
        second_offsets = np.concatenate(
            [collection.get_offsets() for collection in axes[1].collections]
        )
        assert any(np.allclose(offset, [7, 3]) for offset in first_offsets)
        assert any(np.allclose(offset, [8, 0]) for offset in second_offsets)
    finally:
        plt.close(figure)


def test_display_edges_never_truncates_backbone() -> None:
    throughput = np.linspace(1.0, 0.01, 80)
    on_backbone = np.zeros(80, dtype=bool)
    on_backbone[35:75] = True

    selected = _display_edges(throughput, on_backbone)

    assert selected.tolist() == list(range(35, 75))


def test_display_edges_fills_budget_after_backbone() -> None:
    throughput = np.linspace(1.0, 0.01, 80)
    on_backbone = np.zeros(80, dtype=bool)
    on_backbone[[10, 20, 30]] = True

    selected = _display_edges(throughput, on_backbone)

    assert len(selected) == 32
    assert {10, 20, 30}.issubset(selected.tolist())


def test_display_edges_prioritizes_frozen_corridor_before_other_routes() -> None:
    throughput = np.linspace(1.0, 0.01, 40)
    on_backbone = np.zeros(40, dtype=bool)
    on_backbone[39] = True
    in_frozen_corridor = np.zeros(40, dtype=bool)
    in_frozen_corridor[34:39] = True

    selected = _display_edges(throughput, on_backbone, in_frozen_corridor)

    assert len(selected) == 32
    assert selected[:6].tolist() == [39, 34, 35, 36, 37, 38]
    assert not set(range(34, 40)).difference(selected.tolist())


def test_display_edges_never_truncates_frozen_corridor() -> None:
    throughput = np.linspace(1.0, 0.01, 48)
    on_backbone = np.zeros(48, dtype=bool)
    on_backbone[47] = True
    in_frozen_corridor = np.zeros(48, dtype=bool)
    in_frozen_corridor[8:47] = True

    selected = _display_edges(throughput, on_backbone, in_frozen_corridor)

    assert len(selected) == 40
    assert set(range(8, 48)) == set(selected.tolist())


def test_discovery_route_shows_frozen_candidate_as_hollow_until_confirmed() -> None:
    from matplotlib import pyplot as plt

    artifact = synthetic_artifact()
    artifact.update(
        {
            "analysis_stage": "discovery",
            "hub_layer": np.asarray([2], dtype=np.int16),
            "hub_position": np.asarray([6], dtype=np.int32),
            "carrier_layer": np.asarray([2], dtype=np.int16),
            "carrier_position": np.asarray([6], dtype=np.int32),
            "carrier_confirmed": np.asarray([False]),
        }
    )
    figure, axis = plt.subplots()
    try:
        _draw_route(axis, figure, artifact, None)
        candidate = [
            collection
            for collection in axis.collections
            if np.allclose(collection.get_offsets(), [[2.0, 6.0]])
        ]
        assert len(candidate) == 1
        assert candidate[0].get_facecolors().size == 0
    finally:
        plt.close(figure)

    artifact["carrier_confirmed"] = np.asarray([True])
    figure, axis = plt.subplots()
    try:
        _draw_route(axis, figure, artifact, None)
        carrier = [
            collection
            for collection in axis.collections
            if np.allclose(collection.get_offsets(), [[2.0, 6.0]])
        ]
        assert len(carrier) == 1
        assert carrier[0].get_facecolors().size > 0
    finally:
        plt.close(figure)


def test_route_marks_target_at_true_final_layer() -> None:
    from matplotlib import pyplot as plt

    artifact = synthetic_artifact()
    artifact["edge_root_throughput"] = np.asarray(
        artifact["edge_root_throughput"]
    ).copy()
    artifact["edge_root_throughput"][6:] = 0
    figure, axis = plt.subplots()
    try:
        _draw_route(axis, figure, artifact, None)
        target = [
            collection
            for collection in axis.collections
            if np.allclose(collection.get_offsets(), [[4.0, 8.0]])
        ]
        assert len(target) == 1
        assert len(target[0].get_paths()[0].vertices) == 11
    finally:
        plt.close(figure)


def test_route_does_not_mark_pure_residual_continuation_as_hub() -> None:
    from matplotlib import pyplot as plt

    artifact = synthetic_artifact()
    artifact["edge_source"] = np.asarray(artifact["edge_source"]).copy()
    artifact["edge_source"][6] = 3
    artifact["edge_on_backbone"] = np.asarray(
        [True, False, False, False, False, False, True, False]
    )
    artifact["backbone_node_position"] = np.asarray([0, 3, 3, 3, 8])
    artifact["backbone_step_is_residual"] = np.asarray([False, True, True, False])
    figure, axis = plt.subplots()
    try:
        _draw_route(axis, figure, artifact, None)
        continuation = [
            collection
            for collection in axis.collections
            if np.allclose(collection.get_offsets(), [[2.0, 3.0]])
        ]
        assert len(continuation) == 1
        assert len(continuation[0].get_paths()[0].vertices) == 26
    finally:
        plt.close(figure)


def test_intervention_panel_displays_confirmation_chain() -> None:
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    try:
        _draw_interventions(axis, synthetic_artifact())
        text = "\n".join(item.get_text() for item in axis.texts)
        assert "selected_root_confirmed: yes" in text
        assert "corridor_restoration_valid: yes" in text
        assert "corridor_confirmed: yes" in text
        assert "carrier_confirmed[selected=0]: yes" in text
        assert "full_chain_confirmed: yes" in text
    finally:
        plt.close(figure)


def test_discovery_panel_marks_confirmation_not_run_without_nan_bars() -> None:
    from matplotlib import pyplot as plt

    artifact = synthetic_artifact()
    artifact.update(
        {
            "subset_audit_schema": 3,
            "analysis_stage": "discovery",
            "selected_root_evaluated": False,
            "corridor_evaluated": False,
            "carrier_evaluated_count": 0,
            "full_chain_evaluated": False,
            "selected_root_causal_score": np.nan,
            "corridor_necessity": np.nan,
            "corridor_conditional_rescue": np.nan,
            "corridor_mediated_rescue": np.nan,
            "carrier_necessity": np.asarray([np.nan]),
            "carrier_mediated_rescue": np.asarray([np.nan]),
        }
    )
    figure, axis = plt.subplots()
    try:
        _draw_interventions(axis, artifact)
        text = "\n".join(item.get_text() for item in axis.texts)
        assert "Exact confirmation not run" in text
        assert "selected_root_confirmed: not run" in text
        assert "corridor_confirmed: not run" in text
        assert "carrier_confirmed: not run" in text
        assert "full_chain_confirmed: not run" in text
        assert "not closed" not in axis.get_title()
        assert not axis.patches
    finally:
        plt.close(figure)
