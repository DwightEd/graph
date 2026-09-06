from __future__ import annotations

import inspect

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from experiments.reanchor_flow.mechanism_plot import (
    _display_edges,
    _draw_interventions,
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

    return {
        "dataset_sample_id": "synthetic-1",
        "response_start": 3,
        "layer_count": layers,
        "query_position": 8,
        "token_unit_id": np.asarray([0, 0, 1, 2, 2, 2, 2, 2, 2]),
        "selected_root_unit_id": 0,
        "edge_layer": edge_layer,
        "edge_head": np.asarray([0, 1, 2, 0, 1, 2, 0, 2], dtype=np.int16),
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_root_throughput": np.linspace(1.0, 0.2, len(edge_layer)),
        "edge_on_backbone": np.asarray(
            [True, False, True, False, True, False, True, False]
        ),
        "edge_native_functional_score": np.asarray(
            [0.8, 0.4, 0.5, -0.2, 0.35, 0.25, 0.6, -0.1]
        ),
        "edge_root_lineage_action": np.asarray(
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
        "route_row_position": np.asarray([3, 5, 6, 7, 8], dtype=np.int32),
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
