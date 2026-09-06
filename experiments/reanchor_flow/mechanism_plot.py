"""Label-free mechanism figure for one native evidence-to-target audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

EVIDENCE = 0
RESPONSE = 2
MAX_DISPLAY_EDGES = 32


def _array(artifact: Mapping[str, object], name: str, dtype=float) -> np.ndarray:
    return np.asarray(artifact[name], dtype=dtype)


def _target_slot(position: np.ndarray, target: int) -> int:
    exact = np.flatnonzero(position == target)
    if len(exact) != 1:
        raise ValueError("target position is absent from the mechanism ledger")
    return int(exact[0])


def _display_edges(
    throughput: np.ndarray,
    on_backbone: np.ndarray,
) -> np.ndarray:
    """Always show the connected backbone, then fill the display budget."""

    positive = np.isfinite(throughput) & (throughput > 0)
    backbone = np.flatnonzero(positive & on_backbone)
    marginal = np.flatnonzero(positive & ~on_backbone)
    remaining = max(0, MAX_DISPLAY_EDGES - len(backbone))
    order = np.argsort(-throughput[marginal], kind="stable")
    return np.concatenate((backbone, marginal[order[:remaining]]))


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(float(finite.max()) if len(finite) else 0.0, 1e-12)


def _rgba(values: np.ndarray, strength: np.ndarray):
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    limit = _symmetric_limit(values)
    colors = colormaps["coolwarm"](Normalize(-limit, limit)(values))
    scale = np.asarray(strength, dtype=float)
    finite = scale[np.isfinite(scale) & (scale > 0)]
    denominator = float(np.quantile(finite, 0.95)) if len(finite) else 1.0
    alpha = np.clip(np.nan_to_num(scale / denominator), 0.0, 1.0)
    colors[..., 3] = 0.12 + 0.88 * np.sqrt(alpha)
    return colors, limit


def _token_tick(position: int, token_labels: Sequence[str] | None) -> str:
    if token_labels is None or not 0 <= position < len(token_labels):
        return str(position)
    token = str(token_labels[position]).replace("\n", "↵").replace(" ", "·")
    return f"{position}:{token[:10]}"


def _draw_route(axis, figure, artifact, token_labels) -> None:
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    layer = _array(artifact, "edge_layer", int)
    head = _array(artifact, "edge_head", int)
    source = _array(artifact, "edge_source", int)
    target = _array(artifact, "edge_target", int)
    throughput = _array(artifact, "edge_root_throughput")
    on_backbone = _array(artifact, "edge_on_backbone", bool)
    action = _array(artifact, "edge_root_lineage_action")
    selected = _display_edges(throughput, on_backbone)
    query = int(np.asarray(artifact["query_position"]).item())
    layer_count = int(np.asarray(artifact["layer_count"]).item())
    token_unit = _array(artifact, "token_unit_id", int)
    root_unit = int(np.asarray(artifact["selected_root_unit_id"]).item())
    response_start = int(np.asarray(artifact["response_start"]).item())
    root_positions = set(
        np.flatnonzero(
            (token_unit == root_unit) & (np.arange(len(token_unit)) < response_start)
        ).tolist()
    )

    if not len(selected):
        axis.text(0.5, 0.5, "No positive root-conditioned route", ha="center")
        axis.set_axis_off()
        return

    maximum = float(throughput[selected].max())
    action_limit = _symmetric_limit(action[selected])
    color_norm = Normalize(-action_limit, action_limit)
    cmap = colormaps["coolwarm"]
    nodes: dict[tuple[int, int], np.ndarray] = {}
    node_mass: dict[tuple[int, int], float] = {}
    position_change_nodes: set[tuple[int, int]] = set()
    for edge in selected:
        start = (int(layer[edge]), int(source[edge]))
        end = (int(layer[edge]) + 1, int(target[edge]))
        for node in (start, end):
            nodes.setdefault(node, np.zeros(4))
            node_mass[node] = node_mass.get(node, 0.0) + float(throughput[edge])
        if source[edge] != target[edge]:
            position_change_nodes.update((start, end))

    backbone_layer = _array(artifact, "backbone_node_layer", int)
    backbone_position = _array(artifact, "backbone_node_position", int)
    backbone_mass = _array(artifact, "backbone_node_throughput")
    backbone_origin = _array(artifact, "backbone_node_origin")
    for depth, position, mass, carried in zip(
        backbone_layer,
        backbone_position,
        backbone_mass,
        backbone_origin,
        strict=True,
    ):
        node = (int(depth), int(position))
        nodes[node] = np.asarray(carried, dtype=float) * float(mass)
        node_mass[node] = max(node_mass.get(node, 0.0), float(mass))

    carrier_nodes = set()
    if "carrier_layer" in artifact and "carrier_position" in artifact:
        carrier_layer = _array(artifact, "carrier_layer", int)
        carrier_position = _array(artifact, "carrier_position", int)
        confirmed = np.asarray(
            artifact.get("carrier_confirmed", np.zeros(len(carrier_layer))), dtype=bool
        )
        carrier_nodes = {
            (int(depth), int(position))
            for depth, position, keep in zip(
                carrier_layer, carrier_position, confirmed, strict=True
            )
            if keep
        }

    node_origin = _array(artifact, "route_node_origin")
    node_throughput = _array(artifact, "route_node_throughput")
    for node in carrier_nodes:
        nodes.setdefault(node, np.zeros(4))
        node_mass.setdefault(node, 0.0)
    for depth, position in nodes:
        if 0 <= depth <= layer_count and 0 <= position < node_origin.shape[1]:
            nodes[(depth, position)] = (
                node_origin[depth, position] * node_throughput[depth, position]
            )
            node_mass[(depth, position)] = max(
                node_mass.get((depth, position), 0.0),
                float(node_throughput[depth, position]),
            )

    for position in sorted({node[1] for node in nodes}):
        depths = [node[0] for node in nodes if node[1] == position]
        axis.plot(
            [min(depths), max(depths)],
            [position, position],
            color="0.75",
            linewidth=0.6,
            linestyle=":",
            zorder=0,
        )

    residual_step = _array(artifact, "backbone_step_is_residual", bool)
    step_throughput = _array(artifact, "backbone_step_throughput")
    for index in np.flatnonzero(residual_step):
        start = (int(backbone_layer[index]), int(backbone_position[index]))
        end = (
            int(backbone_layer[index + 1]),
            int(backbone_position[index + 1]),
        )
        weight = min(1.0, float(step_throughput[index]) / maximum)
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.9 + 2.2 * weight,
                color="#264653",
                alpha=0.75,
                linestyle=":",
                zorder=1.5,
            )
        )

    label_count = 0
    for edge in selected:
        start = (int(layer[edge]), int(source[edge]))
        end = (int(layer[edge]) + 1, int(target[edge]))
        weight = float(throughput[edge]) / maximum
        is_backbone = bool(on_backbone[edge])
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=(1.0 + 3.0 * weight) if is_backbone else (0.3 + weight),
            color=cmap(color_norm(action[edge])),
            alpha=(0.72 + 0.25 * weight) if is_backbone else (0.1 + 0.2 * weight),
            connectionstyle="arc3,rad=0.04",
            zorder=1.6 if is_backbone else 1,
        )
        axis.add_patch(patch)
        if is_backbone or label_count < 10:
            axis.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2,
                f"h{head[edge]}",
                fontsize=5.5,
                color="0.25",
            )
            label_count += 1

    for node, carried in nodes.items():
        total = float(carried.sum())
        evidence_share = float(carried[EVIDENCE] / total) if total > 0 else 0.0
        is_root = node[0] == 0 and node[1] in root_positions
        is_target = node == (layer_count, query)
        message_internal = node in position_change_nodes and node[0] not in (
            0,
            layer_count,
        )
        is_hub = (
            not is_root
            and not is_target
            and (evidence_share > 0 and message_internal or node in carrier_nodes)
        )
        marker = "*" if is_target else "D" if is_hub else "s" if is_root else "o"
        color = (
            "#f2c14e"
            if is_target
            else "#2a9d8f"
            if is_hub
            else "#457b9d"
            if is_root
            else "#b8b8b8"
        )
        axis.scatter(
            node[0],
            node[1],
            s=35 + 75 * node_mass[node] / maximum,
            marker=marker,
            color=color,
            edgecolor="black" if node in carrier_nodes else "white",
            linewidth=1.5 if node in carrier_nodes else 0.5,
            zorder=2,
        )

    shown_positions = sorted({query} | {node[1] for node in carrier_nodes})
    if len(shown_positions) < 10:
        by_mass = sorted(
            {node[1] for node in nodes},
            key=lambda position: (
                -sum(mass for node, mass in node_mass.items() if node[1] == position)
            ),
        )
        shown_positions.extend(by_mass[: 10 - len(shown_positions)])
    shown_positions = sorted(set(shown_positions))
    axis.set_yticks(
        shown_positions,
        [_token_tick(position, token_labels) for position in shown_positions],
        fontsize=7,
    )
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("Token position")
    axis.set_title("A  Selected-root backbone with marginal routes", loc="left")
    axis.grid(axis="x", alpha=0.15)
    axis.legend(
        handles=[
            Line2D(
                [], [], marker="s", linestyle="", label="source root", color="#457b9d"
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                label="root-lineage hub candidate",
                color="#2a9d8f",
            ),
            Line2D(
                [],
                [],
                marker="D",
                markerfacecolor="#2a9d8f",
                markeredgecolor="black",
                linestyle="",
                label="native carrier",
            ),
            Line2D([], [], marker="*", linestyle="", label="target", color="#f2c14e"),
        ],
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper left",
    )
    scalar = figure.colorbar(
        __import__("matplotlib.cm").cm.ScalarMappable(norm=color_norm, cmap=cmap),
        ax=axis,
        fraction=0.035,
        pad=0.02,
    )
    scalar.set_label("selected-root-lineage signed action", fontsize=8)


def _draw_heads(axis, figure, artifact) -> None:
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    position = _array(artifact, "route_row_position", int)
    query = int(np.asarray(artifact["query_position"]).item())
    slot = _target_slot(position, query)
    transport = _array(artifact, "route_head_transport")[:, :, slot]
    action = _array(artifact, "route_head_action")[:, :, slot]
    values = np.concatenate((action[..., EVIDENCE], action[..., RESPONSE]), axis=1)
    strength = np.concatenate(
        (transport[..., EVIDENCE], transport[..., RESPONSE]), axis=1
    )
    colors, limit = _rgba(values, strength)
    axis.imshow(colors, origin="lower", aspect="auto", interpolation="nearest")
    layers, heads = action.shape[:2]
    axis.axvline(heads - 0.5, color="black", linewidth=0.8)
    ticks = [0, heads - 1, heads, 2 * heads - 1]
    axis.set_xticks(ticks, ["E:h0", f"E:h{heads - 1}", "R:h0", f"R:h{heads - 1}"])
    axis.set_yticks(np.arange(0, layers, max(1, layers // 8)))
    axis.set_xlabel("Evidence lineage heads | response-origin heads")
    axis.set_ylabel("Layer")
    axis.set_title(
        "B  Head-resolved target use (hue=action, opacity=transport)",
        loc="left",
    )
    figure.colorbar(
        ScalarMappable(norm=Normalize(-limit, limit), cmap=colormaps["coolwarm"]),
        ax=axis,
        fraction=0.035,
        pad=0.02,
        label="signed action",
    )


def _layer_series(value, stage_slot: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError("layer diagnostic must have one layer and one position axis")
    return array[:, stage_slot]


def _draw_integration(heatmap, continuity, figure, artifact) -> None:
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    query = int(np.asarray(artifact["query_position"]).item())
    row_position = _array(artifact, "route_row_position", int)
    stage_position = _array(artifact, "route_stage_position", int)
    row_slot = _target_slot(row_position, query)
    stage_slot = _target_slot(stage_position, query)
    displacement = _array(artifact, "route_stage_displacement")[:, stage_slot].T
    action = _array(artifact, "route_stage_action")[:, stage_slot].T
    colors, limit = _rgba(action, np.log1p(np.clip(displacement, 0, None)))
    heatmap.imshow(colors, origin="lower", aspect="auto", interpolation="nearest")
    heatmap.set_yticks(range(3), ["residual", "attention", "MLP"])
    heatmap.set_xticks([])
    heatmap.set_title(
        "C  Source-conditioned integration (hue=action, opacity=displacement)",
        loc="left",
    )
    figure.colorbar(
        ScalarMappable(norm=Normalize(-limit, limit), cmap=colormaps["coolwarm"]),
        ax=heatmap,
        fraction=0.035,
        pad=0.02,
        label="gradient · source-cut Δ",
    )

    layer_integration = _array(artifact, "route_layer_integration")
    series = {
        "A–MLP vector cosine": _layer_series(
            artifact["route_module_vector_cosine"], stage_slot
        ),
        "A–MLP functional agreement": _layer_series(
            artifact["route_module_functional_agreement"], stage_slot
        ),
        "state continuity": _layer_series(
            artifact["route_state_continuity"], stage_slot
        ),
        "message coherence": layer_integration[:, row_slot, 2],
    }
    for name, values in series.items():
        continuity.plot(np.arange(len(values)), values, linewidth=1.1, label=name)
    continuity.axhline(0, color="black", linewidth=0.6)
    continuity.set_ylim(-1.05, 1.05)
    continuity.set_xlabel("Layer")
    continuity.set_ylabel("alignment")
    continuity.grid(alpha=0.15)
    continuity.legend(frameon=False, fontsize=6.5, ncol=2)


def _number(value) -> float:
    array = np.asarray(value, dtype=float)
    return float(array.item())


def _flag(value) -> bool:
    return bool(np.asarray(value, dtype=bool).item())


def _carrier_index(artifact: Mapping[str, object]) -> int | None:
    if "carrier_necessity" not in artifact:
        return None
    necessity = np.asarray(artifact["carrier_necessity"], dtype=float).reshape(-1)
    if not len(necessity):
        return None
    confirmed = np.asarray(
        artifact.get("carrier_confirmed", np.zeros(len(necessity))), dtype=bool
    ).reshape(-1)
    throughput = np.asarray(
        artifact.get("carrier_route_throughput", np.ones(len(necessity))), dtype=float
    ).reshape(-1)
    eligible = np.flatnonzero(confirmed)
    pool = eligible if len(eligible) else np.arange(len(necessity))
    return int(pool[np.argmax(throughput[pool])])


def _draw_interventions(axis, artifact) -> None:
    native = _number(artifact["native_margin"])
    root_cut = _number(artifact["root_cut_margin"])
    root_effect = _number(artifact.get("root_value_effect", native - root_cut))
    names = ["root value effect"]
    values = [root_effect]
    for label, field in (
        ("root causal min", "selected_root_causal_score"),
        ("corridor necessity", "corridor_necessity"),
        ("corridor rescue", "corridor_conditional_rescue"),
        ("corridor mediated", "corridor_mediated_rescue"),
    ):
        if field in artifact:
            names.append(label)
            values.append(_number(artifact[field]))

    carrier = _carrier_index(artifact)
    carrier_confirmed = False
    if carrier is not None:
        necessity = np.asarray(artifact["carrier_necessity"], dtype=float).reshape(-1)
        rescue = np.asarray(
            artifact.get("carrier_rescue", np.full(len(necessity), np.nan)),
            dtype=float,
        ).reshape(-1)
        if "carrier_mediated_rescue" in artifact:
            mediated = np.asarray(
                artifact["carrier_mediated_rescue"], dtype=float
            ).reshape(-1)
        else:
            blocked = np.asarray(
                artifact.get("carrier_blocked_rescue", np.zeros(len(necessity))),
                dtype=float,
            ).reshape(-1)
            mediated = rescue - blocked
        names.extend(("carrier necessity", "carrier mediated"))
        values.extend((float(necessity[carrier]), float(mediated[carrier])))
        confirmed = np.asarray(
            artifact.get("carrier_confirmed", np.zeros(len(necessity))), dtype=bool
        ).reshape(-1)
        carrier_confirmed = bool(confirmed[carrier])

    values_array = np.asarray(values)
    order = np.arange(len(values_array))
    colors = np.where(values_array >= 0, "#457b9d", "#e76f51")
    axis.barh(order, values_array, color=colors, alpha=0.85)
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set_yticks(order, names)
    axis.invert_yaxis()
    axis.set_xlabel("Exact target-margin effect")
    full_chain_confirmed = _flag(artifact.get("full_chain_confirmed", False))
    chain_state = "confirmed" if full_chain_confirmed else "not closed"
    axis.set_title(
        f"D  Exact cut / patch / block ladder · chain {chain_state}", loc="left"
    )
    status = (
        ("selected_root_confirmed", _flag(artifact["selected_root_confirmed"])),
        (
            "corridor_restoration_valid",
            _flag(artifact["corridor_restoration_valid"]),
        ),
        ("corridor_confirmed", _flag(artifact["corridor_confirmed"])),
        (f"carrier_confirmed[selected={carrier}]", carrier_confirmed),
        ("full_chain_confirmed", full_chain_confirmed),
    )
    axis.text(
        0.99,
        0.98,
        "\n".join(f"{name}: {'yes' if value else 'no'}" for name, value in status),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#edf7ed" if full_chain_confirmed else "#fff3e6",
            "edgecolor": "0.7",
        },
    )
    axis.text(
        0.99,
        0.02,
        f"native margin={native:.3g}\nroot-cut margin={root_cut:.3g}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axis.grid(axis="x", alpha=0.15)


def save_mechanism_figure(
    path: str | Path,
    artifact: Mapping[str, object],
    token_labels: Sequence[str] | None = None,
) -> Path:
    """Render one target audit without reading or displaying correctness labels."""

    from matplotlib import pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(16, 10), layout="constrained")
    grid = figure.add_gridspec(2, 2)
    route_axis = figure.add_subplot(grid[0, 0])
    head_axis = figure.add_subplot(grid[0, 1])
    integration_grid = grid[1, 0].subgridspec(2, 1, height_ratios=(2.3, 1))
    stage_axis = figure.add_subplot(integration_grid[0])
    continuity_axis = figure.add_subplot(integration_grid[1])
    intervention_axis = figure.add_subplot(grid[1, 1])
    try:
        _draw_route(route_axis, figure, artifact, token_labels)
        _draw_heads(head_axis, figure, artifact)
        _draw_integration(stage_axis, continuity_axis, figure, artifact)
        _draw_interventions(intervention_axis, artifact)
        sample = str(np.asarray(artifact.get("dataset_sample_id", "sample")).item())
        query = int(np.asarray(artifact["query_position"]).item())
        figure.suptitle(f"Mechanism audit · {sample} · predictor q={query}")
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
