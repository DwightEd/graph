"""Label-free mechanism figure for one native evidence-to-target audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

EVIDENCE = 0
MAX_DISPLAY_EDGES = 32
MAX_REANCHOR_PANELS = 4

SOURCE_MARKERS = {
    "prompt_evidence": "s",
    "other_prompt": "X",
    "remote_response": "D",
    "recent_local": "o",
}


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
    in_frozen_corridor: np.ndarray | None = None,
) -> np.ndarray:
    """Show backbone, then frozen-corridor, then other throughput edges."""

    if in_frozen_corridor is None:
        in_frozen_corridor = np.zeros_like(on_backbone, dtype=bool)
    positive = np.isfinite(throughput) & (throughput > 0)
    backbone = np.flatnonzero(positive & on_backbone)
    corridor = np.flatnonzero(positive & in_frozen_corridor & ~on_backbone)
    marginal = np.flatnonzero(positive & ~on_backbone & ~in_frozen_corridor)
    corridor_order = np.argsort(-throughput[corridor], kind="stable")
    corridor = corridor[corridor_order]
    remaining = max(0, MAX_DISPLAY_EDGES - len(backbone) - len(corridor))
    marginal_order = np.argsort(-throughput[marginal], kind="stable")
    return np.concatenate((backbone, corridor, marginal[marginal_order[:remaining]]))


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
    in_frozen_corridor = np.asarray(
        artifact.get("edge_in_frozen_corridor", np.zeros(len(layer))), dtype=bool
    )
    action = _array(artifact, "edge_evidence_lineage_action")
    selected = _display_edges(throughput, on_backbone, in_frozen_corridor)
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

    hub_nodes = set()
    if "hub_layer" in artifact and "hub_position" in artifact:
        hub_nodes = set(
            zip(
                _array(artifact, "hub_layer", int).tolist(),
                _array(artifact, "hub_position", int).tolist(),
                strict=True,
            )
        )

    carrier_nodes = set()
    if "carrier_layer" in artifact and "carrier_position" in artifact:
        carrier_layer = _array(artifact, "carrier_layer", int)
        carrier_position = _array(artifact, "carrier_position", int)
        confirmed = np.asarray(
            artifact.get("carrier_confirmed", np.zeros(len(carrier_layer))), dtype=bool
        )
        evaluated = int(
            np.asarray(
                artifact.get("carrier_evaluated_count", len(carrier_layer))
            ).item()
        )
        carrier_nodes = {
            (int(depth), int(position))
            for index, (depth, position, keep) in enumerate(
                zip(carrier_layer, carrier_position, confirmed, strict=True)
            )
            if index < evaluated and keep
        }

    node_origin = _array(artifact, "route_node_origin")
    node_throughput = _array(artifact, "route_node_throughput")
    for node in hub_nodes | carrier_nodes:
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
        is_corridor = bool(in_frozen_corridor[edge]) and not is_backbone
        if is_backbone:
            linewidth = 1.0 + 3.0 * weight
            alpha = 0.72 + 0.25 * weight
            linestyle = "-"
            zorder = 1.8
        elif is_corridor:
            linewidth = 0.7 + 1.8 * weight
            alpha = 0.4 + 0.35 * weight
            linestyle = "--"
            zorder = 1.4
        else:
            linewidth = 0.3 + weight
            alpha = 0.1 + 0.2 * weight
            linestyle = "-"
            zorder = 1
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=linewidth,
            color=cmap(color_norm(action[edge])),
            alpha=alpha,
            linestyle=linestyle,
            connectionstyle="arc3,rad=0.04",
            zorder=zorder,
        )
        axis.add_patch(patch)
        if is_backbone or is_corridor or label_count < 10:
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
        is_candidate = node in hub_nodes and not is_root and not is_target
        is_carrier = node in carrier_nodes and not is_root and not is_target
        is_evidence_relay = (
            not is_root and not is_target and evidence_share > 0 and message_internal
        )
        marker = (
            "*"
            if is_target
            else "D"
            if is_candidate or is_carrier
            else "s"
            if is_root
            else "o"
        )
        color = (
            "#f2c14e"
            if is_target
            else "#2a9d8f"
            if is_candidate or is_carrier
            else "#8fc9bd"
            if is_evidence_relay
            else "#457b9d"
            if is_root
            else "#b8b8b8"
        )
        facecolor = color if not is_candidate or is_carrier else "none"
        edgecolor = "black" if is_carrier else "#2a9d8f" if is_candidate else "white"
        axis.scatter(
            node[0],
            node[1],
            s=35 + 75 * node_mass[node] / maximum,
            marker=marker,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidth=1.5 if is_carrier or is_candidate else 0.5,
            zorder=2,
        )

    shown_positions = sorted({query} | {node[1] for node in hub_nodes | carrier_nodes})
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
    axis.set_title("A  Frozen backbone, corridor, and candidate hubs", loc="left")
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
                markerfacecolor="none",
                markeredgecolor="#2a9d8f",
                linestyle="",
                label="frozen hub candidate",
            ),
            Line2D(
                [],
                [],
                marker="D",
                markerfacecolor="#2a9d8f",
                markeredgecolor="black",
                linestyle="",
                label="confirmed carrier",
            ),
            Line2D([], [], marker="*", linestyle="", label="target", color="#f2c14e"),
            Line2D([], [], color="0.2", linewidth=2.2, label="backbone edge"),
            Line2D(
                [],
                [],
                color="0.35",
                linewidth=1.5,
                linestyle="--",
                label="frozen corridor edge",
            ),
            Line2D([], [], color="0.65", linewidth=0.7, label="other throughput"),
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
    scalar.set_label("evidence-lineage signed action", fontsize=8)


def _timeline_ticks(position: np.ndarray, token_labels) -> tuple[np.ndarray, list[str]]:
    count = min(8, len(position))
    slot = np.unique(np.linspace(0, len(position) - 1, count, dtype=int))
    return slot, [_token_tick(int(position[index]), token_labels) for index in slot]


def _draw_reanchor_timeline(axis, figure, artifact, token_labels) -> None:
    """Draw every ``(layer, head)`` switch trajectory without head means."""

    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    position = _array(artifact, "route_row_position", int)
    switch = _array(artifact, "reanchor_switch_delta")
    score = _array(artifact, "reanchor_score")
    if switch.shape != score.shape or switch.ndim != 3:
        raise ValueError("re-anchor timeline must have [layer,head,row] shape")
    layers, heads, rows = switch.shape
    values = switch.reshape(layers * heads, rows)
    strength = np.maximum(np.abs(values), score.reshape(layers * heads, rows))
    colors, limit = _rgba(values, strength)
    axis.imshow(colors, origin="lower", aspect="auto", interpolation="nearest")

    query = int(np.asarray(artifact["query_position"]).item())
    query_slot = _target_slot(position, query)
    axis.axvline(query_slot, color="#f2c14e", linewidth=1.2, linestyle="--")
    candidate_layer = np.asarray(
        artifact.get("reanchor_candidate_layer", np.empty(0)), dtype=int
    )
    candidate_head = np.asarray(
        artifact.get("reanchor_candidate_head", np.empty(0)), dtype=int
    )
    candidate_position = np.asarray(
        artifact.get("reanchor_candidate_position", np.empty(0)), dtype=int
    )
    candidate_kind = np.asarray(
        artifact.get("reanchor_candidate_source_kind", np.empty(0)), dtype=str
    )
    immediate = np.asarray(
        artifact.get(
            "reanchor_candidate_current_target_match",
            np.zeros(len(candidate_layer)),
        ),
        dtype=bool,
    )
    slot_lookup = {int(value): index for index, value in enumerate(position)}
    for source_kind in np.unique(candidate_kind):
        selected = np.flatnonzero(candidate_kind == source_kind)
        selected = selected[
            [int(candidate_position[index]) in slot_lookup for index in selected]
        ]
        if not len(selected):
            continue
        x = [slot_lookup[int(candidate_position[index])] for index in selected]
        y = candidate_layer[selected] * heads + candidate_head[selected]
        axis.scatter(
            x,
            y,
            marker=SOURCE_MARKERS.get(str(source_kind), "d"),
            s=23,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
    selected = np.flatnonzero(immediate)
    if len(selected):
        x = [slot_lookup[int(candidate_position[index])] for index in selected]
        y = candidate_layer[selected] * heads + candidate_head[selected]
        axis.scatter(x, y, marker="*", s=45, color="#f2c14e", zorder=4)

    x_tick, x_label = _timeline_ticks(position, token_labels)
    axis.set_xticks(x_tick, x_label, rotation=25, ha="right", fontsize=6.5)
    y_tick = np.unique(
        np.linspace(0, layers * heads - 1, min(9, layers * heads), dtype=int)
    )
    axis.set_yticks(
        y_tick,
        [f"L{index // heads}:h{index % heads}" for index in y_tick],
        fontsize=6.5,
    )
    axis.set_xlabel("Predictor position q (predicts token q+1)")
    axis.set_ylabel("Layer/head track")
    axis.set_title(
        "B  Full-row local→long-range anchor switch (one row/head; no mean)",
        loc="left",
    )
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                markerfacecolor="none",
                markeredgecolor="black",
                linestyle="",
                label="prompt-evidence candidate",
            ),
            Line2D(
                [],
                [],
                marker="D",
                markerfacecolor="none",
                markeredgecolor="black",
                linestyle="",
                label="remote-response candidate",
            ),
            Line2D(
                [],
                [],
                marker="X",
                markerfacecolor="none",
                markeredgecolor="black",
                linestyle="",
                label="other-prompt candidate",
            ),
            Line2D(
                [],
                [],
                marker="*",
                color="#f2c14e",
                linestyle="",
                label="audited next-token event",
            ),
        ],
        frameon=False,
        fontsize=6.5,
        ncol=2,
        loc="upper left",
    )
    figure.colorbar(
        ScalarMappable(norm=Normalize(-limit, limit), cmap=colormaps["coolwarm"]),
        ax=axis,
        fraction=0.035,
        pad=0.02,
        label="Δ long-range W_O(A V) transport share",
    )


def _source_track_arrays(artifact):
    names = np.asarray(artifact["reanchor_bucket_name"], dtype=str)
    position = _array(artifact, "reanchor_source_position", int)
    transport = _array(artifact, "reanchor_source_transport")
    action = _array(artifact, "reanchor_source_downstream_action")
    attention = _array(artifact, "reanchor_source_attention")
    if not (
        position.shape == transport.shape == action.shape == attention.shape
        and position.ndim == 4
        and position.shape[-1] == len(names)
    ):
        raise ValueError("re-anchor source tracks have inconsistent shapes")
    return names, position, transport, action, attention


def _draw_reanchor_triangles(axes, figure, artifact, token_labels) -> None:
    """Show strongest full-row sources around each frozen switch candidate."""

    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    names, source, transport, action, attention = _source_track_arrays(artifact)
    row_position = _array(artifact, "route_row_position", int)
    response_start = int(np.asarray(artifact["response_start"]).item())
    local_window = int(np.asarray(artifact["local_window"]).item())
    layers, heads, _, _ = source.shape
    candidate_layer = _array(artifact, "reanchor_candidate_layer", int)
    candidate_head = _array(artifact, "reanchor_candidate_head", int)
    candidate_position = _array(artifact, "reanchor_candidate_position", int)
    candidate_source = _array(artifact, "reanchor_candidate_source_position", int)
    candidate_source_unit = _array(artifact, "reanchor_candidate_source_unit", int)
    candidate_local_source = _array(
        artifact, "reanchor_candidate_previous_local_source_position", int
    )
    candidate_local_unit = _array(
        artifact, "reanchor_candidate_previous_local_source_unit", int
    )
    candidate_delta = _array(artifact, "reanchor_candidate_switch_delta")
    candidate_rise = _array(artifact, "reanchor_candidate_relative_anchor_rise")
    candidate_fall = _array(artifact, "reanchor_candidate_relative_local_fall")
    candidate_evidence = _array(artifact, "reanchor_candidate_source_evidence_fraction")
    candidate_source_downstream = _array(
        artifact, "reanchor_candidate_source_downstream_action"
    )
    candidate_source_immediate = _array(
        artifact, "reanchor_candidate_source_immediate_action"
    )
    candidate_long_range_downstream = _array(
        artifact, "reanchor_candidate_long_range_downstream_action"
    )
    candidate_long_range_immediate = _array(
        artifact, "reanchor_candidate_long_range_immediate_action"
    )
    candidate_integration = _array(
        artifact, "reanchor_candidate_selected_root_integration_coherence"
    )
    limit = _symmetric_limit(action)
    norm = Normalize(-limit, limit)
    cmap = colormaps["coolwarm"]
    source_max = max(int(row_position[-1]), response_start)

    for panel, axis in enumerate(axes):
        if panel >= len(candidate_layer):
            axis.set_axis_off()
            if panel == 0:
                axis.text(
                    0.5,
                    0.5,
                    "No local→long-range anchor switch candidate",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
            continue
        layer = int(candidate_layer[panel])
        head = int(candidate_head[panel])
        event_position = int(candidate_position[panel])
        if not (0 <= layer < layers and 0 <= head < heads):
            raise ValueError("re-anchor candidate layer/head is invalid")
        axis.axhspan(-0.5, response_start - 0.5, color="#457b9d", alpha=0.06)
        lower = np.maximum(response_start, row_position - local_window)
        axis.fill_between(
            row_position,
            lower,
            row_position,
            color="0.7",
            alpha=0.13,
            label="recent-local band",
        )
        axis.plot(row_position, row_position, color="0.6", linewidth=0.5)
        maximum = float(np.nanmax(transport[layer, head]))
        if not np.isfinite(maximum) or maximum <= 0:
            maximum = 1.0
        for source_kind, name in enumerate(names):
            y = source[layer, head, :, source_kind]
            valid = y >= 0
            if not bool(valid.any()):
                continue
            mass = transport[layer, head, valid, source_kind]
            sizes = 10 + 80 * np.sqrt(np.clip(mass / maximum, 0, 1))
            values = action[layer, head, valid, source_kind]
            opacity = np.clip(
                0.3 + 0.7 * attention[layer, head, valid, source_kind], 0.3, 1
            )
            colors = cmap(norm(values))
            colors[:, 3] = opacity
            axis.scatter(
                row_position[valid],
                y[valid],
                s=sizes,
                marker=SOURCE_MARKERS.get(str(name), "d"),
                c=colors,
                linewidths=0.35,
                edgecolors="0.25",
                label=str(name).replace("_", " "),
                zorder=2,
            )
        axis.axvline(event_position, color="black", linestyle="--", linewidth=0.8)
        axis.scatter(
            [event_position],
            [int(candidate_source[panel])],
            s=125,
            facecolors="none",
            edgecolors="#f2c14e",
            linewidths=1.8,
            zorder=4,
        )
        source_immediate = float(candidate_source_immediate[panel])
        long_range_immediate = float(candidate_long_range_immediate[panel])
        if np.isfinite(source_immediate):
            source_action_label = (
                f"winner-source immediate action={source_immediate:.2g}"
            )
            aggregate_action_label = (
                f"all-long-range immediate action={long_range_immediate:.2g}"
            )
        else:
            source_action_label = (
                "winner-source downstream action to audited q="
                f"{candidate_source_downstream[panel]:.2g}"
            )
            aggregate_action_label = (
                "all-long-range downstream action="
                f"{candidate_long_range_downstream[panel]:.2g}"
            )
        unit_names = np.asarray(artifact.get("unit_name", np.empty(0)), dtype=str)
        unit_id = int(candidate_source_unit[panel])
        unit_label = (
            str(unit_names[unit_id])[:28]
            if 0 <= unit_id < len(unit_names)
            else f"unit {unit_id}"
        )
        local_unit_id = int(candidate_local_unit[panel])
        local_unit_label = (
            str(unit_names[local_unit_id])[:20]
            if 0 <= local_unit_id < len(unit_names)
            else f"unit {local_unit_id}"
        )
        integration_label = (
            f"selected-root integration coherence={candidate_integration[panel]:.2g}"
            if np.isfinite(candidate_integration[panel])
            else "selected-root integration=not applicable"
        )
        axis.text(
            0.02,
            0.98,
            (
                f"switch Δ={candidate_delta[panel]:.2g} · "
                f"rise={candidate_rise[panel]:.2g} · fall={candidate_fall[panel]:.2g}\n"
                f"from local={_token_tick(int(candidate_local_source[panel]), token_labels)} "
                f"· {local_unit_label}\n"
                f"to anchor={_token_tick(int(candidate_source[panel]), token_labels)} "
                f"· {unit_label}\n"
                f"retained-route evidence lineage={candidate_evidence[panel]:.2g}\n"
                f"{source_action_label}\n{aggregate_action_label}\n"
                f"{integration_label}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        axis.set_xlim(float(row_position[0]) - 0.5, float(row_position[-1]) + 0.5)
        axis.set_ylim(-0.5, source_max + 0.5)
        x_tick, x_label = _timeline_ticks(row_position, token_labels)
        axis.set_xticks(
            row_position[x_tick], x_label, rotation=30, ha="right", fontsize=6
        )
        important_source = np.unique(
            np.asarray([response_start, event_position, candidate_source[panel]])
        )
        important_source = important_source[
            (important_source >= 0) & (important_source <= source_max)
        ]
        axis.set_yticks(
            important_source,
            [_token_tick(int(value), token_labels) for value in important_source],
            fontsize=6,
        )
        axis.set_title(
            f"C{panel + 1}  Bucket winners · L{layer}:h{head} · q={event_position}",
            loc="left",
            fontsize=8,
        )
        axis.grid(alpha=0.1)
        if panel == 0:
            axis.set_ylabel("Source position s")
            axis.legend(frameon=False, fontsize=5.8, loc="lower left")
            axis.set_xlabel(
                "Predictor q\n(full row bucketed; winner shown; size=message, alpha=attention)"
            )
        else:
            axis.set_xlabel("Predictor q")

    figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=list(axes),
        fraction=0.015,
        pad=0.01,
        label="winner-source action on audited target (immediate only at query)",
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
        "D  Source-conditioned integration (hue=action, opacity=displacement)",
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
    evaluated = int(
        np.asarray(artifact.get("carrier_evaluated_count", len(necessity))).item()
    )
    necessity = necessity[:evaluated]
    if not len(necessity):
        return None
    confirmed = np.asarray(
        artifact.get("carrier_confirmed", np.zeros(len(necessity))), dtype=bool
    ).reshape(-1)[:evaluated]
    throughput = np.asarray(
        artifact.get("carrier_route_throughput", np.ones(len(necessity))), dtype=float
    ).reshape(-1)[:evaluated]
    eligible = np.flatnonzero(confirmed)
    pool = eligible if len(eligible) else np.arange(len(necessity))
    return int(pool[np.argmax(throughput[pool])])


def _append_finite_effect(
    names: list[str],
    values: list[float],
    name: str,
    value: float,
) -> None:
    if np.isfinite(value):
        names.append(name)
        values.append(value)


def _draw_interventions(axis, artifact) -> None:
    native = _number(artifact["native_margin"])
    root_cut = _number(artifact["root_cut_margin"])
    root_effect = _number(artifact.get("root_value_effect", native - root_cut))
    root_evaluated = _flag(artifact.get("selected_root_evaluated", True))
    corridor_evaluated = _flag(artifact.get("corridor_evaluated", True))
    confirmation_complete = root_evaluated and corridor_evaluated
    names: list[str] = []
    values: list[float] = []
    if root_evaluated:
        _append_finite_effect(names, values, "root value effect", root_effect)
        if "selected_root_causal_score" in artifact:
            _append_finite_effect(
                names,
                values,
                "root causal min",
                _number(artifact["selected_root_causal_score"]),
            )
    if corridor_evaluated:
        for label, field in (
            ("corridor necessity", "corridor_necessity"),
            ("corridor rescue", "corridor_conditional_rescue"),
            ("corridor mediated", "corridor_mediated_rescue"),
        ):
            if field in artifact:
                _append_finite_effect(names, values, label, _number(artifact[field]))

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
        _append_finite_effect(
            names, values, "carrier necessity", float(necessity[carrier])
        )
        _append_finite_effect(
            names, values, "carrier mediated", float(mediated[carrier])
        )
        confirmed = np.asarray(
            artifact.get("carrier_confirmed", np.zeros(len(necessity))), dtype=bool
        ).reshape(-1)
        carrier_confirmed = bool(confirmed[carrier])

    values_array = np.asarray(values, dtype=float)
    if len(values_array):
        order = np.arange(len(values_array))
        colors = np.where(values_array >= 0, "#457b9d", "#e76f51")
        axis.barh(order, values_array, color=colors, alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_yticks(order, names)
        axis.invert_yaxis()
    else:
        axis.set_xlim(0, 1)
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_xlabel("Exact target-margin effect")
    full_chain_confirmed = _flag(artifact.get("full_chain_confirmed", False))
    full_chain_evaluated = _flag(
        artifact.get("full_chain_evaluated", confirmation_complete)
    )
    if confirmation_complete:
        chain_state = "confirmed" if full_chain_confirmed else "not closed"
        axis.set_title(
            f"E  Exact cut / patch / block ladder · chain {chain_state}", loc="left"
        )
    else:
        axis.set_title("E  Exact cut / patch / block ladder", loc="left")
        axis.text(
            0.5,
            0.52,
            "Exact confirmation not run",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="0.35",
        )
    status = (
        (
            "selected_root_confirmed",
            "yes" if _flag(artifact["selected_root_confirmed"]) else "no",
        )
        if root_evaluated
        else ("selected_root_confirmed", "not run"),
        (
            "corridor_restoration_valid",
            "yes" if _flag(artifact["corridor_restoration_valid"]) else "no",
        )
        if corridor_evaluated
        else ("corridor_restoration_valid", "not run"),
        (
            "corridor_confirmed",
            "yes" if _flag(artifact["corridor_confirmed"]) else "no",
        )
        if corridor_evaluated
        else ("corridor_confirmed", "not run"),
        (
            f"carrier_confirmed[selected={carrier}]",
            "yes" if carrier_confirmed else "no",
        )
        if carrier is not None
        else ("carrier_confirmed", "not run"),
        (
            "full_chain_confirmed",
            "yes" if full_chain_confirmed else "no",
        )
        if full_chain_evaluated
        else ("full_chain_confirmed", "not run"),
    )
    axis.text(
        0.99,
        0.98,
        "\n".join(f"{name}: {value}" for name, value in status),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": (
                "#edf7ed"
                if full_chain_confirmed
                else "#fff3e6"
                if confirmation_complete
                else "#f2f2f2"
            ),
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
    figure = plt.figure(figsize=(18, 15), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.15, 1.0))
    route_axis = figure.add_subplot(grid[0, 0])
    timeline_axis = figure.add_subplot(grid[0, 1])
    triangle_grid = grid[1, :].subgridspec(1, MAX_REANCHOR_PANELS)
    triangle_axes = [
        figure.add_subplot(triangle_grid[index]) for index in range(MAX_REANCHOR_PANELS)
    ]
    integration_grid = grid[2, 0].subgridspec(2, 1, height_ratios=(2.3, 1))
    stage_axis = figure.add_subplot(integration_grid[0])
    continuity_axis = figure.add_subplot(integration_grid[1])
    intervention_axis = figure.add_subplot(grid[2, 1])
    try:
        _draw_route(route_axis, figure, artifact, token_labels)
        _draw_reanchor_timeline(timeline_axis, figure, artifact, token_labels)
        _draw_reanchor_triangles(triangle_axes, figure, artifact, token_labels)
        _draw_integration(stage_axis, continuity_axis, figure, artifact)
        _draw_interventions(intervention_axis, artifact)
        sample = str(np.asarray(artifact.get("dataset_sample_id", "sample")).item())
        query = int(np.asarray(artifact["query_position"]).item())
        title = ["Mechanism audit", sample, f"predictor q={query}"]
        if "subset_audit_schema" in artifact:
            schema = int(np.asarray(artifact["subset_audit_schema"]).item())
            title.append(f"schema={schema}")
        if "analysis_stage" in artifact:
            stage = str(np.asarray(artifact["analysis_stage"]).item())
            title.append(stage)
        figure.suptitle(" · ".join(title))
        figure.savefig(destination, dpi=180)
    finally:
        plt.close(figure)
    return destination
