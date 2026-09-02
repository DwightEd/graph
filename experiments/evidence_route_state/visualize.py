"""Explain one saved register graph without inventing omitted source edges."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from .registers import ORIGIN_NAMES

plt.switch_backend("Agg")


TOPOLOGY_NAMES = (
    "log capacity",
    "log effective sources",
    "top-one share",
    "prompt fraction",
    "history fraction",
    "self fraction",
    "head consensus",
)


@dataclass(frozen=True)
class GraphFrameView:
    """One prediction frame with every layer, head, and origin still explicit."""

    index: int
    query_position: int
    prediction_position: int
    node_norm: np.ndarray
    residual_cosine: np.ndarray
    head_write_cosine: np.ndarray
    route_topology: np.ndarray
    mlp_relation: np.ndarray
    margin_contribution: np.ndarray


def read_capture(path: str | Path) -> dict[str, np.ndarray]:
    """Read one compact graph sequence."""

    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def read_score(
    path: str | Path,
    sample_id: str,
    name: str = "conditional_graph_energy",
) -> np.ndarray:
    """Read one sample's frozen token score in response-token order."""

    with np.load(path) as stored:
        selected = stored["sample_id"].astype(str) == str(sample_id)
        order = np.argsort(stored["token_index"][selected], kind="stable")
        return stored[name][selected][order]


def cosine_from_gram(matrix: np.ndarray) -> np.ndarray:
    """Convert a stored signed Gram matrix to signed cosine geometry."""

    gram = np.asarray(matrix, dtype=np.float32)
    norm = np.sqrt(np.clip(np.diagonal(gram, axis1=-2, axis2=-1), 0.0, None))
    scale = norm[..., :, None] * norm[..., None, :]
    return np.divide(gram, scale, out=np.zeros_like(gram), where=scale > 0)


def frame_view(
    capture: Mapping[str, np.ndarray], prediction_position: int
) -> GraphFrameView:
    """Select one graph frame; no head, layer, or origin reduction is applied."""

    positions = np.asarray(capture["prediction_position"])
    selected = np.flatnonzero(positions == int(prediction_position))
    if len(selected) != 1:
        raise ValueError(f"prediction position {prediction_position} is not unique")
    index = int(selected[0])
    node = np.asarray(capture["node_embedding"], dtype=np.float32)[index]
    return GraphFrameView(
        index=index,
        query_position=int(np.asarray(capture["query_position"])[index]),
        prediction_position=int(positions[index]),
        node_norm=np.linalg.norm(node, axis=-1),
        residual_cosine=cosine_from_gram(capture["residual_gram"][index]),
        head_write_cosine=cosine_from_gram(capture["head_write_gram"][index]),
        route_topology=np.asarray(capture["route_topology"], dtype=np.float32)[index],
        mlp_relation=np.asarray(capture["mlp_relation"], dtype=np.float32)[index],
        margin_contribution=np.asarray(
            capture["margin_contribution"], dtype=np.float32
        )[index],
    )


def choose_prediction(
    capture: Mapping[str, np.ndarray], score: np.ndarray | None
) -> int:
    """Choose the largest frozen graph score, or the first valid frame."""

    valid = np.asarray(capture["valid"], dtype=bool)
    positions = np.asarray(capture["prediction_position"])
    if score is None:
        eligible = np.flatnonzero(valid)
        if not len(eligible):
            raise ValueError("sample has no valid graph transition")
        return int(positions[eligible[0]])

    values = np.asarray(score, dtype=np.float32)
    if values.shape != positions.shape:
        raise ValueError("frozen score and prediction lengths differ")
    eligible = np.flatnonzero(valid & np.isfinite(values))
    if not len(eligible):
        raise ValueError("sample has no finite valid graph score")
    return int(positions[eligible[np.argmax(values[eligible])]])


def origin_index(name: str) -> int:
    try:
        return ORIGIN_NAMES.index(name)
    except ValueError as error:
        choices = ", ".join(ORIGIN_NAMES)
        raise ValueError(f"origin must be one of: {choices}") from error


def pair_labels() -> tuple[tuple[int, int], ...]:
    return tuple(
        (first, second)
        for first in range(len(ORIGIN_NAMES))
        for second in range(first + 1, len(ORIGIN_NAMES))
    )


def heatmap(
    axis,
    values: np.ndarray,
    *,
    title: str,
    columns: list[str],
    color: str,
    minimum: float | None = None,
    maximum: float | None = None,
):
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=color,
        vmin=minimum,
        vmax=maximum,
    )
    axis.set_title(title)
    axis.set_xticks(np.arange(len(columns)), columns, rotation=45, ha="right")
    return image


def plot_sample(
    capture_path: str | Path,
    tokenizer,
    output: str | Path,
    *,
    prediction_position: int | None = None,
    graph_score: np.ndarray | None = None,
    route_origin: str = "evidence",
    paired_origin: str = "response",
    sample_name: str | None = None,
) -> Path:
    """Render observed register geometry for one response prediction.

    The plot contains only dense statistics persisted in ``GraphSequence``.
    It deliberately makes no claim about exact source-token edges, which are
    consumed rather than saved during capture.
    """

    capture = read_capture(capture_path)
    selected = (
        choose_prediction(capture, graph_score)
        if prediction_position is None
        else int(prediction_position)
    )
    view = frame_view(capture, selected)
    route = origin_index(route_origin)
    paired = origin_index(paired_origin)

    token_ids = np.asarray(capture["token_ids"], dtype=np.int64)
    token_text = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    positions = np.asarray(capture["prediction_position"])
    response_text = [str(token_text[position]) for position in positions]
    selected_token = response_text[view.index].replace("\n", "\\n")

    figure = plt.figure(figsize=(18, 16), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=(1.0, 1.2, 1.2, 1.0))
    timeline = figure.add_subplot(grid[0, :])
    route_axis = figure.add_subplot(grid[1, 0])
    write_axis = figure.add_subplot(grid[1, 1])
    residual_axis = figure.add_subplot(grid[2, 0])
    mlp_axis = figure.add_subplot(grid[2, 1])
    margin_axis = figure.add_subplot(grid[3, :])

    node_norm = np.linalg.norm(
        np.asarray(capture["node_embedding"], dtype=np.float32), axis=-1
    )
    x = np.arange(len(positions))
    for index, name in enumerate(ORIGIN_NAMES):
        timeline.plot(x, node_norm[:, index], label=name)
    timeline.axvline(view.index, color="black", lw=0.9, ls="--")
    if graph_score is not None:
        score_axis = timeline.twinx()
        score_axis.plot(x, graph_score, color="black", alpha=0.45, label="graph energy")
        score_axis.set_ylabel("frozen conditional graph energy")
    stride = max(1, int(np.ceil(len(x) / 30)))
    ticks = np.arange(0, len(x), stride)
    if view.index not in ticks:
        ticks = np.sort(np.append(ticks, view.index))
    timeline.set_xticks(
        ticks,
        [response_text[index].replace("\n", "\\n") for index in ticks],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    timeline.set(
        title=(
            f"q={view.query_position} predicts p={view.prediction_position}: "
            f"{selected_token}"
        ),
        xlabel="predicted response token",
        ylabel="final registered-vector norm",
    )
    timeline.grid(alpha=0.16)
    timeline.legend(frameon=False, ncols=len(ORIGIN_NAMES))

    topology = view.route_topology[..., route, :]
    effective_sources = np.expm1(topology[..., 1])
    route_image = heatmap(
        route_axis,
        effective_sources,
        title=f"{route_origin} route breadth (all heads)",
        columns=[f"H{head + 1}" for head in range(effective_sources.shape[1])],
        color="viridis",
        minimum=0.0,
    )
    route_axis.set_ylabel("layer")
    route_axis.set_yticks(
        np.arange(effective_sources.shape[0]),
        [f"L{layer + 1}" for layer in range(effective_sources.shape[0])],
    )
    figure.colorbar(route_image, ax=route_axis, label="effective causal endpoints")

    write_cosine = view.head_write_cosine[..., route, paired]
    write_image = heatmap(
        write_axis,
        write_cosine,
        title=f"AVW_O alignment: {route_origin} vs {paired_origin}",
        columns=[f"H{head + 1}" for head in range(write_cosine.shape[1])],
        color="coolwarm",
        minimum=-1.0,
        maximum=1.0,
    )
    write_axis.set_ylabel("layer")
    write_axis.set_yticks(
        np.arange(write_cosine.shape[0]),
        [f"L{layer + 1}" for layer in range(write_cosine.shape[0])],
    )
    figure.colorbar(write_image, ax=write_axis, label="signed cosine")

    pairs = pair_labels()
    residual = np.stack(
        [view.residual_cosine[..., first, second] for first, second in pairs], axis=-1
    )
    residual_image = heatmap(
        residual_axis,
        residual,
        title="Residual-register alignment across layer boundaries",
        columns=[f"{ORIGIN_NAMES[a]} / {ORIGIN_NAMES[b]}" for a, b in pairs],
        color="coolwarm",
        minimum=-1.0,
        maximum=1.0,
    )
    residual_axis.set_ylabel("layer boundary")
    residual_axis.set_yticks(
        np.arange(residual.shape[0]),
        [f"B{boundary}" for boundary in range(residual.shape[0])],
    )
    figure.colorbar(residual_image, ax=residual_axis, label="signed cosine")

    mlp_image = heatmap(
        mlp_axis,
        view.mlp_relation,
        title="Native MLP relation to registered residuals",
        columns=[*ORIGIN_NAMES, "relative update"],
        color="coolwarm",
    )
    mlp_axis.set_ylabel("layer")
    mlp_axis.set_yticks(
        np.arange(view.mlp_relation.shape[0]),
        [f"L{layer + 1}" for layer in range(view.mlp_relation.shape[0])],
    )
    figure.colorbar(mlp_image, ax=mlp_axis, label="cosine / log relative norm")

    colors = ("#1b9e77", "#7570b3", "#d95f02", "#666666")
    margin_axis.bar(ORIGIN_NAMES, view.margin_contribution, color=colors)
    margin_axis.axhline(0.0, color="black", lw=0.8)
    margin_axis.set(
        title="Exact registered contribution to target-vs-competitor margin",
        ylabel="logit margin",
    )
    margin_axis.grid(axis="y", alpha=0.16)

    title = sample_name or Path(capture_path).stem
    figure.suptitle(f"Registered information-route graph: {title}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Plot one registered route graph")
    root.add_argument("capture", type=Path)
    root.add_argument("output", type=Path)
    root.add_argument("--model", type=Path, required=True)
    root.add_argument("--prediction-position", type=int)
    root.add_argument("--scores", type=Path)
    root.add_argument("--sample-id")
    root.add_argument("--route-origin", choices=ORIGIN_NAMES, default="evidence")
    root.add_argument("--paired-origin", choices=ORIGIN_NAMES, default="response")
    return root


def main() -> None:
    args = parser().parse_args()
    from transformers import AutoTokenizer

    sample_id = args.sample_id or args.capture.stem
    score = None if args.scores is None else read_score(args.scores, sample_id)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    plot_sample(
        args.capture,
        tokenizer,
        args.output,
        prediction_position=args.prediction_position,
        graph_score=score,
        route_origin=args.route_origin,
        paired_origin=args.paired_origin,
        sample_name=sample_id,
    )


if __name__ == "__main__":
    main()
