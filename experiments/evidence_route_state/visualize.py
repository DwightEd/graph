"""Single-sample views of route state and exact stored graph endpoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

plt.switch_backend("Agg")


ACCOUNT_NAMES = (
    "prompt-carried evidence",
    "grounded relay",
    "unrooted feedback",
    "predictor self",
    "unknown",
)


def read_capture(path: str | Path) -> dict[str, np.ndarray]:
    """Read one per-sample capture without interpreting omitted graph tails."""

    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def read_posterior(path: str | Path, sample_id: str) -> np.ndarray:
    """Select one sample's ordered captured posterior from frozen scores."""

    with np.load(path) as stored:
        selected = stored["sample_id"].astype(str) == str(sample_id)
        order = np.argsort(stored["token_index"][selected], kind="stable")
        return stored["captured_posterior"][selected][order]


def route_matrices(
    capture: Mapping[str, np.ndarray], prediction_position: int
) -> dict[str, object]:
    """Build layer/head rows for one prediction from persisted graph scalars.

    Explicit columns are exact source-token endpoints.  The final ``None``
    column is the omitted capacity and deliberately has no token endpoint.
    """

    positions = np.asarray(capture["prediction_position"])
    token_index = int(np.flatnonzero(positions == prediction_position)[0])
    direct = np.asarray(capture["prompt_evidence"])
    layers, _tokens, heads = direct.shape
    account = np.stack(
        (
            direct[:, token_index],
            np.asarray(capture["grounded_response_relay"])[:, token_index],
            np.asarray(capture["unrooted_response_feedback"])[:, token_index],
            np.asarray(capture["predictor_self"])[:, token_index],
            np.asarray(capture["unknown_route"])[:, token_index],
        ),
        axis=-1,
    ).reshape(layers * heads, len(ACCOUNT_NAMES))

    row_position = np.asarray(capture["graph_row_prediction_position"])
    selected_rows = np.flatnonzero(row_position == prediction_position)
    edge_start = np.asarray(capture["graph_edge_start"])
    edge_source = np.asarray(capture["graph_edge_source"])
    explicit = []
    for row in selected_rows:
        explicit.extend(edge_source[edge_start[row] : edge_start[row + 1]])
    source_position = tuple(sorted({int(source) for source in explicit}))
    source_column = {source: index for index, source in enumerate(source_position)}
    endpoint = np.zeros((layers * heads, len(source_position) + 1), dtype=np.float32)

    edge_head = np.asarray(capture["graph_edge_head"])
    edge_capacity = np.asarray(capture["graph_edge_capacity"])
    row_layer = np.asarray(capture["graph_row_layer"])
    unknown_capacity = np.asarray(capture["graph_unknown_capacity"])
    for row in selected_rows:
        layer = int(row_layer[row])
        start, stop = int(edge_start[row]), int(edge_start[row + 1])
        for edge in range(start, stop):
            target_row = layer * heads + int(edge_head[edge])
            endpoint[target_row, source_column[int(edge_source[edge])]] += float(
                edge_capacity[edge]
            )
        endpoint[layer * heads : (layer + 1) * heads, -1] += unknown_capacity[row]

    total = endpoint.sum(axis=1, keepdims=True)
    endpoint_share = np.divide(
        endpoint,
        total,
        out=np.zeros_like(endpoint),
        where=total > 0,
    )
    return {
        "account": account,
        "account_name": ACCOUNT_NAMES,
        "endpoint_share": endpoint_share,
        "source_position": (*source_position, None),
        "row_label": tuple(
            f"L{layer + 1}/H{head + 1}"
            for layer in range(layers)
            for head in range(heads)
        ),
    }


def source_labels(
    capture: Mapping[str, np.ndarray],
    positions: Sequence[int | None],
    token_text: Sequence[str],
    query_position: int,
) -> list[str]:
    """Name exact source endpoints by prompt evidence unit or response role."""

    response_start = int(capture["response_start"])
    prompt_unit = np.asarray(capture["prompt_token_unit"])
    evidence_name = np.asarray(capture["evidence_name"]).astype(str)
    labels = []
    for position in positions:
        if position is None:
            labels.append("unknown tail\n(no endpoint)")
            continue
        token = str(token_text[position]).replace("\n", "\\n")
        if position < response_start:
            unit = int(prompt_unit[position])
            role = "other prompt" if unit == 0 else evidence_name[unit - 1]
        elif position < query_position:
            role = f"history r{position - response_start}"
        elif position == query_position:
            role = "predictor self"
        else:
            role = "future"
        labels.append(f"{position}:{token}\n{role}")
    return labels


def choose_prediction(
    capture: Mapping[str, np.ndarray], posterior: np.ndarray | None
) -> int:
    """Choose the strongest valid capture candidate when none is requested."""

    valid = np.asarray(capture["valid"], dtype=bool)
    score = (
        np.asarray(posterior)
        if posterior is not None
        else np.asarray(capture["raw_route_contraction"])
        * np.asarray(capture["takeover"])
    )
    eligible = np.flatnonzero(valid & np.isfinite(score))
    if not len(eligible):
        raise ValueError("sample has no valid response-history prediction")
    return int(
        np.asarray(capture["prediction_position"])[eligible[np.argmax(score[eligible])]]
    )


def plot_sample(
    capture_path: str | Path,
    tokenizer,
    output: str | Path,
    *,
    prediction_position: int | None = None,
    captured_posterior: np.ndarray | None = None,
    sample_name: str | None = None,
) -> Path:
    """Plot temporal route state and one prediction's persisted route graph."""

    capture = read_capture(capture_path)
    positions = np.asarray(capture["prediction_position"])
    posterior = None if captured_posterior is None else np.asarray(captured_posterior)
    if posterior is not None and posterior.shape != positions.shape:
        raise ValueError("captured posterior and response prediction lengths differ")
    selected = (
        choose_prediction(capture, posterior)
        if prediction_position is None
        else int(prediction_position)
    )
    if selected not in positions:
        raise ValueError(f"prediction position {selected} is not in this sample")

    token_ids = np.asarray(capture["token_ids"], dtype=np.int64)
    token_text = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    response_text = [str(token_text[position]) for position in positions]
    selected_index = int(np.flatnonzero(positions == selected)[0])
    query = selected - 1
    view = route_matrices(capture, selected)

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.7), width_ratios=(1, 2.7))
    timeline = figure.add_subplot(grid[0, :])
    accounts = figure.add_subplot(grid[1, 0])
    endpoints = figure.add_subplot(grid[1, 1])

    x = np.arange(len(positions))
    timeline.plot(
        x,
        capture["raw_route_contraction"],
        color="#2166ac",
        label="raw contraction diagnostic",
    )
    timeline.plot(x, capture["takeover"], color="#b2182b", label="unrooted takeover")
    if posterior is not None:
        timeline.plot(x, posterior, color="#762a83", lw=2.2, label="captured posterior")
    timeline.axvline(selected_index, color="black", lw=0.9, ls="--")
    timeline.scatter(
        [selected_index],
        [np.asarray(capture["takeover"])[selected_index]],
        color="#b2182b",
        zorder=3,
    )
    stride = max(1, int(np.ceil(len(x) / 30)))
    ticks = np.arange(0, len(x), stride)
    if selected_index not in ticks:
        ticks = np.sort(np.append(ticks, selected_index))
    timeline.set_xticks(
        ticks,
        [response_text[index].replace("\n", "\\n") for index in ticks],
        rotation=60,
        ha="right",
        fontsize=7,
    )
    timeline.set_ylim(-0.03, 1.03)
    timeline.set(
        ylabel="state coordinate / probability",
        xlabel="predicted response token",
        title=(f"q={query} predicts p={selected}: {response_text[selected_index]}"),
    )
    timeline.grid(alpha=0.16)
    timeline.legend(frameon=False, ncols=3)

    account = np.asarray(view["account"])
    account_image = accounts.imshow(
        account,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=max(1.0, float(account.max(initial=0.0))),
    )
    accounts.set_xticks(
        np.arange(len(ACCOUNT_NAMES)),
        [
            "prompt-carried\nevidence",
            "grounded\nrelay",
            "unrooted\nfeedback",
            "predictor\nself",
            "unknown",
        ],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    accounts.set_yticks(
        np.arange(len(view["row_label"])), view["row_label"], fontsize=6
    )
    accounts.set(
        title="Positive-support route accounts",
        xlabel="named route component",
        ylabel="layer / head",
    )
    figure.colorbar(account_image, ax=accounts, label="constructive support share")

    endpoint_share = np.asarray(view["endpoint_share"])
    endpoint_image = endpoints.imshow(
        endpoint_share,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=max(1e-12, float(endpoint_share.max(initial=0.0))),
    )
    labels = source_labels(
        capture,
        view["source_position"],
        token_text,
        query,
    )
    source_stride = max(1, int(np.ceil(max(len(labels) - 1, 1) / 24)))
    source_ticks = list(range(0, max(len(labels) - 1, 0), source_stride))
    source_ticks.append(len(labels) - 1)
    endpoints.set_xticks(
        source_ticks,
        [labels[index] for index in source_ticks],
        rotation=65,
        ha="right",
        fontsize=6,
    )
    endpoints.set_yticks(
        np.arange(len(view["row_label"])), view["row_label"], fontsize=6
    )
    endpoints.axvline(len(labels) - 1.5, color="white", lw=1.0)
    endpoints.set(
        title="Exact stored source endpoints + endpoint-free omitted tail",
        xlabel="source token and prompt evidence unit",
        ylabel="layer / head",
    )
    figure.colorbar(endpoint_image, ax=endpoints, label="share of AVWO route capacity")

    title = sample_name or Path(capture_path).stem
    figure.suptitle(f"Evidence-route state: {title}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Plot one evidence-route sample")
    root.add_argument("capture", type=Path)
    root.add_argument("output", type=Path)
    root.add_argument("--model", type=Path, required=True)
    root.add_argument("--prediction-position", type=int)
    root.add_argument("--scores", type=Path)
    root.add_argument("--sample-id")
    return root


def main() -> None:
    args = parser().parse_args()
    from transformers import AutoTokenizer

    sample_id = args.sample_id or args.capture.stem
    posterior = None if args.scores is None else read_posterior(args.scores, sample_id)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    plot_sample(
        args.capture,
        tokenizer,
        args.output,
        prediction_position=args.prediction_position,
        captured_posterior=posterior,
        sample_name=sample_id,
    )


if __name__ == "__main__":
    main()
