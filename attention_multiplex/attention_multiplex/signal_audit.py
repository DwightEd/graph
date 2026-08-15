"""Post-hoc audit of discriminative signals in saved multiplex token roles.

Graph construction and every candidate feature are completed before labels are
opened.  Labels are then used only to measure individual feature separation;
no combined detector is trained here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm


AUDIT_SCHEMA = "attention-multiplex-token-signal-audit-v1"
EPSILON = 1e-8


def _depth_delta(values: np.ndarray) -> np.ndarray:
    """Late-quarter minus early-quarter for a ``[layer, token]`` array."""

    band = max(1, values.shape[0] // 4)
    return values[-band:].mean(axis=0) - values[:band].mean(axis=0)


def _view_features(prefix: str, query: np.ndarray, source: np.ndarray, prompt: int):
    """Rotation-invariant token features from one joint SVD view."""

    query = np.asarray(query, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    if query.ndim != 3 or source.ndim != 3 or query.shape[2] != source.shape[2]:
        raise ValueError("query/source spectral roles have incompatible shapes")
    layers, response_tokens, _ = query.shape
    heads, tokens, _ = source.shape
    if not 0 < prompt < tokens or tokens - prompt != response_tokens:
        raise ValueError("spectral roles do not match response_idx")

    result = {}
    norm = np.linalg.norm(query, axis=2)
    result[f"{prefix}_query_norm_mean"] = norm.mean(axis=0)
    result[f"{prefix}_query_norm_depth_delta"] = _depth_delta(norm)

    if layers > 1:
        difference = query[1:] - query[:-1]
        velocity = np.linalg.norm(difference, axis=2)
        result[f"{prefix}_layer_velocity_mean"] = velocity.mean(axis=0)
        left_norm = np.linalg.norm(query[:-1], axis=2)
        right_norm = np.linalg.norm(query[1:], axis=2)
        denominator = left_norm * right_norm
        cosine = np.ones_like(denominator)
        np.divide(
            np.sum(query[:-1] * query[1:], axis=2),
            denominator,
            out=cosine,
            where=denominator > EPSILON,
        )
        result[f"{prefix}_layer_cosine_instability"] = (
            1.0 - np.clip(cosine, -1.0, 1.0)
        ).mean(axis=0)
        path_length = velocity.sum(axis=0)
        displacement = np.linalg.norm(query[-1] - query[0], axis=1)
        result[f"{prefix}_path_efficiency"] = np.divide(
            displacement,
            path_length,
            out=np.zeros_like(displacement),
            where=path_length > EPSILON,
        )
    else:
        zeros = np.zeros(response_tokens, dtype=np.float32)
        result[f"{prefix}_layer_velocity_mean"] = zeros.copy()
        result[f"{prefix}_layer_cosine_instability"] = zeros.copy()
        result[f"{prefix}_path_efficiency"] = zeros.copy()

    if layers > 2:
        acceleration = np.linalg.norm(
            query[2:] - 2.0 * query[1:-1] + query[:-2], axis=2
        )
        result[f"{prefix}_layer_acceleration_mean"] = acceleration.mean(axis=0)
    else:
        result[f"{prefix}_layer_acceleration_mean"] = np.zeros(
            response_tokens, dtype=np.float32
        )

    # Query and source factors share one latent axis. Their dot product is an
    # invariant low-rank reconstruction of a layer/head/source attention edge.
    # Prompt/history centroids divide by their respective number of tokens, so
    # this comparison is not a prompt-length versus history-length mass count.
    prompt_head = source[:, :prompt].mean(axis=1)
    prompt_by_head = np.einsum("lrd,hd->lhr", query, prompt_head, optimize=True)
    prompt_by_head = np.maximum(prompt_by_head, 0.0)
    prompt_route = prompt_by_head.mean(axis=1)

    response_source = source[:, prompt : prompt + response_tokens]
    cumulative = np.cumsum(response_source, axis=1)
    history_head = np.zeros_like(response_source)
    if response_tokens > 1:
        divisor = np.arange(1, response_tokens, dtype=np.float32)[None, :, None]
        history_head[:, 1:] = cumulative[:, :-1] / divisor
    history_by_head = np.einsum(
        "lrd,hrd->lhr", query, history_head, optimize=True
    )
    history_by_head = np.maximum(history_by_head, 0.0)
    history_route = history_by_head.mean(axis=1)
    history_route[:, 0] = np.nan
    history_by_head[:, :, 0] = np.nan

    self_by_head = np.einsum(
        "lrd,hrd->lhr", query, response_source, optimize=True
    )
    self_route = np.maximum(self_by_head, 0.0).mean(axis=1)

    result[f"{prefix}_prompt_route_per_source_mean"] = prompt_route.mean(axis=0)
    result[f"{prefix}_prompt_route_depth_delta"] = _depth_delta(prompt_route)
    result[f"{prefix}_prompt_head_disagreement"] = prompt_by_head.std(axis=1).mean(
        axis=0
    )
    result[f"{prefix}_history_route_per_source_mean"] = history_route.mean(axis=0)
    result[f"{prefix}_history_route_depth_delta"] = _depth_delta(history_route)
    result[f"{prefix}_history_head_disagreement"] = history_by_head.std(axis=1).mean(
        axis=0
    )

    route_total = prompt_route + history_route
    prompt_fraction = np.divide(
        prompt_route,
        route_total,
        out=np.full_like(prompt_route, np.nan),
        where=route_total > EPSILON,
    )
    result[f"{prefix}_prompt_vs_history_fraction"] = prompt_fraction.mean(axis=0)
    result[f"{prefix}_prompt_vs_history_depth_delta"] = _depth_delta(
        prompt_fraction
    )
    history_dominance = (history_route > prompt_route).astype(np.float32)
    history_dominance[:, 0] = np.nan
    result[f"{prefix}_history_dominance_rate"] = history_dominance.mean(axis=0)
    result[f"{prefix}_self_route_mean"] = self_route.mean(axis=0)
    return result


def _cube_features(prefix: str, values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"{prefix} must be [layer, head, response_token]")
    layer_head_mean = values.mean(axis=1)
    result = {
        f"{prefix}_mean": layer_head_mean.mean(axis=0),
        f"{prefix}_depth_delta": _depth_delta(layer_head_mean),
        f"{prefix}_head_disagreement": values.std(axis=1).mean(axis=0),
    }
    if values.shape[0] > 1:
        result[f"{prefix}_layer_variation"] = np.abs(
            values[1:] - values[:-1]
        ).mean(axis=(0, 1))
    else:
        result[f"{prefix}_layer_variation"] = np.zeros(
            values.shape[2], dtype=np.float32
        )
    return result


def extract_sample_features(arrays) -> dict[str, np.ndarray]:
    """Extract fixed, individually interpretable token features from one NPZ."""

    response_idx = int(np.asarray(arrays["response_idx"]).reshape(()))
    token_count = int(np.asarray(arrays["token_ids"]).size)
    response_tokens = token_count - response_idx
    if response_tokens < 1:
        raise ValueError("sample has no response tokens")
    result = {}
    result.update(
        _view_features(
            "mass",
            arrays["mass_query_by_layer"],
            arrays["mass_source_by_head"],
            response_idx,
        )
    )
    result.update(
        _view_features(
            "shape",
            arrays["shape_query_by_layer"],
            arrays["shape_source_by_head"],
            response_idx,
        )
    )
    result.update(_cube_features("self_attention", arrays["self_attention"]))
    # retained_row_mass and unresolved_row_mass are complements except for
    # numerical clipping. Keep only unresolved mass to avoid double weighting
    # the same mechanism family.
    result.update(
        _cube_features("unresolved_mass", arrays["unresolved_row_mass"])
    )
    result["relative_position_control"] = (
        np.arange(response_tokens, dtype=np.float32)
        / max(response_tokens - 1, 1)
    )
    for name, value in result.items():
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (response_tokens,):
            raise ValueError(f"feature {name} has wrong shape {value.shape}")
        result[name] = value
    return result


def _read_index(split_root: Path):
    path = split_root / "index.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError(f"invalid representation index: {path}")
    return rows


def load_split_features(split_root: Path):
    rows = _read_index(split_root)
    blocks = defaultdict(list)
    sample_ids = []
    response_counts = []
    sample_index = []
    task_type = []
    names = None
    for ordinal, row in enumerate(
        tqdm(rows, desc=f"features {split_root.name}", unit="sample")
    ):
        path = split_root / row["path"]
        with np.load(path, allow_pickle=False) as arrays:
            current = extract_sample_features(arrays)
        current_names = tuple(sorted(current))
        if names is None:
            names = current_names
        elif current_names != names:
            raise ValueError("feature schema changes between samples")
        count = len(current[names[0]])
        if count != int(row["response_tokens"]):
            raise ValueError(f"response length mismatch for {row['sample_id']}")
        for name in names:
            blocks[name].append(current[name])
        sample_ids.append(str(row["sample_id"]))
        response_counts.append(count)
        sample_index.append(np.full(count, ordinal, dtype=np.int32))
        task_type.append(
            np.full(count, str(row.get("task_type") or "Unknown"), dtype=object)
        )
    matrix = np.column_stack(
        [np.concatenate(blocks[name]).astype(np.float32, copy=False) for name in names]
    )
    return {
        "feature_names": list(names),
        "matrix": matrix,
        "sample_ids": sample_ids,
        "response_counts": response_counts,
        "sample_index": np.concatenate(sample_index),
        "task_type": np.concatenate(task_type),
    }


def fit_position_reference(matrix, relative_position, bins=20, minimum=64):
    """Fit a label-free train-only robust reference for each position bin."""

    matrix = np.asarray(matrix, dtype=np.float64)
    relative_position = np.asarray(relative_position, dtype=np.float64)
    bins = int(bins)
    bin_id = np.minimum((relative_position * bins).astype(np.int64), bins - 1)
    centers = np.zeros((bins, matrix.shape[1]), dtype=np.float64)
    scales = np.ones_like(centers)
    global_center = np.nanmedian(matrix, axis=0)
    global_mad = np.nanmedian(np.abs(matrix - global_center), axis=0) * 1.4826
    global_scale = np.where(global_mad > EPSILON, global_mad, 1.0)
    for current_bin in range(bins):
        selected = bin_id == current_bin
        for feature in range(matrix.shape[1]):
            values = matrix[selected, feature]
            values = values[np.isfinite(values)]
            if values.size < minimum:
                centers[current_bin, feature] = global_center[feature]
                scales[current_bin, feature] = global_scale[feature]
                continue
            center = float(np.median(values))
            scale = float(np.median(np.abs(values - center)) * 1.4826)
            centers[current_bin, feature] = center
            scales[current_bin, feature] = (
                scale if scale > EPSILON else global_scale[feature]
            )
    return {"bins": bins, "centers": centers, "scales": scales}


def apply_position_reference(matrix, relative_position, reference):
    matrix = np.asarray(matrix, dtype=np.float64)
    bins = int(reference["bins"])
    bin_id = np.minimum(
        (np.asarray(relative_position) * bins).astype(np.int64), bins - 1
    )
    return (
        matrix - reference["centers"][bin_id]
    ) / reference["scales"][bin_id]


def _load_labels(attention_split: Path, sample_ids, response_counts):
    """Open labels only after all feature matrices have been frozen."""

    from research_dataset import open_research_dataset

    dataset = open_research_dataset(
        attention_split,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=True,
    )
    if getattr(dataset, "retain_labels", False):
        for sample_id in tqdm(
            dataset.sample_ids,
            desc=f"evaluation labels {attention_split.name}",
            unit="sample",
        ):
            sample = dataset[sample_id]
            try:
                sample.attention()
            finally:
                sample.release_attention()
    labels = dataset.labels()
    blocks = []
    for sample_id, expected in zip(sample_ids, response_counts):
        sample = dataset[sample_id]
        values = labels.response_labels(sample).detach().cpu().numpy().astype(np.int8)
        if values.shape != (int(expected),):
            raise ValueError(f"label length mismatch for {sample_id}")
        blocks.append(values)
    return np.concatenate(blocks)


def _auc(y, score):
    valid = np.isfinite(score)
    selected_y = y[valid]
    selected_score = score[valid]
    if selected_y.size == 0 or np.unique(selected_y).size < 2:
        return None
    return float(roc_auc_score(selected_y, selected_score))


def _metrics(y, score, direction):
    valid = np.isfinite(score)
    selected_y = y[valid]
    selected_score = np.asarray(score[valid], dtype=np.float64) * float(direction)
    if selected_y.size == 0 or np.unique(selected_y).size < 2:
        return None
    return {
        "n": int(selected_y.size),
        "positives": int(selected_y.sum()),
        "prevalence": float(selected_y.mean()),
        "auroc": float(roc_auc_score(selected_y, selected_score)),
        "auprc": float(average_precision_score(selected_y, selected_score)),
        "correct_median": float(np.median(selected_score[selected_y == 0])),
        "hallucination_median": float(np.median(selected_score[selected_y == 1])),
    }


def _directions(y, matrix):
    result = []
    for column in range(matrix.shape[1]):
        auc = _auc(y, matrix[:, column])
        result.append(1 if auc is None or auc >= 0.5 else -1)
    return np.asarray(result, dtype=np.int8)


def _nonredundant_shortlist(names, matrix, y, directions, limit=12, threshold=0.9):
    aucs = [
        _auc(y, matrix[:, column] * directions[column])
        for column in range(matrix.shape[1])
    ]
    order = sorted(
        range(len(names)),
        key=lambda column: float("inf") if aucs[column] is None else -aucs[column],
    )
    rng = np.random.default_rng(20260815)
    if matrix.shape[0] > 100_000:
        selected_rows = rng.choice(matrix.shape[0], 100_000, replace=False)
        correlation_matrix = matrix[selected_rows]
    else:
        correlation_matrix = matrix
    selected = []
    redundant_with = {}
    for column in order:
        if names[column] == "relative_position_control":
            continue
        conflicts = []
        for previous in selected:
            valid = np.isfinite(correlation_matrix[:, column]) & np.isfinite(
                correlation_matrix[:, previous]
            )
            if valid.sum() < 3:
                correlation = 0.0
            else:
                correlation = float(
                    np.corrcoef(
                        correlation_matrix[valid, column],
                        correlation_matrix[valid, previous],
                    )[0, 1]
                )
                if not np.isfinite(correlation):
                    correlation = 0.0
            if abs(correlation) >= threshold:
                conflicts.append((previous, correlation))
        if conflicts:
            previous, correlation = max(conflicts, key=lambda item: abs(item[1]))
            redundant_with[names[column]] = {
                "feature": names[previous],
                "correlation": correlation,
            }
            continue
        selected.append(column)
        if len(selected) >= int(limit):
            break
    return selected, redundant_with


def _cluster_bootstrap(y, score, direction, sample_index, replicates=200):
    if replicates < 1:
        return None
    clusters = [
        np.flatnonzero(sample_index == sample)
        for sample in range(int(sample_index.max()) + 1)
    ]
    rng = np.random.default_rng(20260815)
    auroc = []
    auprc = []
    for _ in range(int(replicates)):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in chosen])
        metrics = _metrics(y[indices], score[indices], direction)
        if metrics is not None:
            auroc.append(metrics["auroc"])
            auprc.append(metrics["auprc"])
    if not auroc:
        return None
    return {
        "replicates": len(auroc),
        "auroc_ci95": [
            float(np.quantile(auroc, 0.025)),
            float(np.quantile(auroc, 0.975)),
        ],
        "auprc_ci95": [
            float(np.quantile(auprc, 0.025)),
            float(np.quantile(auprc, 0.975)),
        ],
    }


def _write_csv(path: Path, rows):
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows, selected_names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {row["feature"]: row for row in rows}
    names = [name for name in selected_names if name in selected]
    train = [selected[name]["adjusted_train_auroc"] for name in names]
    test = [selected[name]["adjusted_test_auroc"] for name in names]
    y = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.42 * len(names))))
    axes[0].barh(y, train, color="#4472C4")
    axes[1].barh(y, test, color="#ED7D31")
    for axis, title in zip(
        axes,
        ("Train-oriented AUROC", "Held-out test AUROC (train direction)"),
    ):
        axis.axvline(0.5, color="black", linewidth=1, linestyle="--")
        axis.set_xlim(0.35, 1.0)
        axis.set_title(title)
        axis.set_yticks(y, labels=names)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def audit_signals(
    representation_root,
    attention_root,
    output_dir,
    *,
    position_bins=20,
    shortlist=12,
    bootstrap=200,
):
    representation_root = Path(representation_root)
    attention_root = Path(attention_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze both feature matrices before any label store is opened.
    train = load_split_features(representation_root / "train")
    test = load_split_features(representation_root / "test")
    if train["feature_names"] != test["feature_names"]:
        raise ValueError("train/test feature schemas differ")
    names = train["feature_names"]
    position_column = names.index("relative_position_control")
    reference = fit_position_reference(
        train["matrix"], train["matrix"][:, position_column], bins=position_bins
    )
    train_adjusted = apply_position_reference(
        train["matrix"], train["matrix"][:, position_column], reference
    )
    test_adjusted = apply_position_reference(
        test["matrix"], test["matrix"][:, position_column], reference
    )

    train_y = _load_labels(
        attention_root / "train", train["sample_ids"], train["response_counts"]
    )
    test_y = _load_labels(
        attention_root / "test", test["sample_ids"], test["response_counts"]
    )
    raw_direction = _directions(train_y, train["matrix"])
    adjusted_direction = _directions(train_y, train_adjusted)
    selected, redundant_with = _nonredundant_shortlist(
        names,
        train_adjusted,
        train_y,
        adjusted_direction,
        limit=shortlist,
    )

    rows = []
    detailed = {}
    for column, name in enumerate(names):
        raw_train = _metrics(train_y, train["matrix"][:, column], raw_direction[column])
        raw_test = _metrics(test_y, test["matrix"][:, column], raw_direction[column])
        adjusted_train = _metrics(
            train_y, train_adjusted[:, column], adjusted_direction[column]
        )
        adjusted_test = _metrics(
            test_y, test_adjusted[:, column], adjusted_direction[column]
        )
        row = {
            "feature": name,
            "raw_direction": "higher" if raw_direction[column] > 0 else "lower",
            "raw_train_auroc": raw_train["auroc"] if raw_train else None,
            "raw_test_auroc": raw_test["auroc"] if raw_test else None,
            "raw_test_auprc": raw_test["auprc"] if raw_test else None,
            "adjusted_direction": (
                "higher" if adjusted_direction[column] > 0 else "lower"
            ),
            "adjusted_train_auroc": (
                adjusted_train["auroc"] if adjusted_train else None
            ),
            "adjusted_test_auroc": adjusted_test["auroc"] if adjusted_test else None,
            "adjusted_test_auprc": adjusted_test["auprc"] if adjusted_test else None,
            "selected_nonredundant": column in selected,
            "redundant_with": (
                redundant_with.get(name, {}).get("feature")
                if name in redundant_with
                else None
            ),
            "redundancy_correlation": (
                redundant_with.get(name, {}).get("correlation")
                if name in redundant_with
                else None
            ),
        }
        rows.append(row)
        detailed[name] = {
            "raw": {
                "direction_from_train": row["raw_direction"],
                "train": raw_train,
                "test": raw_test,
            },
            "position_adjusted": {
                "direction_from_train": row["adjusted_direction"],
                "train": adjusted_train,
                "test": adjusted_test,
            },
        }

    rows.sort(
        key=lambda row: float("inf")
        if row["adjusted_train_auroc"] is None
        else -row["adjusted_train_auroc"]
    )
    selected_names = [names[column] for column in selected]
    for column in selected:
        name = names[column]
        detailed[name]["position_adjusted"]["test_cluster_bootstrap"] = (
            _cluster_bootstrap(
                test_y,
                test_adjusted[:, column],
                adjusted_direction[column],
                test["sample_index"],
                replicates=bootstrap,
            )
        )
        by_task = {}
        for task in sorted(set(test["task_type"].tolist())):
            mask = test["task_type"] == task
            by_task[task] = _metrics(
                test_y[mask],
                test_adjusted[mask, column],
                adjusted_direction[column],
            )
        detailed[name]["position_adjusted"]["test_by_task"] = by_task

    rows_by_name = {row["feature"]: row for row in rows}
    top_signals = [
        {
            "feature": name,
            "direction_from_train": rows_by_name[name]["adjusted_direction"],
            "train_auroc": rows_by_name[name]["adjusted_train_auroc"],
            "test_auroc": rows_by_name[name]["adjusted_test_auroc"],
            "test_auprc": rows_by_name[name]["adjusted_test_auprc"],
        }
        for name in selected_names
    ]

    csv_path = output_dir / "feature_signal_ranking.csv"
    report_path = output_dir / "feature_signal_report.json"
    figure_path = output_dir / "feature_signal_ranking.png"
    reference_path = output_dir / "position_reference.npz"
    np.savez_compressed(
        reference_path,
        feature_names=np.asarray(names),
        bins=np.asarray(reference["bins"], dtype=np.int32),
        centers=reference["centers"].astype(np.float32),
        scales=reference["scales"].astype(np.float32),
    )
    _write_csv(csv_path, rows)
    _plot(figure_path, rows, selected_names)
    report = {
        "schema": AUDIT_SCHEMA,
        "state": "complete",
        "construction_labels_used": False,
        "labels_read_after_features_frozen": True,
        "protocol": (
            "candidate features and position reference are frozen first; train labels "
            "orient/rank individual exploratory signals; test labels only evaluate the "
            "frozen direction; no combined detector is fitted"
        ),
        "warning": (
            "This is exploratory feature discovery. A final unsupervised detector must "
            "freeze the selected mechanisms without using benchmark test labels."
        ),
        "representation_root": str(representation_root),
        "attention_root": str(attention_root),
        "train_tokens": int(train_y.size),
        "test_tokens": int(test_y.size),
        "test_positives": int(test_y.sum()),
        "test_prevalence": float(test_y.mean()),
        "position_bins": int(position_bins),
        "feature_count": len(names),
        "nonredundant_shortlist": selected_names,
        "top_signals": top_signals,
        "features": detailed,
        "ranking_csv": str(csv_path),
        "ranking_figure": str(figure_path),
        "position_reference": str(reference_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "report": str(report_path),
        "ranking": str(csv_path),
        "figure": str(figure_path),
        "position_reference": str(reference_path),
        "train_tokens": int(train_y.size),
        "test_tokens": int(test_y.size),
        "features": len(names),
        "nonredundant_shortlist": selected_names,
        "top_signals": top_signals,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit individual token-level signals in saved multiplex roles"
    )
    parser.add_argument("--representation-root", required=True)
    parser.add_argument("--attention-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--position-bins", type=int, default=20)
    parser.add_argument("--shortlist", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=200)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    result = audit_signals(
        arguments.representation_root,
        arguments.attention_root,
        arguments.output_dir,
        position_bins=arguments.position_bins,
        shortlist=arguments.shortlist,
        bootstrap=arguments.bootstrap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
