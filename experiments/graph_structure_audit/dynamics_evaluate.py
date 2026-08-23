"""Post-hoc label audit for frozen cross-layer routing dynamics."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import load_npz, write_json


EVALUATION_SCHEMA = "cross-origin-routing-dynamics-evaluation-v1"


def _open_dataset(split_root):
    from research_dataset import open_research_dataset

    return open_research_dataset(split_root, device="cpu", retain_embedded_labels=True)


def _labels(dataset, sample_ids):
    store = dataset.prepare_evaluation_labels()
    labels = np.empty(len(sample_ids), dtype=np.int8)
    start = 0
    while start < len(sample_ids):
        stop = start + 1
        while stop < len(sample_ids) and sample_ids[stop] == sample_ids[start]:
            stop += 1
        labels[start:stop] = store.response_labels(dataset[str(sample_ids[start])]).cpu().numpy()
        start = stop
    return labels


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(labels, score):
    if np.unique(labels).size < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
    }


def _bootstrap(labels, score, groups, replicates, seed):
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    locations = {name: np.flatnonzero(groups == name) for name in names}
    differences = []
    for _ in range(replicates):
        chosen = rng.choice(names, len(names), replace=True)
        index = np.concatenate([locations[name] for name in chosen])
        if np.unique(labels[index]).size < 2:
            continue
        differences.append(
            score[index][labels[index] == 1].mean()
            - score[index][labels[index] == 0].mean()
        )
    if not differences:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def _score_row(name, labels, score, task):
    correct = score[labels == 0]
    hallucination = score[labels == 1]
    difference = (
        float(hallucination.mean() - correct.mean())
        if len(correct) and len(hallucination)
        else float("nan")
    )
    pooled = float(score.std(ddof=1)) if len(score) > 1 else 0.0
    return {
        "score": name,
        "task": task,
        "tokens": len(score),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "correct_mean": float(correct.mean()) if len(correct) else float("nan"),
        "hallucination_mean": (
            float(hallucination.mean()) if len(hallucination) else float("nan")
        ),
        "hallucination_minus_correct": difference,
        "standardized_difference": difference / pooled if pooled > 0 else 0.0,
        "mann_whitney_p": (
            float(mannwhitneyu(hallucination, correct, alternative="two-sided").pvalue)
            if len(correct) and len(hallucination)
            else float("nan")
        ),
        **_metrics(labels, score),
    }


def _bh(rows):
    p = np.asarray([row["mann_whitney_p"] for row in rows], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(p))
    q = np.full(len(rows), np.nan, dtype=np.float64)
    if len(finite):
        order = finite[np.argsort(p[finite])]
        running = 1.0
        for rank in range(len(order) - 1, -1, -1):
            index = order[rank]
            running = min(running, p[index] * len(order) / (rank + 1))
            q[index] = running
    for row, value in zip(rows, q):
        row["mann_whitney_q"] = float(value)
    return rows


def _matched_pairs(arrays, labels):
    sample_id = arrays["sample_id"].astype(str)
    token = arrays["token_index"].astype(float)
    length = arrays["response_length"].astype(float)
    degree = arrays["incoming_pairs"].astype(float)
    channels = arrays["active_channels"].astype(float)
    mass = arrays["retained_mass"].astype(float)
    pairs = []
    start = 0
    while start < len(sample_id):
        stop = start + 1
        while stop < len(sample_id) and sample_id[stop] == sample_id[start]:
            stop += 1
        local = np.arange(start, stop)
        positive = local[labels[local] == 1]
        negative = local[labels[local] == 0]
        for current in positive:
            if not len(negative):
                continue
            cost = (
                np.abs(
                    token[negative] / np.maximum(length[negative] - 1, 1)
                    - token[current] / max(length[current] - 1, 1)
                )
                + 0.2 * np.abs(np.log1p(degree[negative]) - np.log1p(degree[current]))
                + 0.2 * np.abs(np.log1p(channels[negative]) - np.log1p(channels[current]))
                + 0.2 * np.abs(mass[negative] - mass[current])
            )
            pairs.append((current, int(negative[np.argmin(cost)])))
        start = stop
    return pairs


def _masked_head_mean(value, count=None):
    if count is None:
        return value.mean(axis=-1)
    valid = count > 0
    total = (value * valid).sum(axis=-1)
    return total / np.maximum(valid.sum(axis=-1), 1)


def _layer_rows(arrays, labels, tasks):
    families = {
        "edge_transition": ("edge_error_map", "edge_count_map"),
        "prompt_edge_transition": ("prompt_edge_error_map", "prompt_edge_count_map"),
        "response_edge_transition": ("response_edge_error_map", "response_edge_count_map"),
        "diagonal_transition": ("diagonal_error_map", None),
        "support_transition": ("support_error_map", None),
        "self_gate": ("self_gate", None),
        "prompt_gate": ("prompt_gate", None),
        "response_gate": ("response_gate", None),
    }
    rows = []
    for family, (value_name, count_name) in families.items():
        value = arrays[value_name].astype(np.float64)
        if value.ndim == 2:
            layer_value = value
        else:
            count = arrays[count_name].astype(np.float64) if count_name else None
            layer_value = _masked_head_mean(value, count)
        for layer in range(layer_value.shape[1]):
            score = layer_value[:, layer]
            for task in ("__all__", *sorted(np.unique(tasks).tolist())):
                selected = np.isfinite(score) if task == "__all__" else np.isfinite(score) & (tasks == task)
                row = _score_row(family, labels[selected], score[selected], task)
                row["layer"] = layer + 1
                rows.append(row)
    return _bh(rows)


def _layer_head_rows(arrays, labels):
    families = {
        "edge_transition": ("edge_error_map", "edge_count_map"),
        "prompt_edge_transition": ("prompt_edge_error_map", "prompt_edge_count_map"),
        "response_edge_transition": ("response_edge_error_map", "response_edge_count_map"),
        "diagonal_transition": ("diagonal_error_map", None),
    }
    rows = []
    for family, (value_name, count_name) in families.items():
        value = arrays[value_name].astype(np.float64)
        count = arrays[count_name].astype(np.float64) if count_name else None
        for layer in range(value.shape[1]):
            for head in range(value.shape[2]):
                selected = np.ones(len(labels), dtype=bool)
                if count is not None:
                    selected &= count[:, layer, head] > 0
                score = value[selected, layer, head]
                current_labels = labels[selected]
                if len(score) == 0:
                    continue
                row = _score_row(family, current_labels, score, "__all__")
                row["layer"] = layer + 1
                row["head"] = head
                rows.append(row)
    return _bh(rows)


def _band_rows(arrays, labels, tasks):
    transitions = arrays["edge_error_map"].shape[1]
    boundaries = np.linspace(0, transitions, 4, dtype=int)
    rows = []
    map_names = (
        "edge_error_map",
        "prompt_edge_error_map",
        "response_edge_error_map",
        "diagonal_error_map",
    )
    count_names = {
        "edge_error_map": "edge_count_map",
        "prompt_edge_error_map": "prompt_edge_count_map",
        "response_edge_error_map": "response_edge_count_map",
    }
    for name in map_names:
        value = arrays[name].astype(np.float64)
        count_name = count_names.get(name)
        count = arrays[count_name].astype(np.float64) if count_name else None
        layer_value = _masked_head_mean(value, count)
        for band, (start, stop) in zip(("early", "middle", "late"), zip(boundaries[:-1], boundaries[1:])):
            score = layer_value[:, start:stop].mean(axis=1)
            for task in ("__all__", *sorted(np.unique(tasks).tolist())):
                selected = np.isfinite(score) if task == "__all__" else np.isfinite(score) & (tasks == task)
                row = _score_row(name, labels[selected], score[selected], task)
                row["band"] = band
                rows.append(row)
    return rows


def _onset_rows(arrays, labels, window=4):
    sample_id = arrays["sample_id"].astype(str)
    names = (
        "edge_transition",
        "diagonal_transition",
        "edge_state_decoupling",
        "origin_fracture",
        "closure",
    )
    values = {name: [] for name in names}
    start = 0
    while start < len(sample_id):
        stop = start + 1
        while stop < len(sample_id) and sample_id[stop] == sample_id[start]:
            stop += 1
        local_labels = labels[start:stop]
        positive = np.flatnonzero(local_labels == 1)
        if len(positive):
            onset = int(positive[0])
            left = max(0, onset - window)
            right = min(stop - start, onset + window)
            if left < onset:
                for name in names:
                    score = arrays[name][start:stop].astype(np.float64)
                    values[name].append(score[onset:right].mean() - score[left:onset].mean())
        start = stop
    rows = []
    for name, current in values.items():
        current = np.asarray(current, dtype=np.float64)
        std = current.std(ddof=1) if len(current) > 1 else 0.0
        rows.append(
            {
                "score": name,
                "responses": len(current),
                "onset_minus_pre": float(current.mean()) if len(current) else float("nan"),
                "paired_dz": float(current.mean() / std) if std > 0 else 0.0,
            }
        )
    return rows


def evaluate_dynamics_scores(
    *,
    split_root,
    score_path,
    output_dir,
    bootstrap_replicates=500,
    seed=20260823,
):
    arrays = load_npz(score_path)
    dataset = _open_dataset(split_root)
    labels = _labels(dataset, arrays["sample_id"].astype(str))
    valid = arrays["valid_rounds"] > 0
    tasks = arrays["task_type"].astype(str)
    groups = arrays["source_id"].astype(str)
    score_names = (
        "transition_recovery",
        "edge_transition",
        "prompt_edge_transition",
        "response_edge_transition",
        "diagonal_transition",
        "support_transition",
        "edge_state_gap",
        "edge_state_decoupling",
        "origin_gap",
        "origin_fracture",
        "message_gain",
        "prompt_gain",
        "response_gain",
        "closure",
        "layer_order_gain",
        "head_identity_gain",
        "endpoint_gain",
    )

    metric_rows = []
    for index, name in enumerate(score_names):
        score = arrays[name].astype(np.float64)
        for task in ("__all__", *sorted(np.unique(tasks).tolist())):
            selected = valid if task == "__all__" else valid & (tasks == task)
            row = _score_row(name, labels[selected], score[selected], task)
            row.update(
                _bootstrap(
                    labels[selected],
                    score[selected],
                    groups[selected],
                    bootstrap_replicates,
                    seed + index,
                )
            )
            metric_rows.append(row)

    matched = _matched_pairs(arrays, labels)
    matched_rows = []
    for name in score_names:
        score = arrays[name].astype(np.float64)
        difference = np.asarray([score[p] - score[n] for p, n in matched])
        std = difference.std(ddof=1) if len(difference) > 1 else 0.0
        matched_rows.append(
            {
                "score": name,
                "pairs": len(difference),
                "hallucination_minus_matched_correct": (
                    float(difference.mean()) if len(difference) else float("nan")
                ),
                "paired_dz": float(difference.mean() / std) if std > 0 else 0.0,
            }
        )

    layer_rows = _layer_rows(arrays, labels, tasks)
    layer_head_rows = _layer_head_rows(arrays, labels)
    band_rows = _band_rows(arrays, labels, tasks)
    onset_rows = _onset_rows(arrays, labels)

    recoverability = []
    for name in ("edge_transition", "diagonal_transition", "edge_state_decoupling"):
        row = next(item for item in metric_rows if item["score"] == name and item["task"] == "__all__")
        if row["ci_low"] > 0:
            conclusion = "hallucination_higher"
        elif row["ci_high"] < 0:
            conclusion = "hallucination_lower"
        else:
            conclusion = "inconclusive"
        recoverability.append(
            {
                "score": name,
                "hallucination_minus_correct": row["hallucination_minus_correct"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "conclusion": conclusion,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "matched_effects.csv", matched_rows)
    _write_csv(output_dir / "layer_metrics.csv", layer_rows)
    _write_csv(output_dir / "layer_head_metrics.csv", layer_head_rows)
    _write_csv(output_dir / "band_metrics.csv", band_rows)
    _write_csv(output_dir / "onset_profiles.csv", onset_rows)
    _write_csv(output_dir / "recoverability.csv", recoverability)
    write_json(
        output_dir / "evaluation.json",
        {
            "schema": EVALUATION_SCHEMA,
            "labels_read": True,
            "score_schema": str(arrays["schema"].item()),
            "tokens": int(len(labels)),
            "valid_tokens": int(valid.sum()),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "recoverability": recoverability,
            "outputs": {
                "metrics": "metrics.csv",
                "matched_effects": "matched_effects.csv",
                "layer_metrics": "layer_metrics.csv",
                "layer_head_metrics": "layer_head_metrics.csv",
                "band_metrics": "band_metrics.csv",
                "onset_profiles": "onset_profiles.csv",
                "recoverability": "recoverability.csv",
            },
        },
    )
