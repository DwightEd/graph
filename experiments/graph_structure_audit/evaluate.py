"""Post-hoc label audit for frozen multiplex recovery scores."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import EVALUATION_SCHEMA, load_npz, write_json


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
        labels[start:stop] = (
            store.response_labels(dataset[str(sample_ids[start])]).cpu().numpy()
        )
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
        return {
            "auroc": float("nan"),
            "separability": float("nan"),
            "auprc_high": float("nan"),
            "auprc_low": float("nan"),
        }
    auc = float(roc_auc_score(labels, score))
    return {
        "auroc": auc,
        "separability": max(auc, 1.0 - auc),
        "auprc_high": float(average_precision_score(labels, score)),
        "auprc_low": float(average_precision_score(labels, -score)),
    }


def _bootstrap(labels, score, groups, replicates, seed):
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    locations = {name: np.flatnonzero(groups == name) for name in names}
    aucs, differences = [], []
    for _ in range(replicates):
        chosen = rng.choice(names, len(names), replace=True)
        index = np.concatenate([locations[name] for name in chosen])
        if np.unique(labels[index]).size < 2:
            continue
        aucs.append(roc_auc_score(labels[index], score[index]))
        differences.append(
            score[index][labels[index] == 1].mean()
            - score[index][labels[index] == 0].mean()
        )
    if not aucs:
        return {
            name: float("nan")
            for name in (
                "auroc_ci_low",
                "auroc_ci_high",
                "difference_ci_low",
                "difference_ci_high",
            )
        }
    return {
        "auroc_ci_low": float(np.quantile(aucs, 0.025)),
        "auroc_ci_high": float(np.quantile(aucs, 0.975)),
        "difference_ci_low": float(np.quantile(differences, 0.025)),
        "difference_ci_high": float(np.quantile(differences, 0.975)),
    }


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
                + 0.2
                * np.abs(np.log1p(degree[negative]) - np.log1p(degree[current]))
                + 0.2
                * np.abs(
                    np.log1p(channels[negative]) - np.log1p(channels[current])
                )
                + 0.2 * np.abs(mass[negative] - mass[current])
            )
            pairs.append((current, int(negative[np.argmin(cost)])))
        start = stop
    return pairs


def _gain_gate(score, groups, replicates, seed):
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    locations = {name: np.flatnonzero(groups == name) for name in names}
    means = []
    for _ in range(replicates):
        chosen = rng.choice(names, len(names), replace=True)
        index = np.concatenate([locations[name] for name in chosen])
        means.append(score[index].mean())
    return (
        float(score.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def evaluate_recovery_scores(
    *,
    split_root,
    score_path,
    output_dir,
    bootstrap_replicates=500,
    seed=20260822,
):
    arrays = load_npz(score_path)
    dataset = _open_dataset(split_root)
    labels = _labels(dataset, arrays["sample_id"].astype(str))
    valid = arrays["valid_rounds"] > 0
    tasks = arrays["task_type"].astype(str)
    groups = arrays["source_id"].astype(str)
    score_names = (
        "recovery",
        "edge_recovery",
        "diagonal_recovery",
        "message_gain",
        "layer_order_gain",
        "head_identity_gain",
        "endpoint_gain",
        "layer_head_gain",
        "full_channel_gain",
    )

    metric_rows = []
    for column, name in enumerate(score_names):
        score = arrays[name].astype(np.float64)
        for task in ("__all__", *sorted(np.unique(tasks).tolist())):
            selected = valid if task == "__all__" else valid & (tasks == task)
            current_labels = labels[selected]
            current_score = score[selected]
            correct = current_score[current_labels == 0]
            hallucination = current_score[current_labels == 1]
            row = {
                "score": name,
                "task": task,
                "tokens": int(selected.sum()),
                "positives": int(current_labels.sum()),
                "prevalence": (
                    float(current_labels.mean())
                    if len(current_labels)
                    else float("nan")
                ),
                "correct_mean": (
                    float(correct.mean()) if len(correct) else float("nan")
                ),
                "hallucination_mean": (
                    float(hallucination.mean())
                    if len(hallucination)
                    else float("nan")
                ),
                "hallucination_minus_correct": (
                    float(hallucination.mean() - correct.mean())
                    if len(correct) and len(hallucination)
                    else float("nan")
                ),
                "mann_whitney_p": (
                    float(
                        mannwhitneyu(
                            hallucination, correct, alternative="two-sided"
                        ).pvalue
                    )
                    if len(correct) and len(hallucination)
                    else float("nan")
                ),
                **_metrics(current_labels, current_score),
            }
            row.update(
                _bootstrap(
                    current_labels,
                    current_score,
                    groups[selected],
                    bootstrap_replicates,
                    seed + column,
                )
            )
            metric_rows.append(row)

    matched_pairs = _matched_pairs(arrays, labels)
    matched_rows = []
    for name in score_names:
        score = arrays[name].astype(np.float64)
        difference = np.asarray([score[p] - score[n] for p, n in matched_pairs])
        standard = difference.std(ddof=1) if len(difference) > 1 else 0.0
        matched_rows.append(
            {
                "score": name,
                "pairs": len(difference),
                "hallucination_minus_matched_correct": (
                    float(difference.mean()) if len(difference) else float("nan")
                ),
                "median_difference": (
                    float(np.median(difference)) if len(difference) else float("nan")
                ),
                "paired_dz": (
                    float(difference.mean() / standard) if standard > 0 else 0.0
                ),
            }
        )

    structure_rows = []
    for index, name in enumerate(score_names[3:]):
        score = arrays[name].astype(np.float64)[valid]
        mean, low, high = _gain_gate(
            score, groups[valid], bootstrap_replicates, seed + 100 + index
        )
        structure_rows.append(
            {
                "gate": name,
                "mean_gain": mean,
                "ci_low": low,
                "ci_high": high,
                "passed": bool(low > 0),
            }
        )

    recoverability_rows = []
    for name in ("recovery", "edge_recovery", "diagonal_recovery"):
        row = next(
            item
            for item in metric_rows
            if item["score"] == name and item["task"] == "__all__"
        )
        if row["difference_ci_low"] > 0:
            conclusion = "correct_more_recoverable"
        elif row["difference_ci_high"] < 0:
            conclusion = "hallucination_more_recoverable"
        else:
            conclusion = "inconclusive"
        recoverability_rows.append(
            {
                "score": name,
                "hallucination_minus_correct": row[
                    "hallucination_minus_correct"
                ],
                "ci_low": row["difference_ci_low"],
                "ci_high": row["difference_ci_high"],
                "conclusion": conclusion,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "matched_effects.csv", matched_rows)
    _write_csv(output_dir / "structure_gates.csv", structure_rows)
    _write_csv(output_dir / "recoverability.csv", recoverability_rows)
    write_json(
        output_dir / "evaluation.json",
        {
            "schema": EVALUATION_SCHEMA,
            "labels_read": True,
            "tokens": int(len(labels)),
            "valid_tokens": int(valid.sum()),
            "positive_tokens": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "recoverability": recoverability_rows,
            "structure_gates": structure_rows,
            "outputs": {
                "metrics": "metrics.csv",
                "matched_effects": "matched_effects.csv",
                "structure_gates": "structure_gates.csv",
                "recoverability": "recoverability.csv",
            },
        },
    )
