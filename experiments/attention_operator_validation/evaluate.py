"""Post-hoc answer-level mechanism validation and supervised readability probes."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


MODE_PREFIXES = (
    "identity_",
    "operator_raw_",
    "operator_normalized_",
    "operator_permuted_",
)


def feature_direction(name: str) -> str:
    """Registered answer-level mechanism direction.

    ``exploratory`` means that the code exports the feature but does not choose a
    sign from the same labels used to evaluate it.
    """

    if name.startswith("prompt_mass"):
        return "low"
    if name in {
        "route_top1_share_mean",
        "route_mean_lag_fraction",
    }:
        return "low"
    if name.endswith("self_step_switch_late"):
        return "low"
    if name in {
        "prompt_code_effective_heads_mean",
        "history_code_effective_heads_mean",
        "prompt_observed_head_fraction",
        "history_observed_head_fraction",
        "row_mass_conservation_error",
        "prompt_rows_with_mass_fraction",
        "history_rows_with_mass_fraction",
    }:
        return "exploratory"
    return "high"


def _metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if len(label) < 2 or np.unique(label).size < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": float(average_precision_score(label, score)),
    }


def _source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    groups = tuple(dict.fromkeys(source_id.astype(str).tolist()))
    rows = {group: np.flatnonzero(source_id.astype(str) == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(replicates)):
        selected_group = random.choice(groups, len(groups), replace=True)
        selected = np.concatenate([rows[group] for group in selected_group])
        if np.unique(label[selected]).size < 2:
            continue
        estimates.append(
            (
                roc_auc_score(label[selected], score[selected]),
                average_precision_score(label[selected], score[selected]),
            )
        )
    if not estimates:
        return {"replicates_valid": 0}
    value = np.asarray(estimates)
    return {
        "replicates_valid": len(value),
        "auroc_ci_low": float(np.quantile(value[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(value[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(value[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(value[:, 1], 0.975)),
    }


def _benjamini_hochberg(p_value: list[float]) -> list[float]:
    if not p_value:
        return []
    value = np.asarray(p_value, dtype=float)
    order = np.argsort(value)
    adjusted = np.empty_like(value)
    running = 1.0
    count = len(value)
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, value[index] * count / rank)
        adjusted[index] = running
    return adjusted.clip(0.0, 1.0).tolist()


def univariate_report(
    feature: np.ndarray,
    feature_names: tuple[str, ...],
    label: np.ndarray,
    source_id: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    p_values = []
    p_locations = []
    for column, name in enumerate(feature_names):
        value = np.asarray(feature[:, column], dtype=np.float64)
        finite = np.isfinite(value)
        current_label = label[finite]
        current_value = value[finite]
        current_source = source_id[finite]
        if len(current_value) < 2 or np.unique(current_label).size < 2:
            rows.append(
                {
                    "feature": name,
                    "direction": feature_direction(name),
                    "samples": len(current_value),
                    "auroc_raw_high": None,
                    "auprc_raw_high": None,
                }
            )
            continue

        positive = current_value[current_label == 1]
        negative = current_value[current_label == 0]
        statistic = mannwhitneyu(positive, negative, alternative="two-sided")
        rank_biserial = (
            2.0 * float(statistic.statistic) / (len(positive) * len(negative)) - 1.0
        )
        direction = feature_direction(name)
        oriented = (
            current_value
            if direction == "high"
            else -current_value
            if direction == "low"
            else None
        )
        raw = _metrics(current_label, current_value)
        row: dict[str, object] = {
            "feature": name,
            "direction": direction,
            "samples": len(current_value),
            "positive_answers": int(current_label.sum()),
            "positive_mean": float(positive.mean()),
            "negative_mean": float(negative.mean()),
            "positive_median": float(np.median(positive)),
            "negative_median": float(np.median(negative)),
            "rank_biserial_positive_minus_negative": rank_biserial,
            "mann_whitney_p": float(statistic.pvalue),
            "auroc_raw_high": raw["auroc"],
            "auprc_raw_high": raw["auprc"],
        }
        if oriented is not None:
            result = _metrics(current_label, oriented)
            row.update(
                oriented_auroc=result["auroc"],
                oriented_auprc=result["auprc"],
                source_bootstrap=_source_bootstrap(
                    current_label,
                    oriented,
                    current_source,
                    replicates=bootstrap_replicates,
                    seed=seed + column,
                ),
            )
        else:
            row.update(oriented_auroc=None, oriented_auprc=None)
        p_locations.append(len(rows))
        p_values.append(float(statistic.pvalue))
        rows.append(row)

    for location, q_value in zip(
        p_locations,
        _benjamini_hochberg(p_values),
        strict=True,
    ):
        rows[location]["mann_whitney_fdr_q"] = q_value
    return rows


def _column_groups(feature_names: tuple[str, ...]) -> OrderedDict[str, list[int]]:
    mass = [
        index
        for index, name in enumerate(feature_names)
        if not name.startswith(MODE_PREFIXES)
    ]
    identity = [
        index for index, name in enumerate(feature_names) if name.startswith("identity_")
    ]
    raw = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("operator_raw_")
    ]
    normalized = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("operator_normalized_")
    ]
    permuted = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("operator_permuted_")
    ]
    return OrderedDict(
        (
            ("mass_only", mass),
            ("mass_plus_raw_head_code", mass + identity),
            ("mass_plus_operator_raw", mass + raw),
            ("mass_plus_operator_normalized", mass + normalized),
            ("mass_plus_operator_permuted", mass + permuted),
            ("operator_normalized_only", normalized),
        )
    )


def _fit_fold(
    train_feature: np.ndarray,
    train_label: np.ndarray,
    test_feature: np.ndarray,
) -> np.ndarray:
    train_feature = np.where(np.isfinite(train_feature), train_feature, np.nan)
    test_feature = np.where(np.isfinite(test_feature), test_feature, np.nan)
    median = np.nanmedian(train_feature, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    train_feature = np.where(np.isnan(train_feature), median, train_feature)
    test_feature = np.where(np.isnan(test_feature), median, test_feature)
    scaler = StandardScaler().fit(train_feature)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=0,
    )
    model.fit(scaler.transform(train_feature), train_label)
    return model.predict_proba(scaler.transform(test_feature))[:, 1]


def grouped_probe_report(
    feature: np.ndarray,
    feature_names: tuple[str, ...],
    label: np.ndarray,
    source_id: np.ndarray,
    *,
    folds: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    """Source-grouped supervised readability diagnostics.

    These probes use answer labels and are not an unsupervised detector result.
    Their role is to test whether operator-aware features add information beyond
    mass-only and head-permutation controls.
    """

    groups = source_id.astype(str)
    unique_groups = np.unique(groups)
    requested = min(int(folds), len(unique_groups))
    if requested < 2 or np.unique(label).size < 2:
        return {"available": False, "reason": "insufficient groups or classes"}

    split = None
    actual_folds = 0
    for candidate in range(requested, 1, -1):
        try:
            current = list(
                StratifiedGroupKFold(
                    n_splits=candidate,
                    shuffle=True,
                    random_state=seed,
                ).split(feature, label, groups)
            )
        except ValueError:
            continue
        if current:
            split = current
            actual_folds = candidate
            break
    if split is None:
        return {"available": False, "reason": "source-grouped folds cannot be formed"}
    report: dict[str, object] = {
        "available": True,
        "folds_requested": requested,
        "folds_used": actual_folds,
    }
    for name, columns in _column_groups(feature_names).items():
        if not columns:
            continue
        prediction = np.full(len(label), np.nan, dtype=np.float64)
        valid_folds = 0
        for train, test in split:
            if np.unique(label[train]).size < 2 or np.unique(label[test]).size < 2:
                continue
            prediction[test] = _fit_fold(
                feature[train][:, columns],
                label[train],
                feature[test][:, columns],
            )
            valid_folds += 1
        valid = np.isfinite(prediction)
        result = _metrics(label[valid], prediction[valid])
        report[name] = {
            "features": len(columns),
            "folds_valid": valid_folds,
            "answers_scored": int(valid.sum()),
            **result,
            "source_bootstrap": (
                _source_bootstrap(
                    label[valid],
                    prediction[valid],
                    groups[valid],
                    replicates=bootstrap_replicates,
                    seed=seed + len(columns),
                )
                if valid.any() and np.unique(label[valid]).size == 2
                else {"replicates_valid": 0}
            ),
        }
    return report
