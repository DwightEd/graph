"""Post-hoc answer-level mechanism validation and supervised readability probes."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiment_protocol import dataset_manifest_sha256


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
        "prompt_code_valid_row_fraction",
        "history_code_valid_row_fraction",
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
    groups = tuple(sorted(set(source_id.astype(str).tolist())))
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
    directions: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    directions = (
        {name: feature_direction(name) for name in feature_names}
        if directions is None
        else dict(directions)
    )
    if set(directions) != set(feature_names):
        raise ValueError("frozen feature directions do not cover the feature schema")
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
                    "direction": directions[name],
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
        direction = directions[name]
        if direction not in {"high", "low", "exploratory"}:
            raise ValueError(f"invalid frozen direction for {name}")
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


def feature_groups(feature_names: tuple[str, ...]) -> OrderedDict[str, list[str]]:
    """Return the named feature controls frozen into every stage-2 artifact."""

    raw_head_summary_names = {
        "prompt_code_effective_heads_mean",
        "history_code_effective_heads_mean",
    }
    routing = [
        name
        for name in feature_names
        if not name.startswith(MODE_PREFIXES)
        and name not in raw_head_summary_names
    ]
    raw_head_summary = [
        name
        for name in feature_names
        if name in raw_head_summary_names
    ]
    identity = [
        name for name in feature_names if name.startswith("identity_")
    ]
    raw = [
        name
        for name in feature_names
        if name.startswith("operator_raw_")
    ]
    normalized = [
        name
        for name in feature_names
        if name.startswith("operator_normalized_")
    ]
    permuted = [
        name
        for name in feature_names
        if name.startswith("operator_permuted_")
    ]
    return OrderedDict(
        (
            ("routing_only", routing),
            ("routing_plus_head_entropy", routing + raw_head_summary),
            ("routing_plus_raw_head_code", routing + identity),
            ("routing_plus_operator_raw", routing + raw),
            ("routing_plus_operator_normalized", routing + normalized),
            ("routing_plus_operator_permuted", routing + permuted),
            ("operator_normalized_only", normalized),
        )
    )


def _column_groups(
    feature_names: tuple[str, ...],
    frozen_groups: dict[str, list[str]] | None = None,
) -> OrderedDict[str, list[int]]:
    """Resolve a frozen name-based spec to artifact column indices."""

    expected = feature_groups(feature_names)
    if frozen_groups is None:
        frozen_groups = dict(expected)
    if dict(frozen_groups) != dict(expected):
        raise ValueError("frozen probe groups differ from the registered controls")
    location = {name: index for index, name in enumerate(feature_names)}
    return OrderedDict(
        (group, [location[name] for name in frozen_groups[group]])
        for group in expected
    )


def frozen_evaluation_spec(
    feature_names: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Freeze directions and supervised control membership before labels open."""

    directions = {name: feature_direction(name) for name in feature_names}
    groups = {name: list(features) for name, features in feature_groups(feature_names).items()}
    return directions, groups


def validate_frozen_evaluation_spec(
    feature_names: tuple[str, ...],
    directions,
    groups,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    expected_directions, expected_groups = frozen_evaluation_spec(feature_names)
    directions = dict(directions)
    groups = {str(name): list(features) for name, features in dict(groups).items()}
    if directions != expected_directions:
        raise ValueError("frozen feature directions differ from the implementation")
    if groups != expected_groups:
        raise ValueError("frozen probe groups differ from the implementation")
    return directions, groups


def validate_label_free_bindings(table, dataset) -> np.ndarray:
    """Rebind every answer row to canonical data without opening label APIs."""

    recorded_manifest = str(table.metadata["dataset_manifest_sha256"])
    if dataset_manifest_sha256(dataset) != recorded_manifest:
        raise ValueError("evaluation dataset manifest differs from feature artifact")
    recorded_split = str(table.metadata.get("split", ""))
    actual_split = str(dataset.manifest.get("split", ""))
    if actual_split != recorded_split:
        raise ValueError("evaluation dataset split differs from feature artifact")
    if actual_split != "test":
        raise ValueError("operator mechanism labels may only be opened on test")
    available = set(map(str, dataset.sample_ids))
    selected = tuple(table.sample_id.astype(str).tolist())
    if not set(selected).issubset(available):
        raise ValueError("feature artifact contains samples outside the dataset")

    canonical_sources = []
    for row, sample_id in enumerate(selected):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            source_id = str(sample.source_id)
            task_type = str(sample.task_type or "")
            response_length = int(attention.num_response_tokens)
        finally:
            sample.release_attention()
        if source_id != str(table.source_id[row]):
            raise ValueError("feature and canonical source IDs are misaligned")
        if task_type != str(table.task_type[row]):
            raise ValueError("feature and canonical task types are misaligned")
        if response_length != int(table.response_length[row]):
            raise ValueError("feature and canonical response lengths are misaligned")
        canonical_sources.append(source_id)
    return np.asarray(canonical_sources, dtype=str)


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
    frozen_groups: dict[str, list[str]] | None = None,
    response_length: np.ndarray | None = None,
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
    class_group_count = [
        len(np.unique(groups[label == current])) for current in np.unique(label)
    ]
    if min(class_group_count) < 2:
        return {
            "available": False,
            "reason": "each class must occur in at least two source groups",
        }

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
        if current and all(
            np.unique(label[train]).size == 2
            and np.unique(label[test]).size == 2
            for train, test in current
        ):
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
    groups_to_fit: OrderedDict[str, list[int]] = OrderedDict()
    if response_length is not None:
        response_length = np.asarray(response_length)
        if response_length.shape != (len(label),) or bool((response_length < 1).any()):
            raise ValueError("response_length must align with probe rows")
        length_feature = np.log1p(response_length.astype(np.float64))[:, None]
        groups_to_fit["response_length_only"] = []
        report["response_length_transform"] = "log1p"
    else:
        length_feature = None
    groups_to_fit.update(_column_groups(feature_names, frozen_groups))
    for name, columns in groups_to_fit.items():
        if not columns:
            if length_feature is None:
                continue
            selected_feature = length_feature
        else:
            selected_feature = feature[:, columns]
            if length_feature is not None:
                selected_feature = np.column_stack((length_feature, selected_feature))
        prediction = np.full(len(label), np.nan, dtype=np.float64)
        for train, test in split:
            prediction[test] = _fit_fold(
                selected_feature[train],
                label[train],
                selected_feature[test],
            )
        valid = np.isfinite(prediction)
        if not bool(valid.all()):
            raise RuntimeError("source-grouped folds did not score every answer")
        result = _metrics(label[valid], prediction[valid])
        report[name] = {
            "mechanism_features": len(columns),
            "features": selected_feature.shape[1],
            "includes_response_length": length_feature is not None,
            "folds_valid": actual_folds,
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
