"""Post-hoc answer-level evaluation of a frozen mechanism artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiment_protocol import FrozenEvaluation, canonical_source_group

from .artifacts import load_artifact
from .alignment import predecessor_alignment


def _metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(score)
    label = np.asarray(label, dtype=np.int8)[finite]
    score = np.asarray(score, dtype=np.float64)[finite]
    if len(label) < 2 or np.unique(label).size < 2:
        return {"auroc": None, "auprc": None, "samples": len(label)}
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": float(average_precision_score(label, score)),
        "samples": len(label),
    }


def _benjamini_hochberg(p_value: list[float]) -> list[float]:
    """Adjust one preregistered family of finite p-values."""

    if not p_value:
        return []
    value = np.asarray(p_value, dtype=np.float64)
    order = np.argsort(value)
    adjusted = np.empty_like(value)
    running = 1.0
    count = len(value)
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, float(value[index]) * count / rank)
        adjusted[index] = running
    return adjusted.clip(0.0, 1.0).tolist()


def _source_effect_sign_permutation(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Test source-level within-group effects without answer pseudoreplication.

    For every source containing at least one positive and one negative answer,
    the effect is ``mean(score | positive) - mean(score | negative)``.  The
    independent units are these source effects.  A two-sided sign permutation
    tests their mean against zero while preserving every answer/source pairing.
    """

    if int(replicates) < 1:
        raise ValueError("source permutation replicates must be at least one")
    label = np.asarray(label, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    source_id = np.asarray(source_id).astype(str)
    if not (len(label) == len(score) == len(source_id)):
        raise ValueError("source permutation rows are misaligned")
    finite = np.isfinite(score)
    label = label[finite]
    score = score[finite]
    source_id = source_id[finite]
    effects = []
    for source in dict.fromkeys(source_id.tolist()):
        selected = source_id == source
        positive = score[selected & (label == 1)]
        negative = score[selected & (label == 0)]
        if len(positive) and len(negative):
            effects.append(float(positive.mean() - negative.mean()))
    if len(effects) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_sources_with_both_answer_classes",
            "source_effects": len(effects),
            "mean_positive_minus_negative": None,
            "p_value_two_sided": None,
            "replicates": int(replicates),
            "algorithm": "within_source_mean_difference_sign_permutation",
        }

    value = np.asarray(effects, dtype=np.float64)
    observed = float(value.mean())
    random = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(replicates)):
        sign = random.choice((-1.0, 1.0), size=len(value), replace=True)
        permuted = float(np.mean(sign * value))
        exceed += int(abs(permuted) >= abs(observed))
    return {
        "available": True,
        "reason": None,
        "source_effects": len(value),
        "mean_positive_minus_negative": observed,
        "p_value_two_sided": float((exceed + 1) / (int(replicates) + 1)),
        "replicates": int(replicates),
        "algorithm": "within_source_mean_difference_sign_permutation",
    }


def _fit_fold_predict(
    train_feature: np.ndarray,
    train_label: np.ndarray,
    test_feature: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit one fold with imputation and scaling learned on its train rows."""

    train_feature = np.asarray(train_feature, dtype=np.float64)
    test_feature = np.asarray(test_feature, dtype=np.float64)
    median = np.asarray(
        [
            np.median(column[np.isfinite(column)])
            if bool(np.isfinite(column).any())
            else 0.0
            for column in train_feature.T
        ],
        dtype=np.float64,
    )
    train_feature = np.where(np.isfinite(train_feature), train_feature, median)
    test_feature = np.where(np.isfinite(test_feature), test_feature, median)
    scaler = StandardScaler().fit(train_feature)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=int(seed),
    )
    model.fit(scaler.transform(train_feature), train_label)
    return model.predict_proba(scaler.transform(test_feature))[:, 1]


def _source_group_folds(
    label: np.ndarray,
    source_id: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int, str | None]:
    """Find an all-row source-disjoint OOF plan or return an explicit reason."""

    label = np.asarray(label, dtype=np.int8)
    groups = np.asarray(source_id).astype(str)
    if int(folds) < 2:
        raise ValueError("source-group OOF folds must be at least two")
    if len(label) != len(groups):
        raise ValueError("labels and source groups are misaligned")
    if np.unique(label).size < 2:
        return None, 0, "single_class_labels"
    unique_groups = np.unique(groups)
    requested = min(int(folds), len(unique_groups))
    if requested < 2:
        return None, 0, "fewer_than_two_source_groups"

    placeholder = np.zeros((len(label), 1), dtype=np.float64)
    for candidate in range(requested, 1, -1):
        try:
            split = list(
                StratifiedGroupKFold(
                    n_splits=candidate,
                    shuffle=True,
                    random_state=int(seed),
                ).split(placeholder, label, groups)
            )
        except ValueError:
            continue
        if len(split) != candidate:
            continue
        coverage = np.zeros(len(label), dtype=np.int8)
        valid = True
        for train, test in split:
            if np.unique(label[train]).size < 2:
                valid = False
                break
            if set(groups[train]).intersection(groups[test]):
                valid = False
                break
            coverage[test] += 1
        if valid and bool((coverage == 1).all()):
            return split, candidate, None
    return None, 0, "source_group_folds_have_single_class_training_rows"


def _unavailable_oof(
    *,
    reason: str,
    samples: int,
    folds_requested: int,
) -> dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "samples": int(samples),
        "answers_scored": 0,
        "folds_requested": int(folds_requested),
        "folds_used": 0,
        "auroc": None,
        "auprc": None,
        "supervised_readability_not_detector": True,
    }


def _oof_logistic_report(
    feature: np.ndarray,
    label: np.ndarray,
    split: list[tuple[np.ndarray, np.ndarray]] | None,
    *,
    folds_requested: int,
    folds_used: int,
    unavailable_reason: str | None,
    seed: int,
) -> dict[str, object]:
    """Return complete OOF readability predictions; never drop an invalid fold."""

    feature = np.asarray(feature, dtype=np.float64)
    label = np.asarray(label, dtype=np.int8)
    if feature.ndim != 2 or len(feature) != len(label):
        raise ValueError("OOF feature rows are misaligned")
    if split is None:
        return _unavailable_oof(
            reason=str(unavailable_reason),
            samples=len(label),
            folds_requested=folds_requested,
        )

    prediction = np.full(len(label), np.nan, dtype=np.float64)
    for fold, (train, test) in enumerate(split):
        prediction[test] = _fit_fold_predict(
            feature[train],
            label[train],
            feature[test],
            seed=seed + fold,
        )
    if not bool(np.isfinite(prediction).all()):
        return _unavailable_oof(
            reason="incomplete_oof_predictions",
            samples=len(label),
            folds_requested=folds_requested,
        )
    metrics = _metrics(label, prediction)
    return {
        "available": True,
        "reason": None,
        "samples": len(label),
        "answers_scored": len(label),
        "folds_requested": int(folds_requested),
        "folds_used": int(folds_used),
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "supervised_readability_not_detector": True,
    }


def length_confound_report(
    prompt_length: np.ndarray,
    response_length: np.ndarray,
    feature: np.ndarray,
    feature_names: tuple[str, ...],
    label: np.ndarray,
    source_id: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> dict[str, object]:
    """Measure supervised readability beyond prompt/response length alone."""

    if int(folds) < 2:
        raise ValueError("source-group OOF folds must be at least two")
    prompt = np.asarray(prompt_length, dtype=np.float64)
    response = np.asarray(response_length, dtype=np.float64)
    feature = np.asarray(feature, dtype=np.float64)
    label = np.asarray(label, dtype=np.int8)
    source_id = np.asarray(source_id).astype(str)
    rows = len(label)
    if any(len(value) != rows for value in (prompt, response, feature, source_id)):
        raise ValueError("length-control rows are misaligned")
    if feature.ndim != 2 or feature.shape[1] != len(feature_names):
        raise ValueError("primary features and names are misaligned")

    split, folds_used, reason = _source_group_folds(
        label,
        source_id,
        folds=folds,
        seed=seed,
    )
    length = np.column_stack((prompt, response))
    joint = _oof_logistic_report(
        length,
        label,
        split,
        folds_requested=folds,
        folds_used=folds_used,
        unavailable_reason=reason,
        seed=seed,
    )
    increments = {}
    for column, name in enumerate(feature_names):
        augmented = _oof_logistic_report(
            np.column_stack((length, feature[:, column])),
            label,
            split,
            folds_requested=folds,
            folds_used=folds_used,
            unavailable_reason=reason,
            seed=seed,
        )
        available = bool(joint["available"] and augmented["available"])
        increments[name] = {
            "available": available,
            "reason": (
                None
                if available
                else augmented.get("reason") or joint.get("reason")
            ),
            "length_only": joint,
            "length_plus_feature": augmented,
            "auroc_delta": (
                float(augmented["auroc"] - joint["auroc"])
                if available
                and augmented["auroc"] is not None
                and joint["auroc"] is not None
                else None
            ),
            "auprc_delta": (
                float(augmented["auprc"] - joint["auprc"])
                if available
                and augmented["auprc"] is not None
                and joint["auprc"] is not None
                else None
            ),
            "supervised_readability_not_detector": True,
        }
    return {
        "supervised_readability_not_detector": True,
        "joint_length": joint,
        "primary_feature_increment_over_length": increments,
    }


def source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Bootstrap complete source groups, preserving their answer clusters."""

    if int(replicates) < 1:
        raise ValueError("source bootstrap replicates must be at least one")
    finite = np.isfinite(score)
    label = np.asarray(label, dtype=np.int8)[finite]
    score = np.asarray(score, dtype=np.float64)[finite]
    source_id = np.asarray(source_id).astype(str)[finite]
    groups = tuple(dict.fromkeys(source_id.tolist()))
    if not groups:
        return {"replicates_valid": 0}
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(replicates)):
        selected_groups = random.choice(groups, len(groups), replace=True)
        selected = np.concatenate([rows[group] for group in selected_groups])
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
    values = np.asarray(estimates, dtype=np.float64)
    return {
        "replicates_valid": len(values),
        "auroc_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def answer_labels(table, labels) -> tuple[np.ndarray, np.ndarray]:
    """Select one canonical response label/source row in artifact answer order."""

    token_samples = table.token_sample_id.astype(str)
    location = {}
    for row, sample in enumerate(token_samples.tolist()):
        location.setdefault(sample, row)
    selected = np.asarray([location[s] for s in table.sample_id.astype(str)], dtype=int)
    return (
        labels.response_positive[selected].astype(np.int8),
        labels.source_id[selected].astype(str),
    )


def validate_canonical_binding(table, dataset, frozen: FrozenEvaluation) -> None:
    """Bind every frozen target to the label-sealed canonical cache.

    This pass uses a dataset opened with ``retain_embedded_labels=False`` and
    completes source, task, response-boundary, target-token, and predecessor
    checks before an evaluation-capable dataset is constructed.
    """

    frozen.validate_loaded(dataset, table.frozen_rows())
    if Path(str(table.metadata.get("data_root", ""))).resolve() != Path(
        dataset.root
    ).resolve():
        raise ValueError("evaluation data root differs from the capture root")

    token_sample = table.token_sample_id.astype(str)
    for answer, sample_id in enumerate(table.sample_id.astype(str).tolist()):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            response_start = int(attention.response_idx)
            token_ids = attention.token_ids.detach().cpu().numpy()
            alignment = predecessor_alignment(
                token_ids,
                response_start,
                cached_query_count=int(attention.num_response_tokens),
            )
            rows = np.flatnonzero(token_sample == sample_id)
            rows = rows[np.argsort(table.token_index[rows])]
            if str(table.source_id[answer]) != canonical_source_group(sample):
                raise ValueError("artifact answer source differs from canonical cache")
            if str(table.task_type[answer]) != str(sample.task_type or ""):
                raise ValueError("artifact task type differs from canonical cache")
            if str(table.generator_model[answer]) != str(sample.generator_model or ""):
                raise ValueError("artifact generator model differs from canonical cache")
            if int(table.prompt_length[answer]) != response_start:
                raise ValueError("artifact prompt boundary differs from canonical cache")
            if int(table.response_length[answer]) != alignment.response_length:
                raise ValueError("artifact response length differs from canonical cache")
            if not np.array_equal(
                table.response_token_id[rows].astype(np.int64),
                alignment.target_token_id,
            ):
                raise ValueError("artifact response token IDs differ from canonical cache")
            if not np.array_equal(
                table.predictor_position[rows].astype(np.int64),
                alignment.predictor_position,
            ):
                raise ValueError("artifact predictors differ from canonical cache")
        finally:
            sample.release_attention()


def univariate_answer_report(
    feature: np.ndarray,
    names: tuple[str, ...],
    directions: dict[str, str],
    label: np.ndarray,
    source_id: np.ndarray,
    *,
    primary_names: tuple[str, ...] = (),
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    """Report frozen directions and one source-level primary test family."""

    if int(bootstrap_replicates) < 1:
        raise ValueError("source permutation replicates must be at least one")
    primary = set(primary_names)
    if not primary.issubset(names):
        raise ValueError("primary univariate features are absent")
    rows = []
    primary_locations = []
    primary_p_values = []
    for column, name in enumerate(names):
        direction = directions.get(name, "exploratory")
        if direction not in {"high", "low", "exploratory"}:
            raise ValueError(f"invalid preregistered direction for {name}")
        value = feature[:, column].astype(np.float64)
        raw = _metrics(label, value)
        permutation = _source_effect_sign_permutation(
            label,
            value,
            source_id,
            replicates=bootstrap_replicates,
            seed=seed + column,
        )
        p_value = permutation["p_value_two_sided"]
        row: dict[str, object] = {
            "feature": name,
            "direction": direction,
            "raw_high": raw,
            "source_group_permutation": permutation,
            "included_in_primary_fdr": name in primary,
            "source_group_permutation_fdr_q": None,
        }
        if direction == "exploratory":
            row["oriented"] = None
            row["source_bootstrap"] = None
        else:
            oriented = value if direction == "high" else -value
            row["oriented"] = _metrics(label, oriented)
            row["source_bootstrap"] = source_bootstrap(
                label,
                oriented,
                source_id,
                replicates=bootstrap_replicates,
                seed=seed + column,
            )
        if name in primary and p_value is not None:
            primary_locations.append(len(rows))
            primary_p_values.append(p_value)
        rows.append(row)

    for location, q_value in zip(
        primary_locations,
        _benjamini_hochberg(primary_p_values),
        strict=True,
    ):
        rows[location]["source_group_permutation_fdr_q"] = q_value
    return rows


def _source_mean_bootstrap(
    value: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    if int(replicates) < 1:
        raise ValueError("source bootstrap replicates must be at least one")
    finite = np.isfinite(value)
    value = np.asarray(value, dtype=np.float64)[finite]
    source_id = np.asarray(source_id).astype(str)[finite]
    groups = tuple(dict.fromkeys(source_id.tolist()))
    if not groups:
        return {"replicates_valid": 0, "ci_low": None, "ci_high": None}
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(replicates)):
        selected_groups = random.choice(groups, len(groups), replace=True)
        selected = np.concatenate([rows[group] for group in selected_groups])
        estimates.append(float(np.mean(value[selected])))
    return {
        "replicates_valid": len(estimates),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def onset_report(
    table,
    labels,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    """Compare first-onset drops with source-disjoint same-position controls."""

    requested = tuple(table.metadata.get("onset_feature_names", ()))
    lookup = {name: i for i, name in enumerate(table.token_feature_names)}
    sample_id = table.token_sample_id.astype(str)
    report = {}
    for name in requested:
        if name not in lookup:
            raise ValueError(f"onset feature {name!r} is absent from the artifact")
        value = table.token_feature[:, lookup[name]].astype(np.float64)
        onset_values = []
        onset_deltas: list[tuple[int, str, float]] = []
        controls: dict[int, list[tuple[str, float]]] = {}
        responses = 0
        for sample in dict.fromkeys(sample_id.tolist()):
            rows = np.flatnonzero(sample_id == sample)
            rows = rows[np.argsort(table.token_index[rows])]
            current = labels.token_label[rows]
            current_source = str(labels.source_id[rows[0]])
            for position in range(1, len(rows)):
                previous, row = rows[position - 1], rows[position]
                if current[position - 1] == 0 and current[position] == 0:
                    if np.isfinite(value[row]) and np.isfinite(value[previous]):
                        controls.setdefault(position, []).append(
                            (current_source, float(value[row] - value[previous]))
                        )
            transitions = np.flatnonzero(
                (current == 1)
                & np.concatenate((np.ones(1, dtype=bool), current[:-1] == 0))
            )
            if not len(transitions):
                continue
            responses += 1
            position = int(transitions[0])
            row = rows[position]
            if np.isfinite(value[row]):
                onset_values.append(float(value[row]))
            if position > 0:
                previous = rows[position - 1]
                if np.isfinite(value[row]) and np.isfinite(value[previous]):
                    onset_deltas.append(
                        (
                            position,
                            current_source,
                            float(value[row] - value[previous]),
                        )
                    )
        matched_effect = []
        matched_source = []
        matched_control = []
        matched_onset = []
        for position, source, delta in onset_deltas:
            eligible = [
                control
                for control_source, control in controls.get(position, ())
                if control_source != source
            ]
            if not eligible:
                continue
            control_mean = float(np.mean(eligible))
            matched_onset.append(delta)
            matched_control.append(control_mean)
            matched_effect.append(delta - control_mean)
            matched_source.append(source)
        effect = np.asarray(matched_effect, dtype=np.float64)
        report[name] = {
            "responses_with_first_onset": responses,
            "onsets_available": len(onset_values),
            "mean_onset_value": None
            if not onset_values
            else float(np.mean(onset_values)),
            "onset_deltas_available": len(onset_deltas),
            "mean_onset_minus_previous": None
            if not onset_deltas
            else float(np.mean([delta for _, _, delta in onset_deltas])),
            "source_disjoint_same_position_matches": len(effect),
            "mean_matched_onset_delta": None
            if not matched_onset
            else float(np.mean(matched_onset)),
            "mean_matched_non_onset_delta": None
            if not matched_control
            else float(np.mean(matched_control)),
            "mean_onset_minus_matched_non_onset_delta": None
            if not len(effect)
            else float(effect.mean()),
            "source_bootstrap": _source_mean_bootstrap(
                effect,
                np.asarray(matched_source, dtype=str),
                replicates=bootstrap_replicates,
                seed=seed + lookup[name],
            ),
        }
    return report


def evaluate_artifact(
    data_root,
    artifact_path,
    output_path,
    *,
    bootstrap_replicates: int = 1000,
    cv_folds: int = 5,
    seed: int = 20260828,
) -> dict[str, object]:
    """Freeze bytes first, validate bindings, then and only then open labels."""

    if int(bootstrap_replicates) < 1:
        raise ValueError("bootstrap_replicates must be at least one")
    if int(cv_folds) < 2:
        raise ValueError("cv_folds must be at least two")

    from research_dataset import open_research_dataset

    frozen = FrozenEvaluation.capture(artifact_path, expected_split="test")
    table = load_artifact(frozen.artifact.path)
    sealed_dataset = open_research_dataset(
        data_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=False,
    )
    validate_canonical_binding(table, sealed_dataset, frozen)

    # Only after all target and provenance checks pass do we construct a
    # dataset whose public API is allowed to expose hallucination labels.
    dataset = open_research_dataset(
        data_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=True,
    )
    labels = frozen.align_loaded(dataset, table.frozen_rows())
    label, source_id = answer_labels(table, labels)
    directions = dict(table.metadata.get("answer_feature_directions", {}))
    primary_names = tuple(table.metadata["primary_answer_feature_names"])
    primary_set = set(primary_names)
    univariate = univariate_answer_report(
        table.answer_feature,
        table.answer_feature_names,
        directions,
        label,
        source_id,
        primary_names=primary_names,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    primary_columns = [
        table.answer_feature_names.index(name) for name in primary_names
    ]
    length_control = length_confound_report(
        table.prompt_length,
        table.response_length,
        table.answer_feature[:, primary_columns],
        primary_names,
        label,
        source_id,
        folds=cv_folds,
        seed=seed + 300_000,
    )
    prompt_length_baseline = {
        "direction": "raw_high_no_label_selected_orientation",
        **_metrics(label, table.prompt_length.astype(np.float64)),
        "source_bootstrap": source_bootstrap(
            label,
            table.prompt_length.astype(np.float64),
            source_id,
            replicates=bootstrap_replicates,
            seed=seed + 90_000,
        ),
    }
    response_length_baseline = {
        "direction": "raw_high_no_label_selected_orientation",
        **_metrics(label, table.response_length.astype(np.float64)),
        "source_bootstrap": source_bootstrap(
            label,
            table.response_length.astype(np.float64),
            source_id,
            replicates=bootstrap_replicates,
            seed=seed + 100_000,
        ),
    }
    report = {
        "schema": "attention-hallucination-mechanism-answer-evaluation",
        "version": 1,
        "artifact": str(frozen.artifact.path),
        "artifact_sha256": frozen.artifact.sha256,
        "labels_used": True,
        "labels_used_during": "posthoc_answer_and_onset_evaluation_only",
        "supervised_readability_not_detector": True,
        "answer_label_definition": (
            "1 iff any response token is canonically labeled hallucinated"
        ),
        "samples": len(label),
        "positive_answers": int(label.sum()),
        "prevalence": float(label.mean()),
        "mechanism_observability": table.metadata.get(
            "mechanism_observability", {}
        ),
        "observer_generator_audit": table.metadata.get(
            "observer_generator_audit", {}
        ),
        "primary_answer_feature_names": list(primary_names),
        "primary_answer_univariate": [
            row for row in univariate if row["feature"] in primary_set
        ],
        "all_answer_univariate": univariate,
        "exploratory_answer_univariate": [
            row for row in univariate if row["feature"] not in primary_set
        ],
        "length_baselines": {
            "prompt_length": prompt_length_baseline,
            "response_length": response_length_baseline,
            "joint_length": length_control["joint_length"],
        },
        "prompt_length_baseline": prompt_length_baseline,
        "response_length_baseline": response_length_baseline,
        "joint_length_baseline": length_control["joint_length"],
        "primary_feature_length_increment": length_control[
            "primary_feature_increment_over_length"
        ],
        "length_confound_control": length_control,
        "token_onset_diagnostics": onset_report(
            table,
            labels,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 200_000,
        ),
        "claim_boundary": (
            "These are post-hoc mechanism tests of frozen trajectories. "
            "A×gradient is local first-order attribution; interventions measure "
            "evidence/history sensitivity. Neither alone proves that knowledge "
            "came from model parameters. When generator and observer differ, "
            "the result is a teacher-forced observer audit, not formation inside "
            "the original generator."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "evaluation": str(output_path.resolve())}
