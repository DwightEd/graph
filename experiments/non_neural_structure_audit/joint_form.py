"""Grouped cross-validation gate for additive versus interaction structure."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import EvaluationConfig
from .evaluation_data import FrozenSample, aligned_matrix
from .features import RELATION_NAMES


def _model(*, interaction: bool, seed: int):
    steps = [StandardScaler()]
    if interaction:
        steps.append(
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        )
    steps.append(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="liblinear",
        )
    )
    return make_pipeline(*steps)


def compare_joint_forms(
    labels: np.ndarray,
    relations: np.ndarray,
    groups: np.ndarray,
    *,
    direct_columns: tuple[int, ...],
    folds: int = 5,
    seed: int = 20260823,
) -> list[dict[str, object]]:
    """Compare direct, all-additive, and pairwise-interaction readouts."""

    labels = np.asarray(labels, dtype=np.int8)
    relations = np.asarray(relations, dtype=np.float32)
    groups = np.asarray(groups)
    splitter = StratifiedGroupKFold(
        n_splits=min(folds, len(np.unique(groups))),
        shuffle=True,
        random_state=seed,
    )
    specifications = (
        ("direct", relations[:, direct_columns], False),
        ("additive", relations, False),
        ("interaction", relations, True),
    )
    rows = []
    previous_auprc = None
    for name, values, interaction in specifications:
        fold_auprc = []
        fold_auroc = []
        for train, test in splitter.split(values, labels, groups):
            model = _model(interaction=interaction, seed=seed)
            model.fit(values[train], labels[train])
            probability = model.predict_proba(values[test])[:, 1]
            fold_auprc.append(float(average_precision_score(labels[test], probability)))
            fold_auroc.append(
                float(roc_auc_score(labels[test], probability))
                if np.unique(labels[test]).size == 2
                else float("nan")
            )
        mean_auprc = float(np.mean(fold_auprc))
        rows.append(
            {
                "model": name,
                "folds": len(fold_auprc),
                "mean_auprc": mean_auprc,
                "mean_auroc": float(np.nanmean(fold_auroc)),
                "delta_auprc_from_previous": None
                if previous_auprc is None
                else mean_auprc - previous_auprc,
                "fold_auprc": fold_auprc,
            }
        )
        previous_auprc = mean_auprc
    return rows


def evaluate_joint_forms(
    samples: list[FrozenSample], config: EvaluationConfig
) -> list[dict[str, object]]:
    labels, relations, groups = [], [], []
    for sample in samples:
        current_labels, current_relations = aligned_matrix(
            sample.relation, sample.labels, sample.eligible
        )
        labels.append(current_labels)
        relations.append(current_relations)
        groups.extend([sample.source_id] * len(current_labels))
    labels = np.concatenate(labels)
    groups = np.asarray(groups)
    folds = min(
        config.grouped_cv_folds,
        len(np.unique(groups)),
        len(np.unique(groups[labels == 1])),
        len(np.unique(groups[labels == 0])),
    )
    if folds < 2:
        return []
    direct_columns = tuple(
        RELATION_NAMES.index(name)
        for name in (
            "direct_role",
            "endpoint_concentration",
            "head_fracture",
            "censoring_control",
        )
    )
    return compare_joint_forms(
        labels,
        np.concatenate(relations),
        groups,
        direct_columns=direct_columns,
        folds=folds,
        seed=config.random_seed,
    )
