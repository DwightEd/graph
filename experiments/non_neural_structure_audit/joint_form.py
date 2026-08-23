"""Grouped cross-validation gate for additive versus interaction structure."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
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
