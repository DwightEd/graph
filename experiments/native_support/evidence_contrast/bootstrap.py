"""Paired source-cluster uncertainty; used after scoring, never for selection."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import trange

from .scoring import CONTROLS, METHODS, PRIMARY


def source_bootstrap(records, repeats, seed=37):
    labels = np.concatenate([record["labels"] for record in records])
    scores = np.concatenate([record["scores"] for record in records])
    source = np.concatenate([np.repeat(record["source_id"], len(record["labels"])) for record in records])
    names, group = np.unique(source, return_inverse=True)
    if repeats == 0 or len(names) < 2:
        return dict(status="disabled_or_insufficient_sources", sources=len(names), repeats=repeats)
    selected = [PRIMARY, *CONTROLS]
    scores = scores[:, [METHODS.index(name) for name in selected]]
    generator = np.random.default_rng(seed)
    differences = {metric: [] for metric in ("ap", "auroc")}
    for _ in trange(repeats, desc="source bootstrap", leave=False):
        multiplicity = np.bincount(generator.integers(len(names), size=len(names)), minlength=len(names))
        weights = multiplicity[group]
        if not 0 < weights[labels == 1].sum() < weights.sum():
            continue
        for name, metric in (("ap", average_precision_score), ("auroc", roc_auc_score)):
            values = [metric(labels, column, sample_weight=weights) for column in scores.T]
            differences[name].append(np.asarray(values[0]) - values[1:])
    intervals = []
    for metric, values in differences.items():
        if values:
            bounds = np.quantile(values, [.025, .975], axis=0)
            intervals.extend(dict(candidate=PRIMARY, control=control, metric=metric,
                low=float(bounds[0, index]), high=float(bounds[1, index])) for index, control in enumerate(CONTROLS))
    return dict(status="estimated" if intervals else "no_two_class_replicates", sources=len(names),
                repeats=repeats, valid_repeats=len(differences["ap"]), seed=seed,
                resampling="whole_source_multiplicity; all_answers_of_source_together",
                automatic_model_selection=False, intervals=intervals)
