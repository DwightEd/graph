"""Optional SUPERVISED layer-block LDA: audits information loss after frozen masking.

This is never used to fit priors, choose heads, orient unsupervised scores or
choose the main detector. Natural labels are read only in this separate phase.
"""

import json

import numpy as np

from ..evaluate import Ranking, label_views
from ..evaluation_data import EvaluationBinding, read_sources
from ..head_geometry.pipeline import read_json, selected_observations
from ..offline_span.data import write_json
from .pipeline import load_masks, masked_contrast
from .profile import write_csv


def read_labels(args, records):
    with (args.dataset / "response.jsonl").open(encoding="utf-8") as stream:
        annotations = {str(row["id"]): row for row in map(json.loads, stream)}
    sources, _ = read_sources(args.dataset / "response.jsonl", args.source_info)
    binder = EvaluationBinding(dict(cache="/"), args.tokenizer)
    labels = {}
    for row in records:
        with np.load(args.output / "observations" / row["file"], allow_pickle=False) as saved:
            _, offsets = binder.bind(row, annotations[row["id"]], saved, sources)
            labels[row["file"]] = label_views(offsets, annotations[row["id"]]["labels"])["all_error"][0]
    return labels


def fit_block_lda(values, labels, keep):
    """Independent within-layer covariance, fixed 0.1 diagonal shrinkage."""
    if len(np.unique(labels)) != 2:
        return None
    layers, heads, features = values.shape[1:]
    coefficients = np.zeros((layers, heads, features))
    keep = keep.reshape(layers, heads)
    for layer in range(layers):
        matrix = values[:, layer, keep[layer]].reshape(len(values), -1)
        center = np.median(matrix, axis=0)
        quartiles = np.quantile(matrix, [.25, .75], axis=0)
        scale = np.maximum(quartiles[1] - quartiles[0], .02)
        normalized = (matrix - center) / scale
        means = np.stack([normalized[labels == value].mean(0) for value in (False, True)])
        residual = normalized - means[labels.astype(int)]
        covariance = residual.T @ residual / max(len(values) - 2, 1)
        covariance = .9 * covariance + .1 * np.diag(np.diag(covariance))
        covariance += 1e-6 * np.eye(len(covariance))
        weights = np.linalg.solve(covariance, means[1] - means[0]) / scale
        coefficients[layer, keep[layer]] = weights.reshape(-1, features)
    return coefficients


def diagnostic(args):
    freeze = read_json(args.output / "predictions/freeze.json")
    if not freeze["complete"] or freeze["labels_used"]:
        raise ValueError("Freeze unsupervised scores before supervised diagnosis")
    records = read_json(args.output / "observations/manifest.json")["records"]
    labels = read_labels(args, records)
    settings = read_json(args.output / "reference/settings.json")
    masks = load_masks(args)
    root = args.output / "diagnostic"
    root.mkdir(exist_ok=True)
    rows = []
    for group, configured in settings["groups"].items():
        chosen = configured["bank_rows"]
        values = selected_observations(args.output / "observations", chosen)
        truth = np.array([labels[file][position] for file, position, _ in chosen])
        test = [row for row in records if row["split"] == "test"
                and row["task"] + "|" + row["generator"] == group]
        rows.extend(diagnose_group(args, group, values, truth, masks, test, labels))
    write_csv(root / "supervised_lda.csv", rows)
    write_json(root / "summary.json", dict(supervised=True, priors_updated=False,
        natural_labels_used_for_lda=True, readout="layer_block_shrinkage_LDA",
        shrinkage=.1, training_tokens="same sampled reference rows as the unsupervised banks"))
    return rows


def diagnose_group(args, group, values, truth, masks, test, labels):
    rows = []
    for basis in ("raw", "contrast"):
        for name, keep in masks.items():
            training = values if basis == "raw" else masked_contrast(values, keep)
            weights = fit_block_lda(training, truth, keep)
            if weights is None:
                rows.append(dict(group=group, basis=basis, mask=name, tokens=0,
                                 auroc=None, ap=None, status="training_has_one_class"))
                continue
            scores, targets = [], []
            for row in test:
                with np.load(args.output / "observations" / row["file"], allow_pickle=False) as saved:
                    observed = saved["observations"][saved["coverage"]]
                    targets.extend(labels[row["file"]][saved["coverage"]])
                if basis == "contrast":
                    observed = masked_contrast(observed, keep)
                scores.extend(np.einsum("nlhf,lhf->n", observed, weights))
            metric = Ranking(np.asarray(targets), np.asarray(scores)).measure()
            rows.append(dict(group=group, basis=basis, mask=name, tokens=len(scores),
                             auroc=metric["auroc"], ap=metric["ap"], status="supervised_diagnostic_only"))
    return rows
