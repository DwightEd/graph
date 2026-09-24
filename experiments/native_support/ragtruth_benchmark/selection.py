"""Optional label-assisted weight selection; official test labels never enter."""

import hashlib

import numpy as np
from state_audit.storage import read_arrays, start_stage, write_arrays

from ..evaluate import annotation_targets, ranking
from .data import annotations
from .scoring import FUSIONS, WEIGHTS

SELECTION_METHODS = (FUSIONS[0], "source_full_unit_mean", "source_pair_unit_mean",
                     "raw_route", "raw_route_offline_mean", *FUSIONS[1:])


def development_sources(records, task):
    train = {r["source_id"] for r in records if r["task"] == task and r["split"] == "train"}
    test = {r["source_id"] for r in records if r["task"] == task and r["split"] == "test"}
    if train & test:
        raise ValueError(f"{task}: official train/test overlap in source IDs")
    if len(train) < 2:
        raise ValueError(f"{task}: selection requires at least two official train sources")
    ordered = sorted(train, key=lambda source: hashlib.sha256(f"42:{source}".encode()).hexdigest())
    return ordered[:max(1, int(np.ceil(.2 * len(ordered))))]


def choose_readout(output, manifest, task):
    sources = development_sources(manifest["records"], task)
    records = [r for r in manifest["records"] if r["task"] == task and r["split"] == "train"
               and r["source_id"] in sources]
    truth = annotations(output, manifest, records)
    labels, scores = [], []
    for record in records:
        values = read_arrays(output / record["directory"] / "scores.npz")
        target, _, _, valid = annotation_targets(truth[record["id"]], record["tokens"], record["id"])
        if not np.array_equal(values["token_id"], truth[record["id"]]["token_ids"]):
            raise ValueError("Selection annotation tokens differ from frozen scores")
        labels.append(target[valid])
        scores.append(np.column_stack([values[name][valid] for name in SELECTION_METHODS]))
    target, matrix = np.concatenate(labels), np.concatenate(scores)
    weights = dict(zip(FUSIONS, WEIGHTS))
    trials = [dict(method=name, weight=weights.get(name), **ranking(target, matrix[:, column]))
              for column, name in enumerate(SELECTION_METHODS)]
    if trials[0]["auroc"] is None:
        raise ValueError(f"{task}: development tokens need both classes to select by AUROC")
    selected = max(trials, key=lambda row: row["auroc"])
    return dict(method=selected["method"], weight=selected["weight"], development_sources=sources,
                development_answers=[r["id"] for r in records], trials=trials)


def select(args, manifest):
    tasks = sorted({record["task"] for record in manifest["records"]})
    choices = {task: choose_readout(args.output, manifest, task) for task in tasks}
    protocol = dict(status="selected", method="selected_detector", choices=choices,
        selection="maximum_pooled_token_AUROC_on_20percent_hashed_official_train_sources",
        seed=42, tie_policy="first_in_declared_pool; local_mean_then_baselines_then_increasing_route_weight",
        candidates=SELECTION_METHODS, labels_used_for_selection=True,
        test_labels_used=False, classifier_training=False, unsupervised=False,
        calibration="unlabelled_task_and_split_transductive; no_label_fit")
    start_stage(args.output / "selection.json", protocol, args.resume)
    for record in manifest["records"]:
        values = read_arrays(args.output / record["directory"] / "scores.npz")
        selected = values[choices[record["task"]]["method"]]
        write_arrays(args.output / record["directory"] / "selected_scores.npz",
                     token_id=values["token_id"], selected_detector=selected)
    return protocol
