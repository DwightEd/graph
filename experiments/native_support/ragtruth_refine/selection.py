"""One frozen candidate per task, chosen only from train-development token AUROC."""

import numpy as np
from tqdm import tqdm
from state_audit.storage import read_arrays, start_stage

from ..evaluate import annotation_targets, ranking
from ..ragtruth_benchmark.report import pair_counts
from .data import read_annotations, require_scores
from .scoring import SELECTION_POOL


def choose_task(output, reader, manifest, task):
    records = [row for row in manifest["records"] if row["task"] == task and row["split"] == "train"]
    truth = read_annotations(reader, manifest, records)
    labels, scores = [], []
    for record in records:
        saved = read_arrays(output / record["directory"] / "scores.npz")
        annotation = truth[record["id"]]
        if not np.array_equal(saved["token_id"], annotation["token_ids"]):
            raise ValueError(f"{record['id']}: selection annotation token alignment differs")
        target, _, _, valid = annotation_targets(annotation, record["tokens"], record["id"])
        labels.append(target[valid])
        scores.append(np.column_stack([saved[name][valid] for name in SELECTION_POOL]))
    target, matrix = np.concatenate(labels), np.concatenate(scores)
    candidates = list(enumerate(SELECTION_POOL))
    trials = [dict(method=name, **ranking(target, matrix[:, column]))
              for column, name in tqdm(candidates, desc=f"select {task} on train development")]
    if trials[0]["auroc"] is None:
        raise ValueError(f"{task}: development selection requires both token classes")
    # Exact half-integer pair counts avoid choosing a complex method for a 1e-16 AUC tail.
    concordant, pairs = pair_counts(target, matrix)
    for trial, count in zip(trials, concordant):
        trial.update(concordant_pairs=float(count), positive_negative_pairs=pairs)
    best = trials[int(np.argmax(concordant))]
    return dict(method=best["method"], trials=trials,
        development_sources=manifest["development_sources"][task],
        development_answers=[row["id"] for row in records])


def select(args, reader, manifest):
    require_scores(args.output, manifest)
    choices = {task: choose_task(args.output, reader, manifest, task) for task in args.tasks}
    result = dict(method="refined_detector", choices=choices, candidates=SELECTION_POOL,
        labels_used_for_selection=True, classifier_training=False, unsupervised=False,
        test_labels_used=False, selection="maximum_development_pooled_token_AUROC",
        tie_policy="first_in_declared_pool; baselines_then_tie_then_small_residuals",
        development_fraction=.2, seed=42,
        test_status="previous_test_results_informed_design; exploratory_re_evaluation")
    start_stage(args.output / "selection.json", result, args.resume)
    return result
