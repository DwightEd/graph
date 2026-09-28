"""Fit without natural labels, develop explicitly, freeze all tasks, then evaluate."""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import evaluate_all, source_bootstrap, subset
from .data import load_inputs
from .model import fit_models, fit_ranks, fusion_scores, rank_scores, raw_scores


TASKS = ("QA", "Summary", "Data2txt")
REPORTED = ("local", "full", "pair", "route", "isolation", "graph", "shuffled",
            "fixed_unsupervised", "selected_dev", "selected_mixed_threshold")


def train_labels(packs, task, mask):
    with np.load(packs / f"{task}_train.npz", allow_pickle=False) as saved:
        return {name: saved[name][mask] for name in ("labels", "onsets", "firsts")}


def fit_task(args, task):
    started = time.monotonic()
    directory = args.output / task
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "models.joblib").exists():
        raise FileExistsError("Preserve previous fit; choose a new output")
    pack, metadata = load_inputs(args.packs, task, "train")
    train = subset(pack, ~pack["development"])
    dev = subset(pack, pack["development"])
    print(f"FIT {task}: {len(train['target'])} tokens; no natural labels", flush=True)
    models = fit_models(train)
    train_raw = raw_scores(models, train)
    models["ranks"] = fit_ranks(train_raw)
    fit_scores = rank_scores(train_raw, models["ranks"])
    dev_scores = rank_scores(raw_scores(models, dev), models["ranks"])
    joblib.dump(models, directory / "models.joblib")
    write_arrays(directory / "fit_scores.npz", **fit_scores)
    write_arrays(directory / "development_scores.npz", **dev_scores)
    write_json(directory / "fit.json", dict(seconds=time.monotonic() - started,
        fit_tokens=len(train["target"]), development_tokens=len(dev["target"]), feature_dimensions=123,
        fit_sources=sorted({r["source_id"] for r in metadata["records"] if r["partition"] == "fit"}),
        dev_sources=sorted({r["source_id"] for r in metadata["records"] if r["partition"] == "dev"}),
        natural_labels_used_for_fitting=False, graph="answer-local +/-4 valid tokens, no self edges",
        epochs=12, batches_per_epoch=32, batch_size=4096, seed=42,
        graph_loss=models["graph_loss"], shuffled_loss=models["shuffled_loss"]))


def develop_task(args, task, fuse):
    directory = args.output / task
    stage = "selection" if fuse else "pilot"
    if (directory / "test_scores.npz").exists() or (directory / f"{stage}.json").exists():
        raise FileExistsError("Preserve frozen development selection; use a new output")
    pack, _ = load_inputs(args.packs, task, "train")
    dev = subset(pack, pack["development"])
    dev.update(train_labels(args.packs, task, pack["development"]))
    scores = read_arrays(directory / "development_scores.npz")
    if fuse:
        scores = fusion_scores(scores)
    # This stage deliberately uses development truth and is reported as such.
    metrics = evaluate_all(dev, scores, detailed=False)
    selected = max(metrics, key=lambda name: metrics[name]["auroc"])
    mixed = {name: float(np.quantile(values, .95, method="higher")) for name, values in scores.items()}
    write_json(directory / f"{stage}.json", dict(selected=selected, candidates=metrics,
        thresholds={name: value["threshold"] for name, value in metrics.items()}, mixed_thresholds=mixed,
        labels_used_for_selection=True, natural_labels_used_for_fitting=False,
        test_labels_used=False, strict_unsupervised_primary="fixed_unsupervised"))
    print(task, stage, selected, {name: round(metrics[name]["auroc"], 5)
          for name in (selected, "fixed_unsupervised", "graph", "shuffled", "isolation")}, flush=True)


def score_task(args, task):
    directory = args.output / task
    if (directory / "test_scores.npz").exists():
        raise FileExistsError("Test scores are frozen; do not overwrite")
    pack, metadata = load_inputs(args.packs, task, "test")
    fitted = read_json(directory / "fit.json")
    if (set(fitted["fit_sources"]) | set(fitted["dev_sources"])) & set(metadata["sources"]):
        raise ValueError("Test source overlaps model fitting or development")
    models = joblib.load(directory / "models.joblib")
    selected = read_json(directory / "selection.json")
    scores = fusion_scores(rank_scores(raw_scores(models, pack), models["ranks"]))
    scores["selected_dev"] = scores[selected["selected"]].copy()
    scores["selected_mixed_threshold"] = scores["selected_dev"].copy()
    write_arrays(directory / "test_scores.npz", **scores)
    write_json(directory / "test_frozen.json", dict(tokens=len(pack["target"]),
        responses=len(metadata["records"]), sources=len(metadata["sources"]), test_labels_used=False))


def thresholds(directory):
    selected = read_json(directory / "selection.json")
    result = dict(selected["mixed_thresholds"])
    result["selected_dev"] = selected["thresholds"][selected["selected"]]
    result["selected_mixed_threshold"] = selected["mixed_thresholds"][selected["selected"]]
    return result


def old_baselines(args, task):
    directory = args.packs.parent / task
    with np.load(directory / "test_scores.npz", allow_pickle=False) as saved:
        scores = {name: saved[name] for name in ("source_refine", "conditioned", "trees")}
    limits = read_json(directory / "selection.json")["thresholds"]
    limits.update(read_json(directory / "readout_selection.json")["thresholds"])
    return scores, {name: limits[name] for name in scores}


def evaluate_task(args, task):
    directory = args.output / task
    pack, metadata = load_inputs(args.packs, task, "test")
    pack.update(evaluation_labels(Path(metadata["source_cache"]), pack, metadata))
    stored = read_arrays(directory / "test_scores.npz")
    scores = {name: stored[name] for name in REPORTED}
    limits = thresholds(directory)
    old, old_limits = old_baselines(args, task)
    scores.update(old)
    limits.update(old_limits)
    metrics = evaluate_all(pack, scores, limits)
    for name in REPORTED:
        if name != "selected_dev":
            metrics[name]["threshold_rule"] = "unlabelled_dev_mixture_95th_percentile_strict_gt"
    write_json(directory / "test_metrics.json", metrics)
    intervals = {}
    for candidate, baseline in (("selected_dev", "source_refine"), ("selected_dev", "conditioned"),
                                ("graph", "shuffled"), ("selected_dev", "fixed_unsupervised")):
        intervals[f"{candidate}-{baseline}"] = source_bootstrap(pack, scores[candidate], scores[baseline])
    write_json(directory / "bootstrap.json", intervals)
    by_generator = {}
    for generator in sorted({row["generator"] for row in metadata["records"]}):
        answers = [index for index, row in enumerate(metadata["records"]) if row["generator"] == generator]
        mask = np.isin(pack["answer_index"], answers)
        by_generator[generator] = evaluate_all(subset(pack, mask),
            {name: value[mask] for name, value in scores.items()}, limits, detailed=False)
    write_json(directory / "test_by_generator.json", by_generator)
    print("TEST", task, {name: round(row["auroc"], 6) for name, row in metrics.items()}, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "pilot", "select", "score", "evaluate", "cases"), required=True)
    parser.add_argument("--packs", type=Path, default=Path("outputs/probabilistic_detection_20260928_full/packs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.phase == "evaluate" or (args.phase == "cases" and args.include_test):
        for task in TASKS:
            read_json(args.output / task / "test_frozen.json")
    if args.phase == "cases":
        from .report import cases
        cases(args)
        return
    for task in args.tasks:
        if args.phase == "fit":
            fit_task(args, task)
        elif args.phase in ("pilot", "select"):
            develop_task(args, task, fuse=args.phase == "select")
        elif args.phase == "score":
            score_task(args, task)
        else:
            evaluate_task(args, task)
