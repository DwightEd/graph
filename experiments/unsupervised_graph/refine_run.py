"""CPU scalar refinement with label-free calibration and explicit dev selection."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import evaluate_all, source_bootstrap, subset
from .data import load_inputs
from .refinement import PRIMARY, STRICT, observations, score_candidates
from .scalar import fit_ranks


TASKS = ("QA", "Summary", "Data2txt")
REPORTED = ("pair", "full", "fixed_unsupervised", PRIMARY, STRICT, "selected_dev")
THRESHOLD_RULE = "unlabelled_dev_mixture95_strict_gt; lexicographic_cutoff_for_tie"


def prepare(args, task):
    directory = args.output / task
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pack, metadata = load_inputs(args.packs, task, "train")
    train = subset(pack, ~pack["development"])
    dev = subset(pack, pack["development"])
    fit_raw = observations(train)
    references = fit_ranks(fit_raw)
    scores, thresholds, calibration = score_candidates(dev, observations(dev), references)
    write_arrays(directory / "references.npz", **references)
    write_arrays(directory / "development_scores.npz", **scores)
    write_json(directory / "calibration.json", calibration)
    write_json(directory / "development_thresholds.json", thresholds)
    fit_scores, fit_thresholds, _ = score_candidates(train, fit_raw, references, calibration)
    write_arrays(directory / "fit_scores.npz", **fit_scores)
    write_json(directory / "fit_thresholds.json", fit_thresholds)
    fit_sources = sorted({r["source_id"] for r in metadata["records"] if r["partition"] == "fit"})
    dev_sources = sorted({r["source_id"] for r in metadata["records"] if r["partition"] == "dev"})
    if set(fit_sources) & set(dev_sources):
        raise ValueError("Fitting and development sources overlap")
    write_json(directory / "prepare.json", dict(seconds=time.monotonic() - started,
        fit_sources=fit_sources, dev_sources=dev_sources, labels_used=False,
        primary=PRIMARY, strict=STRICT, candidates=list(scores),
        fit_tokens=len(train["target"]), dev_tokens=len(dev["target"])))
    print("PREPARED", task, len(scores), "candidates", flush=True)


def select(args, task):
    directory = args.output / task
    if (args.output / "SCORING_FREEZE.json").exists() or (directory / "selection.json").exists():
        raise FileExistsError("Preserve frozen selection; use a new output directory")
    pack, _ = load_inputs(args.packs, task, "train")
    dev = subset(pack, pack["development"])
    with np.load(args.packs / f"{task}_train.npz", allow_pickle=False) as saved:
        dev["labels"] = saved["labels"][pack["development"]]
    scores = read_arrays(directory / "development_scores.npz")
    limits = read_json(directory / "development_thresholds.json")
    metrics = evaluate_all(dev, scores, limits, detailed=False)
    selected = max(metrics, key=lambda name: metrics[name]["auroc"])
    write_json(directory / "selection.json", dict(selected=selected, candidates=metrics,
        labels_used_for_selection=True, labels_used_for_fitting=False,
        labels_used_for_threshold=False, fixed_unsupervised_primary=PRIMARY))
    print("SELECTED", task, selected, metrics[selected]["auroc"], flush=True)


def freeze(args):
    hashes, selected = {}, {}
    for task in TASKS:
        directory = args.output / task
        selected[task] = read_json(directory / "selection.json")["selected"]
        for name in ("references.npz", "calibration.json", "selection.json"):
            hashes[f"{task}/{name}"] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    path = args.output / "SCORING_FREEZE.json"
    if path.exists():
        previous = read_json(path)
        if hashes != previous["hashes"]:
            raise ValueError("Scoring inputs changed after freeze")
    else:
        write_json(path, dict(hashes=hashes, selected=selected, primary=PRIMARY,
            test_labels_used=False, exploratory=True))


def score(args, task):
    directory = args.output / task
    if (directory / "test_scores.npz").exists():
        raise FileExistsError("Preserve frozen test scores")
    pack, metadata = load_inputs(args.packs, task, "test")
    prepared = read_json(directory / "prepare.json")
    if (set(prepared["fit_sources"]) | set(prepared["dev_sources"])) & set(metadata["sources"]):
        raise ValueError("Test sources overlap fitting/development")
    scores, limits, _ = score_candidates(pack, observations(pack),
        read_arrays(directory / "references.npz"), read_json(directory / "calibration.json"))
    selected = read_json(directory / "selection.json")["selected"]
    scores["selected_dev"] = scores[selected].copy()
    limits["selected_dev"] = limits[selected]
    write_arrays(directory / "test_scores.npz", **scores)
    write_json(directory / "test_thresholds.json", limits)
    write_json(directory / "test_frozen.json", dict(tokens=len(pack["target"]),
        responses=len(metadata["records"]), sources=len(metadata["sources"]), labels_used=False))
    print("SCORED", task, len(pack["target"]), flush=True)


def evaluate(args, task):
    directory = args.output / task
    pack, metadata = load_inputs(args.packs, task, "test")
    pack.update(evaluation_labels(Path(metadata["source_cache"]), pack, metadata))
    stored = read_arrays(directory / "test_scores.npz")
    scores = {name: stored[name] for name in REPORTED}
    limits = read_json(directory / "test_thresholds.json")
    old = args.packs.parent / task
    with np.load(old / "test_scores.npz", allow_pickle=False) as saved:
        for name in ("source_refine", "conditioned", "trees"):
            scores[name] = saved[name]
    limits.update(read_json(old / "selection.json")["thresholds"])
    limits.update(read_json(old / "readout_selection.json")["thresholds"])
    previous = args.previous / task
    with np.load(previous / "test_scores.npz", allow_pickle=False) as saved:
        scores["previous_selected_dev"] = saved["selected_dev"]
    previous_selection = read_json(previous / "selection.json")
    limits["previous_selected_dev"] = previous_selection["mixed_thresholds"][previous_selection["selected"]]
    metrics = evaluate_all(pack, scores, limits)
    for name in (*REPORTED, "previous_selected_dev"):
        metrics[name]["threshold_rule"] = THRESHOLD_RULE
    write_json(directory / "test_metrics.json", metrics)
    write_json(directory / "all_candidates_test.json", evaluate_all(pack, stored,
        read_json(directory / "test_thresholds.json"), detailed=False))
    intervals = {}
    for candidate, baseline in ((PRIMARY, "fixed_unsupervised"), (STRICT, "pair"),
                                ("selected_dev", "fixed_unsupervised"), ("selected_dev", "source_refine"),
                                ("selected_dev", "previous_selected_dev")):
        intervals[f"{candidate}-{baseline}"] = source_bootstrap(pack, scores[candidate], scores[baseline])
    write_json(directory / "bootstrap.json", intervals)
    grouped = {}
    for generator in sorted({r["generator"] for r in metadata["records"]}):
        answers = [i for i, r in enumerate(metadata["records"]) if r["generator"] == generator]
        mask = np.isin(pack["answer_index"], answers)
        grouped[generator] = evaluate_all(subset(pack, mask),
            {name: values[mask] for name, values in scores.items()}, limits, detailed=False)
    write_json(directory / "test_by_generator.json", grouped)
    print("TEST", task, {name: round(row["auroc"], 6) for name, row in metrics.items()}, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "select", "score", "evaluate", "cases"), required=True)
    parser.add_argument("--packs", type=Path, default=Path("outputs/probabilistic_detection_20260928_full/packs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("outputs/unsupervised_graph_20260928"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args(argv)
    if args.phase in ("score", "evaluate") or args.include_test:
        freeze(args)
    if args.phase == "evaluate" or args.include_test:
        for task in TASKS:
            read_json(args.output / task / "test_frozen.json")
    if args.phase == "cases":
        from .refine_report import cases
        cases(args)
        return
    phases = dict(prepare=prepare, select=select, score=score, evaluate=evaluate)
    for task in args.tasks:
        phases[args.phase](args, task)
