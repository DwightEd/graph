"""Supervised source-disjoint likelihood-ratio experiments on frozen native caches."""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from .data import BASELINES, evaluation_labels, load_pack, prepare_task, source_weights
from .evaluation import evaluate_all, source_bootstrap, subset
from .model import ConditionalGaussian
from .iteration import fit_readouts, score_readouts, freeze_detector


GAUSSIANS = (("full", .5), ("full", .1), ("full", .9), ("diagonal", .5), ("shared", .5))
TEMPERATURES = (.1, .25, .5, 1.)
PRIMARY = "full_s0.5_t0.25"


def gaussian_name(covariance, shrinkage):
    return f"{covariance}_s{shrinkage:g}"


def append_gaussian_scores(scores, name, components):
    scores["context_prior"] = components["anchor_logit"]
    for temperature in TEMPERATURES:
        scores[f"{name}_t{temperature:g}"] = components["anchor_logit"] + temperature * components["conditional_log_ratio"]


def feature_views(pack, gaussian):
    raw = np.column_stack((pack["context"], pack["observations"]))
    matched = np.column_stack((gaussian._context_design(pack["context"])[:, 1:],
        gaussian.observation_transform_.transform(pack["observations"])))
    return raw, matched


def fit_baselines(train, dev, gaussian, weights):
    raw_train, matched_train = feature_views(train, gaussian)
    raw_dev, matched_dev = feature_views(dev, gaussian)
    scaler = StandardScaler().fit(raw_train, sample_weight=weights)
    logistic = LogisticRegression(C=1., max_iter=1500, random_state=42)
    logistic.fit(scaler.transform(raw_train), train["labels"], sample_weight=weights)
    matched = LogisticRegression(C=1., max_iter=1500, random_state=42)
    matched.fit(matched_train, train["labels"], sample_weight=weights)
    trees = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
        min_samples_leaf=50, l2_regularization=1., learning_rate=.08,
        early_stopping=False, random_state=42)
    trees.fit(raw_train, train["labels"], sample_weight=weights)
    scores = dict(logistic=logistic.decision_function(scaler.transform(raw_dev)),
        logistic_matched=matched.decision_function(matched_dev), trees=trees.decision_function(raw_dev))
    models = dict(scaler=scaler, logistic=logistic, logistic_matched=matched, trees=trees)
    return models, scores


def existing_scores(pack):
    return {name: pack["baselines"][:, index] for index, name in enumerate(BASELINES)
            if np.isfinite(pack["baselines"][:, index]).all()}


def fit_task(args, task):
    started = time.monotonic()
    pack, metadata = load_pack(args.output / "packs", task, "train")
    train, dev = subset(pack, ~pack["development"]), subset(pack, pack["development"])
    weights = source_weights(train["source_index"])
    models, scores, components = {}, existing_scores(dev), {}
    for covariance, shrinkage in GAUSSIANS:
        name = gaussian_name(covariance, shrinkage)
        print(f"{task}: fit {name}, train={len(train['labels'])}, dev={len(dev['labels'])}", flush=True)
        model = ConditionalGaussian(covariance, shrinkage).fit(
            train["context"], train["observations"], train["labels"], weights)
        models[name] = model
        components[name] = model.predict_components(dev["context"], dev["observations"])
        append_gaussian_scores(scores, name, components[name])
    primary_model = models["full_s0.5"]
    competitors, other_scores = fit_baselines(train, dev, primary_model, weights)
    models.update(competitors)
    scores.update(other_scores)
    finish_development(args, task, train, dev, metadata, models, scores, components, started)


def finish_development(args, task, train, dev, metadata, models, scores, components, started):
    simple = evaluate_all(dev, scores, detailed=False)
    gaussian_candidates = ["context_prior", *(name for name in scores if "_t" in name and "_s" in name)]
    selected = max(gaussian_candidates, key=lambda name: simple[name]["auroc"])
    scores["selected_gaussian"] = scores[selected].copy()
    reported = [*BASELINES, PRIMARY, "diagonal_s0.5_t0.25", "shared_s0.5_t0.25",
                "context_prior", "selected_gaussian", "logistic", "logistic_matched", "trees"]
    detailed = evaluate_all(dev, {name: scores[name] for name in reported if name in scores})
    directory = args.output / task
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, directory / "models.joblib")
    write_arrays(directory / "development_scores.npz", **scores)
    frozen = dict(task=task, selected=selected, primary=PRIMARY, candidates=simple,
        thresholds={name: row["threshold"] for name, row in simple.items()},
        fit_sources=sorted({r["source_id"] for r in metadata["records"] if r["partition"] == "fit"}),
        dev_sources=sorted({r["source_id"] for r in metadata["records"] if r["partition"] == "dev"}),
        supervised=True, test_labels_used=False, seconds=time.monotonic() - started)
    frozen["thresholds"]["selected_gaussian"] = frozen["thresholds"][selected]
    write_json(directory / "selection.json", frozen)
    write_json(directory / "development_metrics.json", detailed)
    matrices = {name: model.covariances_ for name, model in models.items() if isinstance(model, ConditionalGaussian)}
    write_arrays(directory / "covariances.npz", **matrices)
    print(task, {k: round(v["auroc"], 5) for k, v in detailed.items()}, flush=True)


def score_task(args, task):
    directory = args.output / task
    if (directory / "test_scores.npz").exists():
        raise FileExistsError("Test predictions already frozen; use a new experiment directory")
    pack, metadata = load_pack(args.output / "packs", task, "test")
    models = joblib.load(directory / "models.joblib")
    selected = read_json(directory / "selection.json")
    occupied = set(selected["fit_sources"]) | set(selected["dev_sources"])
    if occupied & set(metadata["sources"]):
        raise ValueError("Official test sources overlap supervised fitting/selection")
    scores = existing_scores(pack)
    for covariance, shrinkage in GAUSSIANS:
        name = gaussian_name(covariance, shrinkage)
        result = models[name].predict_components(pack["context"], pack["observations"])
        append_gaussian_scores(scores, name, result)
    raw, matched = feature_views(pack, models["full_s0.5"])
    scores.update(logistic=models["logistic"].decision_function(models["scaler"].transform(raw)),
        logistic_matched=models["logistic_matched"].decision_function(matched),
        trees=models["trees"].decision_function(raw))
    scores["selected_gaussian"] = scores[selected["selected"]].copy()
    if (directory / "readout_selection.json").exists():
        scores.update(score_readouts(directory, pack))
    if (directory / "detector_selection.json").exists():
        deployment = read_json(directory / "detector_selection.json")
        scores["selected_detector"] = scores[deployment["selected"]].copy()
    write_arrays(directory / "test_scores.npz", **scores)
    write_json(directory / "test_frozen.json", dict(status="all_predictions_frozen", tokens=len(raw),
        responses=len(metadata["records"]), sources=len(metadata["sources"]), test_labels_used=False))


def evaluate_task(args, task):
    directory = args.output / task
    read_json(directory / "test_frozen.json")
    pack, metadata = load_pack(args.output / "packs", task, "test")
    pack.update(evaluation_labels(args.cache, pack, metadata))
    scores = read_arrays(directory / "test_scores.npz")
    selection = read_json(directory / "selection.json")
    methods = [*BASELINES, PRIMARY, "diagonal_s0.5_t0.25", "shared_s0.5_t0.25",
               "context_prior", "selected_gaussian", "logistic", "logistic_matched", "trees"]
    readouts = directory / "readout_selection.json"
    if readouts.exists():
        iteration = read_json(readouts)
        methods.extend(iteration["reported"])
        selection["thresholds"].update(iteration["thresholds"])
    detector = directory / "detector_selection.json"
    if detector.exists():
        methods.append("selected_detector")
        selection["thresholds"]["selected_detector"] = read_json(detector)["threshold"]
    measured = evaluate_all(pack, {name: scores[name] for name in methods}, selection["thresholds"])
    write_json(directory / "test_metrics.json", measured)
    comparisons = {}
    pairs = [("selected_gaussian", "source_refine"),
             (PRIMARY, "diagonal_s0.5_t0.25"), ("trees", "source_refine")]
    if readouts.exists():
        pairs.extend(("selected_readout", name) for name in ("source_refine", "linear", "trees"))
        pairs.extend((("interactions", "squares"), ("conditioned", "interactions"),
                      (PRIMARY, "shared_correlation")))
    if detector.exists():
        pairs.append(("selected_detector", "source_refine"))
    for candidate, baseline in pairs:
        comparisons[f"{candidate}__{baseline}"] = source_bootstrap(
            pack, scores[candidate], scores[baseline], args.bootstrap)
    write_json(directory / "bootstrap.json", comparisons)
    generators = {row["generator"] for row in metadata["records"]}
    grouped = {}
    for generator in sorted(generators):
        ids = [i for i, row in enumerate(metadata["records"]) if row["generator"] == generator]
        mask = np.isin(pack["answer_index"], ids)
        grouped[generator] = evaluate_all(subset(pack, mask),
            {name: scores[name][mask] for name in methods}, selection["thresholds"], detailed=False)
    write_json(directory / "test_by_generator.json", grouped)
    print(task, {k: round(v["auroc"], 5) for k, v in measured.items()}, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("prepare-train", "develop", "iterate", "freeze", "prepare-test", "score", "evaluate"), required=True)
    parser.add_argument("--cache", type=Path, default=Path("outputs/native_support_ragtruth_all/source_first_v1"))
    parser.add_argument("--refine", type=Path, default=Path("outputs/native_support_ragtruth_all/source_refine_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["QA", "Summary", "Data2txt"])
    parser.add_argument("--train-source-limit", type=int)
    parser.add_argument("--dev-source-limit", type=int)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args(argv)
    for task in args.tasks:
        if args.phase.startswith("prepare-"):
            split = args.phase.removeprefix("prepare-")
            path = args.output / "packs"
            path.mkdir(parents=True, exist_ok=True)
            if (path / f"{task}_{split}.npz").exists():
                raise FileExistsError("Packed data already exists; reuse it without preparing again")
            prepare_task(args.cache, args.refine, path, task, split,
                         args.train_source_limit, args.dev_source_limit)
        elif args.phase == "develop":
            if (args.output / task / "selection.json").exists():
                raise FileExistsError("Fitted run already exists; use a new experiment directory")
            fit_task(args, task)
        elif args.phase == "score":
            score_task(args, task)
        elif args.phase == "iterate":
            fit_readouts(args, task)
        elif args.phase == "freeze":
            freeze_detector(args, task)
        else:
            if not all((args.output / item / "test_frozen.json").exists() for item in args.tasks):
                raise ValueError("Freeze every requested task before test evaluation")
            evaluate_task(args, task)


if __name__ == "__main__":
    main()
