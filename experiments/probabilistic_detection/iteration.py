"""M2 controlled development iteration; official test never selects a readout."""

import time

import joblib
from state_audit.storage import read_json, write_arrays, write_json

from .data import load_pack, source_weights
from .evaluation import evaluate_all, subset
from .model import ConditionalGaussian
from .readout import ABLATIONS, DESIGNS, REGULARIZATION, QuadraticReadout


def fit_readouts(args, task):
    directory = args.output / task
    if (directory / "readout_selection.json").exists():
        raise FileExistsError("Readouts already frozen; use a new experiment directory")
    started = time.monotonic()
    pack, _ = load_pack(args.output / "packs", task, "train")
    train, dev = subset(pack, ~pack["development"]), subset(pack, pack["development"])
    weights = source_weights(train["source_index"])
    transform = joblib.load(directory / "models.joblib")["full_s0.5"]
    models, scores = {}, {}
    for design in DESIGNS:
        matrix = QuadraticReadout(transform, design).matrix(train["context"], train["observations"])
        for regularization in REGULARIZATION:
            name = f"{design}_c{regularization:g}"
            print(f"{task}: fit {name}, columns={matrix.shape[1]}", flush=True)
            model = QuadraticReadout(transform, design, regularization).fit_matrix(matrix, train["labels"], weights)
            models[name] = model
            scores[name] = model.decision_function(dev["context"], dev["observations"])
    for ablation in ABLATIONS:
        print(f"{task}: fit {ablation}", flush=True)
        model = QuadraticReadout(transform, ablation=ablation)
        matrix = model.matrix(train["context"], train["observations"])
        model.fit_matrix(matrix, train["labels"], weights)
        models[ablation] = model
        scores[ablation] = model.decision_function(dev["context"], dev["observations"])
    print(f"{task}: fit shared correlation control", flush=True)
    shared = ConditionalGaussian("shared_correlation", .5).fit(
        train["context"], train["observations"], train["labels"], weights)
    models["shared_correlation"] = shared
    parts = shared.predict_components(dev["context"], dev["observations"])
    scores["shared_correlation"] = parts["anchor_logit"] + .25 * parts["conditional_log_ratio"]
    simple = evaluate_all(dev, scores, detailed=False)
    choices = {design: max((f"{design}_c{c:g}" for c in REGULARIZATION),
        key=lambda name: simple[name]["auroc"]) for design in DESIGNS}
    selected = max(choices.values(), key=lambda name: simple[name]["auroc"])
    choices["selected_readout"] = selected
    for alias, name in choices.items():
        scores[alias] = scores[name].copy()
    reported = [*choices, *ABLATIONS, "shared_correlation"]
    detailed = evaluate_all(dev, {name: scores[name] for name in reported})
    frozen = dict(selected=selected, choices=choices, candidates=simple,
        thresholds={name: row["threshold"] for name, row in simple.items()},
        reported=reported, supervised=True, test_labels_used=False, seconds=time.monotonic() - started)
    for alias, name in choices.items():
        frozen["thresholds"][alias] = frozen["thresholds"][name]
    joblib.dump(models, directory / "readouts.joblib")
    write_arrays(directory / "readout_development_scores.npz", **scores)
    write_json(directory / "readout_selection.json", frozen)
    write_json(directory / "readout_development_metrics.json", detailed)
    print(task, {k: round(v["auroc"], 5) for k, v in detailed.items()}, flush=True)


def score_readouts(directory, pack):
    models = joblib.load(directory / "readouts.joblib")
    selected = read_json(directory / "readout_selection.json")
    scores = {}
    for name, model in models.items():
        if name == "shared_correlation":
            parts = model.predict_components(pack["context"], pack["observations"])
            scores[name] = parts["anchor_logit"] + .25 * parts["conditional_log_ratio"]
        else:
            scores[name] = model.decision_function(pack["context"], pack["observations"])
    for alias, name in selected["choices"].items():
        scores[alias] = scores[name].copy()
    return scores


def freeze_detector(args, task):
    """Choose a deployable comparator by development AUROC, with honest provenance."""
    directory = args.output / task
    destination = directory / "detector_selection.json"
    if destination.exists() or (directory / "test_scores.npz").exists():
        raise FileExistsError("Detector or test predictions already frozen")
    metrics = read_json(directory / "development_metrics.json")
    metrics.update(read_json(directory / "readout_development_metrics.json"))
    names = ("logistic", "logistic_matched", "trees", "selected_gaussian", "selected_readout")
    selected = max(names, key=lambda name: metrics[name]["auroc"])
    write_json(destination, dict(selected=selected, candidates={name: metrics[name] for name in names},
        threshold=metrics[selected]["threshold"], supervised=True, test_labels_used=False,
        interpretation="Engineering selection among explicitly reported existing model families; not a novel estimator"))
    print(f"{task}: frozen detector={selected}, development AUROC={metrics[selected]['auroc']:.6f}", flush=True)
