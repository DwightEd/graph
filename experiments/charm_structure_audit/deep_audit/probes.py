"""Source-disjoint SUPERVISED diagnostic readouts; the original model is frozen."""

import json
from pathlib import Path
import zlib

import numpy as np
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..graph import transform_graph
from .capture import read_graph
from .common import metric, roles, write_csv, write_json
from .controls import model_graph
from .representations import features, trace_model
from .scores import within_answers


def probe_records(samples, checkpoint, prepared=None):
    """Use the parent's actual fit partition; never invent a token-random split."""
    training_path = Path(checkpoint).parent / "training.json"
    if not training_path.exists():
        return None, "Parent training.json is absent; historical training membership is unknown."
    training = json.loads(training_path.read_text())
    if prepared is None:
        prepared = Path(samples[0]["graph"]).parents[2]
    index_path = Path(prepared) / "index.json"
    records = json.loads(index_path.read_text())
    lookup = {str(row["id"]): row for row in records}
    parts = {name: [lookup[str(identity)] for identity in identities]
             for name, identities in training["partitions"].items()}
    sources = {name: {str(row["source_id"]) for row in rows} for name, rows in parts.items()}
    names = list(sources)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            if sources[name] & sources[other]:
                raise ValueError("Source overlap in parent partitions: " + name + " / " + other)
    test_sources = {sample["source_id"] for sample in samples}
    known_test = {str(row["id"]) for row in parts["test"]}
    if test_sources & sources["fit"] or not {s["id"] for s in samples} <= known_test:
        raise ValueError("Probe evaluation is not the parent's held-out test set")
    return parts["fit"], None


def fit_feature_cache(records, model, settings, directory, prepared=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for record in tqdm(records, desc="frozen fit-set features", unit="answer"):
        destination = directory / (str(record["id"]) + ".npz")
        graph, graph_path = read_graph(record, prepared)
        stamp = [graph_path.stat().st_size, graph_path.stat().st_mtime_ns]
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if saved["graph_stamp"].tolist() != stamp:
                    raise ValueError("Stale fit-feature cache: " + str(destination))
        else:
            seed = zlib.crc32(str(record["id"]).encode()) + settings["seed"]
            view, _ = transform_graph(model_graph(graph), settings["variant"], seed)
            _, states = trace_model(model, view)
            values = {"feature_" + name: array for name, array in features(view, states).items()}
            with destination.with_suffix(".partial").open("wb") as stream:
                np.savez_compressed(stream, **values, gold=graph["gold"], graph_stamp=np.asarray(stamp))
            destination.with_suffix(".partial").replace(destination)
            del view, states, values
        paths.append(destination)
        del graph
    return paths


def train_readouts(paths, epochs=10, seed=42):
    """Stream one answer at a time; scaling and all fitting use fit sources only."""
    with np.load(paths[0], allow_pickle=False) as saved:
        names = sorted(key for key in saved.files if key.startswith("feature_"))
    scalers = {name: StandardScaler() for name in names}
    positive = total = 0
    for path in tqdm(paths, desc="fit-only scaling", unit="answer"):
        with np.load(path, allow_pickle=False) as saved:
            positive += int(saved["gold"].sum())
            total += len(saved["gold"])
            for name in names:
                scalers[name].partial_fit(saved[name])
    if not 0 < positive < total:
        raise ValueError("Diagnostic probe fit set must contain both classes")
    positive_weight = min((total - positive) / positive, 8.)
    models = {name: SGDClassifier(loss="log_loss", alpha=1e-4, average=True, random_state=seed)
              for name in names}
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(paths))
        for index in tqdm(order, desc=f"diagnostic linear readouts {epoch + 1}/{epochs}", unit="answer"):
            with np.load(paths[index], allow_pickle=False) as saved:
                gold = saved["gold"].astype(int)
                weight = np.where(gold, positive_weight, 1.)
                for name in names:
                    values = scalers[name].transform(saved[name]).astype(np.float64)
                    models[name].partial_fit(values, gold, classes=np.array([0, 1]), sample_weight=weight)
    return models, scalers


def ranking_report(samples, key):
    gold = np.concatenate([s["gold"] for s in samples])
    score = np.concatenate([s[key] for s in samples])
    fields = ("tokens", "positive_tokens", "negative_tokens", "prevalence", "auroc", "ap")
    measured = metric(gold, score, np.inf)
    result = {name: measured[name] for name in fields}
    result["within_answer"] = within_answers(samples, np.inf, key)[0]
    result["roles_vs_all_normal"] = {}
    for role in ("first_error", "later_onset", "continuation"):
        selected = np.concatenate([roles(s)[role] | ~s["gold"].astype(bool) for s in samples])
        positive = np.concatenate([roles(s)[role] for s in samples])
        measured = metric(positive[selected], score[selected], np.inf)
        result["roles_vs_all_normal"][role] = {name: measured[name] for name in fields}
    return result


def run_probes(samples, model, settings, checkpoint, output, prepared=None, epochs=10):
    output = Path(output)
    records, reason = probe_records(samples, checkpoint, prepared)
    if records is None:
        result = dict(status="unavailable", reason=reason)
        write_json(output / "probes.json", result)
        return result
    paths = fit_feature_cache(records, model, settings, output / "fit_features", prepared)
    models, scalers = train_readouts(paths, epochs)
    enriched = []
    score_dir = output / "probe_scores"
    score_dir.mkdir(exist_ok=True)
    for sample in samples:
        values = {}
        with np.load(output / "captures" / (sample["id"] + ".npz"), allow_pickle=False) as saved:
            for name in models:
                feature = scalers[name].transform(saved[name]).astype(np.float64)
                values["probe_" + name[8:]] = expit(models[name].decision_function(feature))
        np.savez_compressed(score_dir / (sample["id"] + ".npz"), **values)
        enriched.append(dict(sample, **values))
    result = dict(status="completed", purpose="supervised diagnostic readouts, not unsupervised detection",
                  fit_answers=len(records), fit_sources=len({r["source_id"] for r in records}), epochs=epochs,
                  threshold=None, full_attention_entropy="unavailable in prepared graph; never filled with zeros",
                  original=ranking_report(samples, "score"),
                  probes={name[8:]: ranking_report(enriched, "probe_" + name[8:]) for name in models},
                  note="Equal linear-readout recipe; representation quality, NOT a capacity-matched GNN ablation. Retained entropy omits discarded weights.")
    write_json(output / "probes.json", result)
    coefficients = []
    parameters = {}
    for name in models:
        parameters[name + "_coefficient"] = models[name].coef_
        parameters[name + "_intercept"] = models[name].intercept_
        parameters[name + "_mean"] = scalers[name].mean_
        parameters[name + "_scale"] = scalers[name].scale_
        for index, weight in enumerate(models[name].coef_[0]):
            coefficients.append(dict(feature=name[8:], coordinate=index, standardized_weight=float(weight),
                                     raw_unit_weight=float(weight / scalers[name].scale_[index])))
    np.savez_compressed(output / "linear_readouts.npz", **parameters)
    write_csv(output / "probe_coefficients.csv", coefficients)
    return result
