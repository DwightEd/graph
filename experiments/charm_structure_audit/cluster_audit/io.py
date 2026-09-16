"""Read finalized original predictions/graphs and verified optional captures."""

import json
from pathlib import Path

import numpy as np

from .matching import merged_spans


GRAPH_FIELDS = ("x", "edge_index", "edge_attr", "edge_mark", "prompt_length", "layers", "heads", "token_ids")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_predictions(root, completed_only=False):
    root = Path(root)
    settings = read_json(root / "prediction_settings.json")
    paths = sorted((root / "samples").glob("*.npz")) if completed_only else [
        root / "samples" / Path(path).name for path in read_json(root / "predictions.json")]
    samples = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            sample = json.loads(str(saved["record_json"]))
            fields = [k for k in saved.files if k in ("score", "gold", "spans", "offsets", "response", "onset") or k.startswith("score_")]
            sample.update({key: saved[key] for key in fields})
        sample.update(id=str(sample["id"]), source_id=str(sample["source_id"]),
                      response=str(sample["response"]), prediction_file=str(path))
        samples.append(sample)
    validate_samples(samples)
    return samples, settings


def validate_samples(samples):
    if not samples or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Need nonempty finalized predictions with unique response IDs")
    for sample in samples:
        count = len(sample["gold"])
        if sample["offsets"].shape != (count, 2):
            raise ValueError("Prediction offsets do not align")
        union = np.zeros(count, bool)
        for start, end in sample["spans"]:
            if not 0 <= start < end <= count:
                raise ValueError("Invalid gold interval")
            union[start:end] = True
        if not np.array_equal(union, sample["gold"].astype(bool)):
            raise ValueError("Saved spans disagree with token labels")
        for key in ("score", *(k for k in sample if k.startswith("score_"))):
            if np.shape(sample[key]) != (count,):
                raise ValueError("Score/token shape mismatch: " + key)
        if not np.isfinite(sample["score"]).all():
            raise ValueError("Baseline must have a finite score for every token")


def verify_identity(sample, other):
    for key in ("id", "source_id", "response"):
        if str(sample[key]) != str(other[key]):
            raise ValueError("Identity mismatch: " + key)
    for key in ("gold", "offsets", "spans"):
        if not np.array_equal(sample[key], other[key]):
            raise ValueError("Token alignment mismatch: " + key)


def load_graph(sample, prepared):
    path = Path(sample["graph"]) if prepared is None else (
        Path(prepared) / "graphs" / sample["split"] / (sample["id"] + ".npz"))
    with np.load(path, allow_pickle=False) as saved:
        identity = json.loads(str(saved["record_json"]))
        identity.update({k: saved[k] for k in ("response", "gold", "offsets", "spans")})
        verify_identity(sample, identity)
        graph = {key: saved[key] for key in GRAPH_FIELDS}
    prompt = int(graph["prompt_length"])
    source, target = graph["edge_index"]
    if len(graph["x"]) - prompt != len(sample["gold"]) or len(graph["token_ids"]) != len(graph["x"]):
        raise ValueError("Graph nodes/token IDs are not aligned")
    if np.any(source >= target) or np.any(target < prompt) or np.any(source < 0) or np.any(target >= len(graph["x"])):
        raise ValueError("Expected strictly past-to-response graph")
    return graph, path


def load_capture(sample, graph_path, directory):
    """Current main deep_audit/captures schema; do NOT merge by filename alone."""
    path = Path(directory) / (sample["id"] + ".npz")
    if not path.exists():
        return {}, {}, dict(status="not_available")
    with np.load(path, allow_pickle=False) as saved:
        stamp = [graph_path.stat().st_size, graph_path.stat().st_mtime_ns]
        if saved["graph_stamp"].tolist() != stamp:
            raise ValueError("Extra capture refers to a different prepared graph")
        if not np.allclose(saved["replay_score"], sample["score"], rtol=0, atol=2e-5):
            raise ValueError("Extra capture baseline does not replay original scores")
        scores = {"capture_" + k: saved[k] for k in saved.files if k.startswith("score_")}
        states = {k[8:]: saved[k] for k in saved.files if k == "feature_projected" or k.startswith("feature_layer_")}
    for values in (*scores.values(), *states.values()):
        if len(values) != len(sample["gold"]):
            raise ValueError("Capture token axis does not match predictions")
    return scores, states, dict(status="verified", file=str(path), scores=list(scores), states=list(states))


def read_comparisons(arguments, samples, baseline_root):
    result, thresholds, notes = {}, {}, {}
    base_training = Path(baseline_root).parent / "training.json"
    base_recipe = read_json(base_training) if base_training.exists() else None
    for argument in arguments:
        name, directory = argument.split("=", 1)
        key = "model_" + name
        if key in result:
            raise ValueError("Duplicate comparison model name")
        other, settings = read_predictions(directory)
        mapping = {s["id"]: s for s in other}
        result[key] = {}
        for sample in samples:
            if sample["id"] in mapping:
                verify_identity(sample, mapping[sample["id"]])
                result[key][sample["id"]] = mapping[sample["id"]]["score"]
        thresholds[key] = float(settings["threshold"]["value"])
        training = Path(directory).parent / "training.json"
        recipe = read_json(training) if training.exists() else None
        ignored = {"variant"}
        recipe_same = ({k: v for k, v in base_recipe.items() if k not in ignored} ==
                       {k: v for k, v in recipe.items() if k not in ignored}) if base_recipe and recipe else None
        notes[key] = dict(directory=directory, matched_answers=len(result[key]),
                          threshold=settings["threshold"], same_recipe_except_variant=recipe_same,
                          interpretation="independent_training_comparison" if recipe_same else "prediction_comparison_not_isolated_graph_gain")
    return result, thresholds, notes


def population(samples):
    tokens = sum(len(s["gold"]) for s in samples)
    errors = sum(int(s["gold"].sum()) for s in samples)
    first = sum(bool(s["gold"].any()) for s in samples)
    spans = [p for s in samples for p in merged_spans(s["spans"])]
    return dict(answers=len(samples), sources=len({s["source_id"] for s in samples}),
                tokens=tokens, error_tokens=errors, error_fraction=errors / tokens,
                first_error_tokens=first, first_error_fraction_all=first / tokens,
                first_error_fraction_error=first / errors if errors else None,
                mapped_spans=len(spans), singletons=sum(b - a == 1 for a, b in spans),
                mult_token_spans=sum(b - a >= 2 for a, b in spans),
                scope="saved evaluated samples, not the complete RAGTruth corpus")
