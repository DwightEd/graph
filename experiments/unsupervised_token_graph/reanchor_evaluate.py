"""Evaluate frozen reanchor scores against RAGTruth token annotations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .lookback import InformationDiagnostics


def _auc(labels, scores):
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(order), dtype=float); ranks[order] = np.arange(len(order)) + 1
    positives = labels == 1; negatives = labels == 0
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) /
                 (positives.sum() * negatives.sum()))


def _ap(labels, scores):
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]; positions = np.flatnonzero(ranked == 1)
    return float(np.cumsum(ranked)[positions].dot(1 / (positions + 1)) / ranked.sum())


def _labels(annotation):
    offsets = np.asarray(annotation["offsets"])
    values = np.zeros(len(offsets), dtype=int)
    for span in annotation["labels"]:
        values[(offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])] = 1
    return values


def evaluate(prediction_dir, metadata_path, annotations_path):
    metadata = {}
    for line in Path(metadata_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line); metadata[row["trace"][:-4]] = row
    annotations = {}
    for line in Path(annotations_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        annotations[hashlib.sha256(row["response"].encode()).hexdigest()] = row
    fields = {"event": [], "event_strength": [], "lookback_ratio": [], "waad": [], "fai": [], "distribution_shift": []}
    labels, source_ids, events = [], [], []
    for file in sorted(Path(prediction_dir).glob("*.npz")):
        meta = metadata[file.stem]
        if "response_sha256" in meta:
            annotation = annotations.get(meta["response_sha256"])
        else:
            annotation = next((row for row in annotations.values() if row["response"] == meta["response"]), None)
        if annotation is None:
            raise ValueError(
                f"{file.stem} is not identity-bound to RAGTruth; "
                "reanchor evaluation requires the exact annotated response, not a same-source response"
            )
        with np.load(file) as values:
            length = len(values["event"]); target = _labels(annotation)
            if len(target) != length:
                raise ValueError(f"token/label length mismatch: {file.stem}")
            labels.extend(target); source_ids.extend([meta["source_id"]] * length)
            events.extend(values["event"].astype(bool))
            for name in fields: fields[name].extend(values[name])
    labels = np.asarray(labels); source_ids = np.asarray(source_ids); events = np.asarray(events)
    result = {"tokens": len(labels), "positives": int(labels.sum()), "responses": len(set(metadata))}
    metrics = {}
    for name, values in fields.items():
        values = np.asarray(values, dtype=float)
        metrics[name] = {"auroc": _auc(labels, values), "ap": _ap(labels, values),
                         "information": InformationDiagnostics.histogram_kl(values[labels == 0], values[labels == 1])}
    result["metrics"] = metrics
    result["event_enrichment"] = {
        "error_rate_event": float(labels[events].mean()) if events.any() else None,
        "error_rate_no_event": float(labels[~events].mean()) if (~events).any() else None,
        "event_tokens": int(events.sum()),
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.predictions, args.metadata, args.annotations)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "tokens": result["tokens"], "positives": result["positives"]}))


if __name__ == "__main__":
    main()