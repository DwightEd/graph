"""RAGTruth-aligned reanchor proxy over the frozen S10 feature export.

S10 stores causal scalar observables rather than full attention. This module
therefore does not claim WAAD/FAI; it measures remote-history reallocation
relative to source mass and evaluates it after the score freeze.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def auc(labels, scores):
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(order), dtype=float); ranks[order] = np.arange(len(order)) + 1
    positive, negative = labels == 1, labels == 0
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) /
                 (positive.sum() * negative.sum()))


def average_precision(labels, scores):
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]; positions = np.flatnonzero(ranked == 1)
    return float(np.cumsum(ranked)[positions].dot(1 / (positions + 1)) / ranked.sum())


def label_tokens(annotation):
    offsets = np.asarray(annotation["offsets"])
    labels = np.zeros(len(offsets), dtype=int)
    for span in annotation["labels"]:
        labels[(offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])] = 1
    return labels


def proxy(values):
    entropy = values[:, 0].astype(float)
    source = values[:, 2].astype(float)
    remote = values[:, 3].astype(float)
    share = remote / np.maximum(source + remote, 1e-8)
    delta = np.r_[0., np.diff(share)]
    entropy_delta = np.r_[0., np.diff(entropy)]
    score = np.maximum(delta, 0.) * (1. + np.maximum(entropy_delta, 0.))
    valid = np.arange(len(score)) >= 2
    threshold = np.quantile(score[valid], .9) if valid.any() else np.inf
    event = valid & (score >= threshold) & (delta > 0)
    return {"remote_share": share, "remote_share_change": delta,
            "entropy_change": entropy_delta, "reanchor_proxy": score,
            "event": event}


def run(features, annotations, output):
    root, output = Path(features), Path(output)
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    labels = {}
    for line in Path(annotations).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        labels[str(row["id"])] = row
    all_scores = {name: [] for name in ("remote_share", "remote_share_change", "entropy_change", "reanchor_proxy")}
    all_labels, all_events, source_ids = [], [], []
    rows = []
    for record in tqdm(records, desc="s10 reanchor proxy", unit="response"):
        rid = str(record["id"]); path = root / f"{rid}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["artifact_sha256"]:
            raise ValueError(f"changed S10 feature artifact: {rid}")
        with np.load(path) as saved:
            values = saved["values"].copy()
            offsets = saved["offsets"].copy()
        if len(values) != record["tokens"] or len(offsets) != record["tokens"]:
            raise ValueError(f"token count mismatch: {rid}")
        annotation = labels[rid]
        if hashlib.sha256(annotation["response"].encode()).hexdigest() != record["response_sha256"]:
            raise ValueError(f"response identity mismatch: {rid}")
        y = label_tokens(annotation); scores = proxy(values)
        all_labels.extend(y); all_events.extend(scores["event"]); source_ids.extend([record["source_id"]] * len(y))
        for name in all_scores: all_scores[name].extend(scores[name])
        rows.append({"id": rid, "source_id": record["source_id"], "split": record["official_split"],
                     "tokens": len(y), "events": int(scores["event"].sum()), "errors": int(y.sum())})
    y, events = np.asarray(all_labels), np.asarray(all_events, bool)
    metrics = {}
    for name, values in all_scores.items():
        values = np.asarray(values)
        metrics[name] = {"auroc": auc(y, values), "ap": average_precision(y, values)}
    result = {"scope": {"method": "s10_remote_history_reanchor_proxy", "labels_read": True,
                         "full_attention_available": False, "responses": len(rows),
                         "tokens": len(y), "positives": int(y.sum())},
              "metrics": metrics,
              "event_enrichment": {"event_tokens": int(events.sum()),
                                   "error_rate_event": float(y[events].mean()),
                                   "error_rate_no_event": float(y[~events].mean())},
              "responses": rows}
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run(args.features, args.annotations, args.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()