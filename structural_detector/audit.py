"""Rescore frozen S10 outputs; no training or new LLM forwards.

python -m structural_detector.audit --features ... --predictions ... --annotations ... --output ...
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from binding_detector.evaluation import evaluate_records, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(predictions_dir, feature_dir):
    """Load all official-test rows without opening any annotation file."""
    root, features = Path(predictions_dir), Path(feature_dir)
    if not json.loads((root / "complete.json").read_text())["complete"]:
        raise ValueError("S10 run is not complete")
    freeze = json.loads((root / "prediction_freeze.json").read_text())
    for name, expected in freeze["hashes"].items():
        if sha(root / name) != expected:
            raise ValueError("frozen prediction artifact changed: " + name)
    manifest = json.loads((features / "manifest.json").read_text())
    if not manifest.get("complete") or sha(features / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("incomplete or changed feature export")
    protocol = json.loads((root / "protocol_freeze.json").read_text())
    if sha(features / "manifest.json") != protocol["feature_manifest_sha256"]:
        raise ValueError("features differ from the frozen S10 run")
    rows = list(map(json.loads, (features / "records.jsonl").read_text().splitlines()))
    metadata = {str(r["id"]): r for r in rows}
    if len(metadata) != len(rows):
        raise ValueError("duplicate feature ID")
    index = json.loads((root / "prediction_index.json").read_text())
    records, predictions = [], {}
    end, seen = 0, set()
    with np.load(root / "predictions.npz", allow_pickle=False) as saved:
        for row in index:
            if str(row["id"]) in seen or row["start"] != end or row["end"] <= end:
                raise ValueError("duplicate or non-contiguous prediction index")
            end = row["end"]
            seen.add(str(row["id"]))
            if row["split"] != "test":
                continue
            rid = str(row["id"])
            meta = metadata[rid]
            if str(meta["source_id"]) != str(row["source_id"]) or meta["official_split"] != "test":
                raise ValueError("prediction index source/split mismatch")
            file = features / (rid + ".npz")
            if sha(file) != meta["artifact_sha256"]:
                raise ValueError("changed token offsets: " + rid)
            with np.load(file, allow_pickle=False) as f:
                offsets = f["offsets"].copy()
            a, b = row["start"], row["end"]
            if b - a != len(offsets) or any(b > len(saved[k]) for k in saved.files):
                raise ValueError("prediction index length mismatch")
            records.append({**meta, "offsets": offsets.tolist()})
            predictions[rid] = {k: saved[k][a:b].copy() for k in saved.files}
        if any(len(saved[k]) != end for k in saved.files):
            raise ValueError("prediction index does not cover complete arrays")
    if not records:
        raise ValueError("no test predictions")
    return records, predictions


def audit(args):
    records, predictions = load_frozen(args.predictions, args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    # These are evaluations of the SAME frozen scores, not new calibrated heads.
    onset_views = ("span_onset_full_stream", "span_onset_vs_normal", "first_error_full_stream", "first_error_until_first")
    primary = {"default": "error__combined", **{name: "onset__combined" for name in onset_views}}
    controls = {"default": ("error__instant", "raw_entropy", "raw_remote_history_excl16_minus_source"),
                **{name: ("onset__instant", "raw_entropy", "raw_remote_history_excl16_minus_source") for name in onset_views}}
    result = evaluate_records(records, predictions, args.annotations, primary=primary, controls=controls,
                              bootstrap=args.bootstrap)
    result["scope"]["upstream_score_supervision"] = "natural training and calibration labels"
    result["notes"] = ["onset head was trained on all annotated span onsets, not answer-first errors",
                       "no separate continuation classifier exists in S10",
                       "first-error views here are post-hoc; do not choose heads or thresholds using them",
                       "historically reused official test is not a pristine confirmation set"]
    write_json(out / "evaluation.json", result)
    write_json(out / "complete.json", {"complete": True, "new_model_forwards": 0, "models_retrained": 0})
    for name, view in result["views"].items():
        for method in ("error__combined", "onset__combined", "raw_entropy", "raw_remote_history_excl16_minus_source"):
            m = view["metrics"][method]
            print(json.dumps(dict(view=name, method=method, **m), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "predictions", "annotations", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--bootstrap", type=int, default=200)
    audit(p.parse_args())


if __name__ == "__main__":
    main()
