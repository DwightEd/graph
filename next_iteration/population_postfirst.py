"""Frozen post-first analysis of completed observer measurements; CPU only."""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from decoding.ragtruth_population_io import verify_response
from sklearn.metrics import average_precision_score, roc_auc_score

from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import write_json_once

PROTOCOL = {"schema": "ragtruth-post-first-descriptive@1", "subsets": ["all_tokens", "through_first_error", "post_first_error"],
    "score_direction": "fixed higher vs error; no label-based flipping/normalization",
    "bootstrap": 0, "source_balanced": True, "labels_stage": "evaluation only; oracle boundaries not detector inputs",
    "scope": "coarse observer replay dependence, not ownership/routing/continuous-span detector"}


def masks(y):
    first = np.ones(len(y), dtype=bool)
    if y.any():
        first[np.flatnonzero(y)[0] + 1:] = False
    return {"all_tokens": np.ones(len(y), dtype=bool), "through_first_error": first, "post_first_error": ~first}


def fixed_scores(measured):
    return {"entropy": measured["base__entropy"].copy(), "negative_margin": -measured["base__margin"].copy(),
        "source_small_js": measured["source_01__js"].copy(), "history_small_js": measured["history_01__js"].copy(),
        "source_saved_support": -measured["source_01__saved_logp_change"].copy(),
        "history_saved_support": -measured["history_01__saved_logp_change"].copy(),
        "history_minus_source_support": measured["source_01__saved_logp_change"] - measured["history_01__saved_logp_change"],
        "source_permute_js": measured["source_permute__js"].copy(), "mlp_small_js": measured["mlp_01__js"].copy()}


def weighted_metrics(y, scores, inverse):
    if len(np.unique(y)) < 2:
        return None
    counts = np.bincount(inverse)
    weights = 1. / counts[inverse]
    return {"auroc": float(roc_auc_score(y, scores, sample_weight=weights)),
        "auprc": float(average_precision_score(y, scores, sample_weight=weights)), "bootstrap_replicates": 0}


def run(args):
    if args.output.exists():
        raise FileExistsError("fresh post-first analysis output required")
    run_settings = json.loads((args.run / "settings.json").read_text())
    completion = (args.run / "COMPLETE").read_text().strip()
    progress = json.loads((args.run / "progress.json").read_text())
    if completion != "completed=17790 failed=0 total=17790" or progress["status"] != "complete" or progress["failed"]:
        raise ValueError("requires completed frozen full population")
    labels_path = Path(run_settings["dataset"]) / "response.jsonl"
    input_manifest = json.loads((args.run / "input_manifest.json").read_text())
    if file_sha256(labels_path) != run_settings["input_sha256"]["response.jsonl"] or file_sha256(args.run / "inputs.jsonl") != input_manifest["sha256"]:
        raise ValueError("frozen input/annotation bytes differ")
    paths = [Path(__file__), Path(__file__).resolve().parents[1] / "refine-logs/POST_FIRST_ANALYSIS_PLAN_20260913.md"]
    import decoding.ragtruth_population_io as population_io

    paths.append(Path(population_io.__file__))
    parent_files = [args.run / n for n in ("settings.json", "input_manifest.json", "inputs.jsonl", "COMPLETE", "progress.json")]
    evaluation_files = sorted(args.run.glob("evaluation_*.json"))
    if len(evaluation_files) != 1:
        raise ValueError("requires the unique completed frozen population evaluation")
    parent_files.extend(evaluation_files)
    settings = {"protocol": PROTOCOL, "code_and_plan_sha256": {str(p.resolve()): file_sha256(p) for p in paths},
        "parent_sha256": {str(p.resolve()): file_sha256(p) for p in parent_files},
        "labels_path": str(labels_path), "labels_sha256": file_sha256(labels_path),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "model_forwards": 0}
    write_json_once(args.output / "settings.json", settings)
    for i, p in enumerate(paths):
        destination = args.output / "executed_code" / f"{i}_{p.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
        if file_sha256(destination) != settings["code_and_plan_sha256"][str(p.resolve())]:
            raise ValueError("analysis code changed during snapshot")
    # Code/plan are frozen before any annotation values are read.
    roster = [json.loads(line) for line in (args.run / "inputs.jsonl").open()]
    annotations = [json.loads(line) for line in labels_path.open()]
    previous_evaluation = json.loads(evaluation_files[0].read_text())
    if len(roster) != 17790 or len({str(r["id"]) for r in roster}) != len(roster) or len({str(r["id"]) for r in annotations}) != len(annotations):
        raise ValueError("roster coverage/uniqueness mismatch")
    labels = {str(r["id"]): r for r in annotations}
    if set(labels) != {str(r["id"]) for r in roster}:
        raise ValueError("full annotation and frozen response roster ID sets differ")
    groups, census, response_manifests = defaultdict(list), Counter(), {}
    for i, row in enumerate(roster):
        rid = str(row["id"])
        directory = args.run / "responses" / rid
        verify_response(directory, row)
        response_manifests[rid] = file_sha256(directory / "manifest.json")
        record = json.loads((directory / "record.json").read_text())
        annotation = labels[rid]
        if (str(annotation["source_id"]) != str(row["source_id"]) or annotation["model"] != row["generator"]
                or annotation["split"] != row["official_split"] or record["id"] != rid
                or hashlib.sha256(annotation["response"].encode()).hexdigest() != row["response_sha256"]):
            raise ValueError("actual annotation/measurement identity mismatch")
        with np.load(directory / "tokens.npz", allow_pickle=False) as token_file:
            offsets = token_file["offsets"].copy()
            length = len(token_file["token_ids"]) - row["prompt_length"]
        if offsets.shape != (length, 2) or np.any(offsets[:, 0] < 0) or np.any(offsets[:, 1] > len(row["response"])) or np.any(offsets[:, 0] >= offsets[:, 1]):
            raise ValueError("invalid original response offsets")
        y = np.zeros(length, dtype=bool)
        for span in annotation["labels"]:
            if not 0 <= span["start"] < span["end"] <= len(row["response"]):
                raise ValueError("invalid actual annotation span")
            y |= (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])
        with np.load(directory / "metrics.npz", allow_pickle=False) as measured:
            scores = fixed_scores(measured)
        if any(v.shape != y.shape or not np.isfinite(v).all() for v in scores.values()):
            raise ValueError("score shape/finiteness mismatch")
        split_masks = masks(y)
        starts = y & ~np.r_[False, y[:-1]]
        census["responses"] += 1
        census["tokens"] += len(y)
        census["error_tokens"] += int(y.sum())
        census["error_run_starts"] += int(starts.sum())
        census["error_run_interiors"] += int((y & ~starts).sum())
        for name, mask in split_masks.items():
            census[name + ":tokens"] += int(mask.sum())
            census[name + ":errors"] += int(y[mask].sum())
        key = (row["task"], row["generator"], row["official_split"])
        groups[key].append({"id": rid, "source_id": str(row["source_id"]), "y": y, "scores": scores, "masks": split_masks})
        if (i + 1) % 1000 == 0:
            print(json.dumps({"verified_responses": i + 1}), flush=True)
    results = []
    for key, members in sorted(groups.items()):
        y = np.concatenate([r["y"] for r in members])
        source = np.concatenate([np.full(len(r["y"]), r["source_id"]) for r in members])
        subsets = {}
        for name in PROTOCOL["subsets"]:
            mask = np.concatenate([r["masks"][name] for r in members])
            _, inverse = np.unique(source[mask], return_inverse=True)
            subset_scores = {}
            for score_name in members[0]["scores"]:
                values = np.concatenate([r["scores"][score_name] for r in members])[mask]
                subset_scores[score_name] = weighted_metrics(y[mask], values, inverse)
            subsets[name] = {"tokens": int(mask.sum()), "errors": int(y[mask].sum()),
                "sources": len(np.unique(inverse)), "metrics": subset_scores}
        results.append({"task": key[0], "generator": key[1], "official_split": key[2], "responses": len(members), "subsets": subsets})
    previous_groups = {(g["task"], g["generator"], g["official_split"]): g for g in previous_evaluation["groups"]}
    if (set(previous_groups) != set(groups) or previous_evaluation["verified_completed_responses"] != len(roster)
            or previous_evaluation["run_settings_sha256"] != file_sha256(args.run / "settings.json")):
        raise ValueError("prior evaluation differs from complete source population")
    common_scores = {"entropy": "entropy", "negative_margin": "negative_margin",
        "source_small_js": "source_small_sensitivity", "history_small_js": "history_small_sensitivity",
        "source_saved_support": "source_saved_token_support", "source_permute_js": "topology_sensitivity", "mlp_small_js": "mlp_sensitivity"}
    comparisons = 0
    for result in results:
        prior = previous_groups[(result["task"], result["generator"], result["official_split"])]
        if result["responses"] != prior["responses"]:
            raise ValueError("prior response group denominator changed")
        for name in ("all_tokens", "through_first_error"):
            current, old = result["subsets"][name], prior["subsets"][name]
            if current["tokens"] != old["tokens"] or current["errors"] != old["error_tokens"]:
                raise ValueError("prior first/all annotation masks differ")
            for new_score, old_score in common_scores.items():
                a, b = current["metrics"][new_score], old["metrics"][old_score]["descriptive_ranking"]
                if (a is None) != (b is None) or (a is not None and any(not np.isclose(a[k], b[k], atol=1e-12, rtol=1e-12) for k in ("auroc", "auprc"))):
                    raise ValueError("shared fixed ranking differs from frozen evaluation")
                comparisons += 1
    for p, sha in {**settings["code_and_plan_sha256"], **settings["parent_sha256"], str(labels_path): settings["labels_sha256"]}.items():
        if file_sha256(p) != sha:
            raise ValueError("analysis code/input changed during evaluation")
    write_json_once(args.output / "response_manifest_hashes.json", response_manifests)
    write_json_once(args.output / "results.json", {"status": "complete", "protocol": PROTOCOL, "census": dict(census),
        "legacy_shared_group_metric_pairs_verified": comparisons, "groups": results})
    write_json_once(args.output / "manifest.json", {"status": "complete", "settings_file_sha256": file_sha256(args.output / "settings.json"),
        "artifacts": {p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file() and p.name != "settings.json"}})
    print(json.dumps(dict(census)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
