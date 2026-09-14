"""Label-free binding projection, followed by a separate optional evaluation.

Input packets supply candidate matches and evidence relations; this program
DOES NOT extract semantic facts or invent bindings from attention scalars.
Run --demo to test the mathematics, not to claim RAGTruth detection performance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .projection import Factor, cosine_logits, permute_relations, project


def write(path, value):
    def safe(x):
        if isinstance(x, dict):
            return {k: safe(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [safe(v) for v in x]
        return None if isinstance(x, float) and not math.isfinite(x) else x
    Path(path).write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metadata(raw):
    rid = str(raw["id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
        raise ValueError("record ID must be safe as a filename")
    offsets = np.asarray(raw["offsets"])
    if (offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets)
            or not np.issubdtype(offsets.dtype, np.integer)
            or (offsets[:, 0] < 0).any() or (offsets[:, 1] < offsets[:, 0]).any()
            or (np.diff(offsets[:, 0]) < 0).any()):
        raise ValueError("full response-relative integer token offsets required")
    digest = raw.get("response_sha256")
    if "response" in raw:
        actual = hashlib.sha256(raw["response"].encode()).hexdigest()
        if digest is not None and digest != actual:
            raise ValueError("response checksum mismatch")
        digest = actual
        if offsets[:, 1].max() > len(raw["response"]):
            raise ValueError("offset beyond original response")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("response or response_sha256 required")
    split = raw["official_split"]
    if split not in ("train", "test", "calibration"):
        raise ValueError("unknown split")
    return dict(id=rid, source_id=str(raw["source_id"]), official_split=split,
                response_sha256=digest, offsets=offsets.tolist(), task=raw.get("task", "unknown"),
                generator=raw.get("generator", "unknown"))


def event_projection(event, directory, max_states, permutations, seed):
    domains, logits, groups = {}, {}, {}
    representations = None
    if "representations" in event:
        with np.load(directory / event["representations"], allow_pickle=False) as data:
            representations = {k: data[k].copy() for k in data.files}
        if not event.get("representation_space"):
            raise ValueError("native vectors require declared observer/layer/head/state space")
    for v in event["variables"]:
        name, candidates = v["id"], v["candidates"]
        if name in domains:
            raise ValueError("duplicate variable")
        domains[name] = candidates
        groups[name] = v.get("groups", ["all"] * len(candidates))
        if "log_weights" in v:
            logits[name] = v["log_weights"]
        elif representations is not None:
            logits[name] = cosine_logits(representations[v["query_array"]], representations[v["candidate_array"]],
                                         event.get("temperature", .1)).tolist()
        else:
            raise ValueError("each variable needs log_weights or aligned cached vectors")
    factors = [Factor(tuple(f["variables"]), frozenset(tuple(x) for x in f.get("allowed", [])),
                      frozenset(tuple(x) for x in f.get("forbidden", [])), f.get("closed_world", False),
                      f.get("provenance", "")) for f in event["factors"]]
    if any(type(f.closed_world) is not bool for f in factors):
        raise ValueError("closed_world must be boolean")
    joint = event.get("joint_log_weights")
    r = project(domains, logits, factors, joint_log_weights=joint, max_states=max_states)
    shuffled, moves = [], []
    for i in range(permutations):
        fs, movement = permute_relations(domains, factors, seed + i, groups)
        shuffled.append(project(domains, logits, fs, joint_log_weights=joint, max_states=max_states)["lower_nats"])
        moves.append(movement["moved_fraction"])
    r.update(binding_shuffled_lower_nats=float(np.mean(shuffled)) if shuffled else None,
             shuffled_lower_by_seed=shuffled, permuted_fraction_by_seed=moves,
             seed=seed, lower_infinite=math.isinf(r["lower_nats"]), upper_infinite=math.isinf(r["upper_nats"]))
    return r


def score_record(record, packet, directory, max_states, permutations, seed):
    n = len(record["offsets"])
    offsets = np.asarray(record["offsets"])
    # All original tokens remain. Missing bindings are NaN, NEVER a zero risk.
    keys = ("binding_exact_nats", "binding_lower_nats", "binding_upper_nats", "binding_shuffled_lower_nats",
            "binding_uniform_lower_nats", "node_uncertainty", "node_entropy", "no_constraints")
    scores = {k: np.full(n, np.nan) for k in keys}
    available_after = np.full(n, -1, dtype=np.int64)
    diagnostics = []
    for j, event in enumerate(packet.get("events", [])):
        start, end = event["target_span"]
        available = event["available_after_char"]
        if not 0 <= start < end <= available <= int(offsets[:, 1].max()):
            raise ValueError("event must declare valid target span and when its information is available")
        hit = (offsets[:, 0] < end) & (offsets[:, 1] > start) & (offsets[:, 1] > offsets[:, 0])
        if not hit.any():
            raise ValueError("event has no target token")
        if (available_after[hit] >= 0).any():
            raise ValueError("overlapping target events: supply one joint event, not implicit max/mean aggregation")
        available_after[hit] = available
        r = event_projection(event, directory, max_states, permutations, seed)
        scores["node_uncertainty"][hit] = 1 - r["node_confidence"]
        scores["node_entropy"][hit] = r["node_entropy"]
        scores["no_constraints"][hit] = 0.
        # Inconsistent user constraints are a failed model, not evidence of hallucination.
        usable = r["status"] not in ("no_constraints", "inconsistent_constraints")
        if usable:
            exact = r["lower_nats"] if r["status"] in ("closed_relations", "zero_possible_probability") else None
            values = (exact, r["lower_nats"], r["upper_nats"], r["binding_shuffled_lower_nats"],
                      r["uniform_lower_nats"], 1 - r["node_confidence"], r["node_entropy"], 0.)
            for key, value in zip(keys, values):
                if value is not None:
                    scores[key][hit] = value
        diagnostics.append(dict(event=j, target_span=[start, end], available_after_char=available,
                                score_available=usable and r["status"] in ("closed_relations", "zero_possible_probability"),
                                bounds_available=usable, **r))
    return scores, available_after, diagnostics


def demo_packets():
    # Both local marginals have identical entropy/confidence in the two cases.
    for rid, p in (("demo_supported", [.95, .05]), ("demo_cross_bound", [.05, .95])):
        yield dict(schema="binding-packet-v1", id=rid, source_id=rid, task="controlled_demo",
                   official_split="test", response="A has value", offsets=[[0, 1], [2, 5], [6, 11]],
                   events=[dict(target_span=[6, 11], available_after_char=11,
                                variables=[dict(id="entity", candidates=["a", "b"], log_weights=np.log([.95, .05]).tolist()),
                                           dict(id="value", candidates=["a", "b"], log_weights=np.log(p).tolist())],
                                factors=[dict(variables=["entity", "value"], allowed=[["a", "a"], ["b", "b"]],
                                              closed_world=True, provenance="constructed_demo_identity_relation")])])


def run(args):
    if args.max_states < 1 or args.permutations < 1:
        raise ValueError("positive max_states and permutations required")
    if args.demo and getattr(args, "s10_predictions", None):
        raise ValueError("demo cannot use natural baseline scores")
    if args.demo:
        packets, directory = list(demo_packets()), Path.cwd()
        if args.annotations or args.roster:
            raise ValueError("demo cannot be evaluated as natural RAGTruth")
    else:
        packets, directory = read_rows(args.cases), Path(args.cases).resolve().parent
    if any(p.get("schema") != "binding-packet-v1" for p in packets):
        raise ValueError("expected binding-packet-v1, not a scalar attention cache")
    by_id = {str(p["id"]): p for p in packets}
    if len(by_id) != len(packets):
        raise ValueError("duplicate packet ID")
    # A full label-free roster fixes the denominator, including answers without a packet.
    baseline = {}
    if getattr(args, "s10_predictions", None):
        if args.roster or args.split != "test" or not args.s10_features:
            raise ValueError("S10 comparison requires --split test and --s10-features; it supplies the full test roster")
        from structural_detector.audit import load_frozen
        raw_roster, baseline = load_frozen(args.s10_predictions, args.s10_features)
    else:
        if getattr(args, "s10_features", None):
            raise ValueError("--s10-features needs --s10-predictions")
        raw_roster = read_rows(args.roster) if args.roster else packets
    records = [metadata(r) for r in raw_roster if args.split == "all" or r["official_split"] == args.split]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("empty or duplicate roster")
    if not set(by_id) <= {str(r["id"]) for r in raw_roster}:
        raise ValueError("packet not present in fixed roster")
    for r in records:
        if r["id"] in by_id:
            p = metadata(by_id[r["id"]])
            for key in ("source_id", "official_split", "response_sha256", "offsets"):
                if p[key] != r[key]:
                    raise ValueError("packet/roster identity mismatch: " + key)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", dict(method="binding-projection-v1", labels_used_for_scoring=False,
                                      semantics="supplied relations, not automatically certified", seed=args.seed,
                                      max_states=args.max_states, permutations=args.permutations,
                                      timing="offline target attribution; available_after_char is the detection time",
                                      missing_packets="retained as unavailable", demo=args.demo))
    (out / "cases.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in packets), encoding="utf-8")
    predictions, hashes = {}, {}
    event_count, covered = 0, 0
    for i, r in enumerate(records):
        scores, available, diagnostics = score_record(r, by_id.get(r["id"], {}), directory,
                                                     args.max_states, args.permutations, args.seed)
        scores.update(baseline.get(r["id"], {}))
        predictions[r["id"]] = scores
        np.savez_compressed(out / (r["id"] + ".npz"), offsets=r["offsets"], available_after_char=available, **scores)
        hashes[r["id"]] = hashlib.sha256((out / (r["id"] + ".npz")).read_bytes()).hexdigest()
        write(out / (r["id"] + ".events.json"), diagnostics)
        event_count += len(diagnostics)
        covered += int((~np.isnan(scores["binding_exact_nats"])).sum())
        if i % 25 == 0 or i + 1 == len(records):
            print(json.dumps(dict(scored=i + 1, planned=len(records), events=event_count, covered_tokens=covered)), flush=True)
    write(out / "records.json", records)
    write(out / "prediction_freeze.json", dict(labels_used_for_scoring=False, artifacts=hashes,
                                              responses=len(records), events=event_count, covered_tokens=covered))
    if args.annotations:
        from .evaluation import evaluate_records
        result = evaluate_records(records, predictions, args.annotations, split=args.split,
                                  controls=("node_uncertainty", "node_entropy", "binding_shuffled_lower_nats", "no_constraints",
                                            "error__combined", "onset__combined", "raw_entropy"),
                                  bootstrap=args.bootstrap)
        write(out / "evaluation.json", result)
        print(json.dumps(result["views"]["all_error"]["metrics"], ensure_ascii=False), flush=True)
    write(out / "complete.json", dict(complete=True, demo=args.demo, labels_used_for_scoring=False,
                                      new_llm_forwards=0, trained_parameters=0, events=event_count,
                                      covered_tokens=covered, total_tokens=sum(len(r["offsets"]) for r in records)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", help="JSONL binding packets with matches and supplied relations")
    source.add_argument("--demo", action="store_true", help="two constructed cases; NOT a natural benchmark")
    p.add_argument("--roster", help="full label-free reanchor population inputs.jsonl; fixes all-token denominator")
    p.add_argument("--s10-predictions", help="frozen S10 run: include all official-test baseline predictions")
    p.add_argument("--s10-features", help="matching S10 feature export; validates offsets and response identity")
    p.add_argument("--annotations", help="RAGTruth response.jsonl; opened ONLY after predictions are saved")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "test", "calibration", "all"), default="all")
    p.add_argument("--max-states", type=int, default=200_000)
    p.add_argument("--permutations", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--bootstrap", type=int, default=200)
    run(p.parse_args())


if __name__ == "__main__":
    main()
