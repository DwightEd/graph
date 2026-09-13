"""Past-only alarms using thresholds calibrated on held-out fit sources."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from route_graph.archive import FeatureArchive, digest
from route_graph.data import read_jsonl, text_digest, write_jsonl
from route_graph.detector import SourceReference, _stratum


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights)[order])
    index = min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(order) - 1)
    return float(np.asarray(values)[order[index]])


def calibrate(archive: Path, output: Path, quantile=0.95):
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie inside (0, 1)")
    if output.exists():
        raise FileExistsError(output)
    fit = [r for group in FeatureArchive(archive).responses() for r in group if r["split"] == "train"]
    by_source = defaultdict(list)
    for row in fit:
        by_source[row["source_id"]].append(row)
    output.mkdir(parents=True)
    samples, excluded = [], []
    for held, rows in tqdm(sorted(by_source.items()), desc="leave-one-source calibration"):
        reference = SourceReference(neighbors=3, per_source=8).fit([r for r in fit if r["source_id"] != held])
        unavailable = {_stratum(r) for r in rows if _stratum(r) not in reference.profiles
                       or len(reference.profiles[_stratum(r)]["source_ids"]) < 3}
        if unavailable:
            excluded.append(dict(source_id=held, tokens=len(rows), strata=sorted(unavailable)))
            continue
        for row in rows:
            scores = reference.score(row)
            scores.update(entropy=row["entropy"], negative_margin=row["negative_margin"], position=row["token_index"])
            samples.append(dict(source_id=held, response_id=row["response_id"], token_index=row["token_index"],
                                weight=1 / len(rows), scores=scores))
    if not samples:
        raise ValueError("no fit source has complete leave-one-source reference coverage")
    write_jsonl(output / "scores.jsonl", samples)
    thresholds = {name: weighted_quantile([r["scores"][name] for r in samples], [r["weight"] for r in samples], quantile)
                  for name in samples[0]["scores"]}
    result = dict(schema="route-graph/alarm-thresholds@1", labels_used=False,
                  archive_index_sha256=digest(archive / "index.json"), scores_sha256=digest(output / "scores.jsonl"),
                  quantile=quantile, threshold_comparison="strictly_greater", weighting="equal_total_weight_per_source",
                  calibration="leave-one-fit-source-out", sources=len({r["source_id"] for r in samples}),
                  tokens=len(samples), excluded_sources=excluded, thresholds=thresholds)
    (output / "thresholds.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def alarm_positions(values, threshold, cooldown=8):
    if cooldown < 1:
        raise ValueError("cooldown must be positive")
    alarms, last = [], -cooldown
    for t, value in enumerate(values):
        if value > threshold and t - last >= cooldown:
            alarms.append(t)
            last = t
    return alarms


def evaluate(scores: Path, labels: Path, calibration: Path, output: Path, lead=8, cooldown=8):
    frozen = json.loads(calibration.read_text())
    if frozen["schema"] != "route-graph/alarm-thresholds@1" or frozen["labels_used"] is not False:
        raise ValueError("invalid unlabeled calibration")
    annotations = {str(r["id"]): r for r in read_jsonl(labels)}
    grouped = defaultdict(list)
    for row in read_jsonl(scores):
        grouped[row["response_id"]].append(row)
    events = []
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r["token_index"])
        if [r["token_index"] for r in rows] != list(range(rows[0]["token_count"])):
            raise ValueError("alarm evaluation requires full responses")
        annotation = annotations[key]
        if any(r["response_sha256"] != text_digest(annotation["response"]) or r["source_id"] != str(annotation["source_id"]) for r in rows):
            raise ValueError("score/label identity mismatch")
        error = []
        for row in rows:
            left, right = row["char_span"]
            if not 0 <= left < right <= len(annotation["response"]):
                raise ValueError("invalid character interval")
            error.append(any(left < s["end"] and right > s["start"] for s in annotation["labels"]))
        first = next((t for t, label in enumerate(error) if label), None)
        stop = len(rows) if first is None else first + 1
        for name, threshold in frozen["thresholds"].items():
            values = [r["scores"][name] for r in rows[:stop]]
            if not np.isfinite(values).all():
                raise ValueError("nonfinite alarm score")
            alarms = alarm_positions(values, threshold, cooldown)
            hits = [t for t in alarms if first is not None and first - lead <= t <= first]
            false = [t for t in alarms if first is None or t < first - lead]
            events.append(dict(response_id=key, source_id=rows[0]["source_id"], method=name,
                               observed_tokens=stop, first_error=first, alarm_positions=alarms,
                               hit=bool(hits), lead=first - hits[0] if hits else None,
                               false_alarms=len(false), any_false_alarm=bool(false)))
    methods = {}
    for name in frozen["thresholds"]:
        selected = [r for r in events if r["method"] == name]
        positives = [r for r in selected if r["first_error"] is not None]
        hits = [r for r in positives if r["hit"]]
        total = sum(r["observed_tokens"] for r in selected)
        methods[name] = dict(first_errors=len(positives), hits=len(hits), recall=len(hits)/len(positives) if positives else None,
                             mean_lead=float(np.mean([r["lead"] for r in hits])) if hits else None,
                             alarms=sum(len(r["alarm_positions"]) for r in selected),
                             false_alarms=sum(r["false_alarms"] for r in selected),
                             alarms_per_100_tokens=100 * sum(len(r["alarm_positions"]) for r in selected)/total,
                             responses_with_false_alarm=sum(r["any_false_alarm"] for r in selected),
                             observed_tokens=total)
    report = dict(schema="route-graph/alarms@1", labels_used_stage="evaluation_only", scores_sha256=digest(scores),
                  calibration_sha256=digest(calibration), lead_window=lead, cooldown=cooldown,
                  scope="stop each response at first error; fit quantile is not an equal realized test alarm budget",
                  methods=methods, responses=events)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return {k:v for k,v in report.items() if k != "responses"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("calibrate")
    fit.add_argument("--archive", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    for name in ("scores", "labels", "calibration", "output"):
        evaluation.add_argument(f"--{name}", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = calibrate(**args) if command == "calibrate" else evaluate(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
