"""Official-label evaluation only after every requested prediction is frozen."""

import json
from pathlib import Path

import numpy as np

from ..evaluate import ranking
from .run import read_jsonl, save_json

ORIGINAL_METHODS = ("direct_sentence", "atomic_raw", "atomic_flat")
METHODS = (*ORIGINAL_METHODS, "direct_isolated", "atomic_isolated", "blind_reconstruction")


def load_baselines(records):
    root = Path(__file__).resolve().parents[3] / "outputs/native_support_ragtruth_all"
    baseline_sets = {}
    for version in ("source_first_v1", "source_refine_v2"):
        directory = root / version
        manifest = json.loads((directory / "manifest.json").read_text())
        choices = json.loads((directory / "selection.json").read_text())["choices"]
        lookup = {row["id"]: row for row in manifest["records"]}
        baseline_sets[version] = (directory, lookup, choices)
    for record in records:
        for version, (directory, lookup, choices) in baseline_sets.items():
            if record["id"] not in lookup:
                continue
            path = directory / lookup[record["id"]]["directory"] / "scores.npz"
            if not path.exists():
                continue
            with np.load(path) as saved:
                if not np.array_equal(saved["token_id"], record["token_ids"]):
                    raise ValueError(f"{record['id']}: baseline token identities differ")
                method = choices[record["task"]]["method"]
                record["scores"][version] = saved[method]
                if version == "source_first_v1":
                    for name in ("raw_route", "raw_route_offline_mean", "source_local_unit_mean"):
                        record["scores"][name] = saved[name]


def label_record(record, official):
    if official["response"] != record["text"] or str(official["source_id"]) != record["source_id"]:
        raise ValueError("Frozen text/source differs from official annotation")
    if official["labels"] is None:
        raise ValueError("Missing labels are not reviewed negatives")
    offsets = np.asarray(record["offsets"])
    labels = np.zeros(len(offsets), dtype=bool)
    spans = []
    for annotation in official["labels"]:
        mask = (offsets[:, 0] < annotation["end"]) & (offsets[:, 1] > annotation["start"])
        if not mask.any():
            raise ValueError("Official span has no original-token overlap")
        labels |= mask
        spans.append(dict(text=record["text"][annotation["start"]:annotation["end"]],
            start=annotation["start"], end=annotation["end"], tokens=np.flatnonzero(mask).tolist()))
    record["labels"] = labels
    record["valid"] = offsets[:, 1] > offsets[:, 0]
    record["spans"] = spans


def classification(labels, scores):
    alarm = scores > 0
    positive = labels.astype(bool)
    true_positive = int((alarm & positive).sum())
    false_positive = int((alarm & ~positive).sum())
    return dict(threshold=0, tp=true_positive, fp=false_positive,
        precision=true_positive / int(alarm.sum()) if alarm.any() else None,
        recall=true_positive / int(positive.sum()) if positive.any() else None,
        fpr=false_positive / int((~positive).sum()) if (~positive).any() else None,
        alarm_fraction=float(alarm.mean()))


def summarize(records, method):
    chosen = [row for row in records if method in row["scores"]]
    if not chosen:
        return dict(answers=0)
    labels = np.concatenate([row["labels"][row["valid"]] for row in chosen])
    scores = np.concatenate([row["scores"][method][row["valid"]] for row in chosen])
    result = dict(answers=len(chosen), sources=len({row["source_id"] for row in chosen}), **ranking(labels, scores))
    numerator, denominator = 0, 0
    for row in chosen:
        target = row["labels"][row["valid"]]
        metric = ranking(target, row["scores"][method][row["valid"]])
        pairs = int(target.sum()) * int((~target).sum())
        if pairs:
            numerator += metric["auroc"] * pairs
            denominator += pairs
    result["within_answer_auroc"] = numerator / denominator if denominator else None
    if method in METHODS:
        result["fixed_alarm"] = classification(labels, scores)
        fractions = [float((row["scores"][method][span["tokens"]] > 0).mean())
                     for row in chosen for span in row["spans"]]
        clean = [row for row in chosen if not row["labels"].any()]
        result["span_detection"] = dict(spans=len(fractions), any_hit=sum(value > 0 for value in fractions),
            half_coverage_hit=sum(value >= .5 for value in fractions))
        result["normal_answers"] = dict(count=len(clean), any_false_alarm=sum(
            bool((row["scores"][method][row["valid"]] > 0).any()) for row in clean))
    return result


def diagnostics(records):
    result = []
    for row in records:
        for span in row["spans"]:
            entry = dict(id=row["id"], source_id=row["source_id"], task=row["task"],
                         group=row["group"], **span, methods={})
            for method in METHODS:
                if method not in row["scores"]:
                    continue
                values = row["scores"][method][span["tokens"]]
                entry["methods"][method] = dict(mean=float(values.mean()), maximum=float(values.max()),
                                                alarm_coverage=float((values > 0).mean()))
            result.append(entry)
    return result


def evaluate(args):
    if args.group != "all":
        raise ValueError("Freeze all regression and holdout predictions before any official-label evaluation")
    manifest = json.loads((args.output / "manifest.json").read_text())
    records = [row for row in manifest["records"] if args.group == "all" or row["group"] == args.group]
    if args.ids:
        raise ValueError("Evaluation requires the entire frozen group, not selected successful IDs")
    for row in records:
        path = args.output / "responses" / row["id"] / "scores.npz"
        with np.load(path) as saved:
            if not np.array_equal(saved["token_id"], row["token_ids"]):
                raise ValueError("Scored token identity mismatch")
            row["scores"] = {method: saved[method] for method in ORIGINAL_METHODS}
            if any(not np.isfinite(value).all() for value in row["scores"].values()):
                raise ValueError("Nonfinite predictions cannot be evaluated")
        if (args.output / "isolation_protocol.json").exists():
            with np.load(path.with_name("isolated_scores.npz")) as saved:
                if not np.array_equal(saved["token_id"], row["token_ids"]):
                    raise ValueError("Isolated score token identity mismatch")
                for method in ("direct_isolated", "atomic_isolated"):
                    values = saved[method]
                    if values.shape != (len(row["token_ids"]),) or not np.isfinite(values).all():
                        raise ValueError("Incomplete isolated predictions")
                    row["scores"][method] = values
        if (args.output / "blind_protocol.json").exists():
            with np.load(path.with_name("blind_scores.npz")) as saved:
                values = saved["blind_reconstruction"]
                if (not np.array_equal(saved["token_id"], row["token_ids"])
                        or values.shape != (len(row["token_ids"]),) or not np.isfinite(values).all()):
                    raise ValueError("Incomplete blind reconstruction predictions")
                row["scores"]["blind_reconstruction"] = values
    wanted = {row["id"] for row in records}
    official = {str(row["id"]): row for row in read_jsonl(Path(manifest["dataset"]) / "response.jsonl")
                if str(row["id"]) in wanted}
    for row in records:
        label_record(row, official[row["id"]])
    load_baselines(records)
    methods = sorted({method for row in records for method in row["scores"]})
    result = dict(primary="atomic_raw", exploratory=True, methods={}, coverage={})
    for group in sorted({row["group"] for row in records}):
        selected = [row for row in records if row["group"] == group]
        result["methods"][group] = {method: summarize(selected, method) for method in methods}
        result["methods"][group + "_by_task"] = {task: {method: summarize(
            [row for row in selected if row["task"] == task], method) for method in methods}
            for task in ("QA", "Summary", "Data2txt")}
        matched = [row for row in selected if "source_refine_v2" in row["scores"]]
        result["methods"][group + "_refine_matched"] = {method: summarize(matched, method) for method in methods}
        audits = [json.loads((args.output / "responses" / row["id"] / "audit.json").read_text()) for row in selected]
        result["coverage"][group] = dict(answers=len(audits), tokens=sum(a["tokens"] for a in audits),
            covered_tokens=sum(a["covered_tokens"] for a in audits),
            claims=sum(len(a["claims"]) for a in audits), exact_evidence=sum(a["evidence_exact"] for a in audits),
            invalid_claims=sum(len(a["invalid_claims"]) for a in audits),
            no_valid_claim_answers=sum(not a["claims"] for a in audits),
            truncated_answers=sum(a["extraction"]["reached_limit"] for a in audits))
        if (args.output / "blind_protocol.json").exists():
            blind = [json.loads((args.output / "responses" / row["id"] / "blind_audit.json").read_text()) for row in selected]
            result["coverage"][group]["blind"] = dict(questions=sum(len(a["questions"]) for a in blind),
                invalid=sum(len(a["invalid"]) for a in blind), covered_tokens=sum(a["covered_tokens"] for a in blind),
                answer_parse_failures=sum(a["answers"] is None for a in blind),
                question_contains_quote=sum(a["question_contains_quote"] for a in blind),
                question_truncations=sum(a["question_raw"]["reached_limit"] for a in blind),
                answer_truncations=sum(a["answer_raw"]["reached_limit"] for a in blind))
    save_json(args.output / f"evaluation_{args.group}.json", result)
    save_json(args.output / f"diagnostics_{args.group}.json", diagnostics(records))
    compact = {group: {name: {key: value for key, value in scores.items()
                            if key in ("answers", "tokens", "auroc", "ap", "fixed_alarm")}
                      for name, scores in methods.items()}
               for group, methods in result["methods"].items() if "_by_task" not in group}
    print(json.dumps(compact, indent=2))
