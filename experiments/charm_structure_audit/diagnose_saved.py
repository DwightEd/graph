"""Read saved CHARM scores: no model, graph reconstruction or threshold fitting.

Separate answer-first errors, later onsets, continuation and normal contexts.
Frozen controls are compared only on the SAME finite token coordinates.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


POSITIVE_ROLES = ("first_error", "later_onset", "continuation", "nonfirst", "all_error")
NEGATIVE_ROLES = ("normal", "normal_clean_answer", "normal_before_first", "normal_after_first")


def read_predictions(root, completed_only=False):
    """Read the existing audit schema, without materializing saved embeddings."""
    root = Path(root)
    settings = json.loads((root / "prediction_settings.json").read_text())
    if completed_only:
        files = sorted((root / "samples").glob("*.npz"))
    else:
        recorded = json.loads((root / "predictions.json").read_text())
        files = [root / "samples" / Path(path).name for path in recorded]
    if not files:
        raise ValueError("no finalized prediction samples; .partial files are not completed")
    samples = []
    fields = {"score", "gold", "onset", "spans", "offsets", "response"}
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            row = json.loads(str(saved["record_json"]))
            sample = {key: saved[key] for key in saved.files
                      if key in fields or key.startswith(("score_", "structure_"))}
            sample.update(id=str(row["id"]), source_id=str(row["source_id"]),
                          task=row["task"], generator=row["generator"],
                          diagnostic=json.loads(str(saved["diagnostic_json"])))
        sample["response"] = str(sample["response"])
        samples.append(sample)
    validate_predictions(samples)
    return samples, settings


def validate_predictions(samples):
    """Check alignment once at the read boundary; never pad missing predictions."""
    identities = [sample["id"] for sample in samples]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate response IDs in prediction files")
    for sample in samples:
        count = len(sample["gold"])
        names = [key for key in sample if key.startswith("score_")]
        for key in ("score", "gold", "onset", *names):
            if sample[key].shape != (count,):
                raise ValueError(f"{sample['id']}: {key} is not aligned to response tokens")
        if not np.isfinite(sample["score"]).all():
            raise ValueError("baseline scores must cover all saved response tokens")
        if np.any(sample["onset"].astype(bool) & ~sample["gold"].astype(bool)):
            raise ValueError("an onset lies outside the saved error labels")


def token_roles(sample):
    """Gold partitions for POST-HOC diagnostics, not detector inputs."""
    error = sample["gold"].astype(bool)
    onset = sample["onset"].astype(bool)
    first = np.zeros(len(error), bool)
    before = np.zeros(len(error), bool)
    after = np.zeros(len(error), bool)
    if error.any():
        position = np.flatnonzero(error)[0]
        first[position] = True
        before[:position] = True
        after[position + 1:] = True
    return dict(first_error=first, later_onset=onset & ~first,
                continuation=error & ~onset, nonfirst=error & ~first, all_error=error,
                normal=~error, normal_clean_answer=~error & (not error.any()),
                normal_before_first=~error & before, normal_after_first=~error & after)


def ranking(labels, scores, threshold):
    """AUROC/AP plus counts at the ORIGINAL saved threshold."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    predicted = scores > threshold
    positive = int(labels.sum())
    negative = int((~labels).sum())
    hits = int((labels & predicted).sum())
    false_alarms = int((~labels & predicted).sum())
    return dict(tokens=len(labels), positives=positive, negatives=negative,
                auroc=float(roc_auc_score(labels, scores)) if positive and negative else None,
                ap=float(average_precision_score(labels, scores)) if positive else None,
                true_positives=hits, false_positives=false_alarms,
                recall=hits / positive if positive else None,
                fpr=false_alarms / negative if negative else None)


def flatten(samples):
    roles = [token_roles(sample) for sample in samples]
    masks = {name: np.concatenate([role[name] for role in roles]) for name in roles[0]}
    scores = np.concatenate([sample["score"] for sample in samples])
    return scores, masks


def context_rankings(samples, threshold):
    """Use explicit negative contexts; cross-answer comparisons remain allowed."""
    scores, masks = flatten(samples)
    result = {}
    for positive in POSITIVE_ROLES:
        result[positive] = {}
        for negative in NEGATIVE_ROLES:
            selected = masks[positive] | masks[negative]
            result[positive][negative] = ranking(masks[positive][selected], scores[selected], threshold)
    return result


def answer_rankings(samples, threshold):
    """Separate within-answer discrimination from cross-answer ranking pairs."""
    rows = []
    for sample in samples:
        measured = ranking(sample["gold"], sample["score"], threshold)
        roles = token_roles(sample)
        first = sample["score"][roles["first_error"]]
        row = dict(id=sample["id"], source_id=sample["source_id"], **measured,
                   mean_score=float(sample["score"].mean()),
                   first_error_score=float(first[0]) if len(first) else None)
        rows.append(row)
    mixed = [row for row in rows if row["auroc"] is not None]
    within_pairs = sum(row["positives"] * row["negatives"] for row in mixed)
    within_wins = sum(row["auroc"] * row["positives"] * row["negatives"] for row in mixed)
    scores, masks = flatten(samples)
    pooled = ranking(masks["all_error"], scores, threshold)
    all_pairs = pooled["positives"] * pooled["negatives"]
    cross_pairs = all_pairs - within_pairs
    cross_wins = pooled["auroc"] * all_pairs - within_wins if all_pairs else 0.
    summary = dict(pooled=pooled, mixed_answers=len(mixed), answers=len(rows),
                   all_normal_answers=sum(row["positives"] == 0 for row in rows),
                   all_error_answers=sum(row["negatives"] == 0 for row in rows),
                   macro_within_answer_auroc=float(np.mean([row["auroc"] for row in mixed])) if mixed else None,
                   macro_within_answer_ap=float(np.mean([row["ap"] for row in mixed])) if mixed else None,
                   within_answer_pairs=within_pairs, cross_answer_pairs=cross_pairs,
                   pair_weighted_within_auroc=within_wins / within_pairs if within_pairs else None,
                   cross_answer_auroc=cross_wins / cross_pairs if cross_pairs else None,
                   note="Pair decomposition, not causal attribution. Many cross-answer pairs are inevitable; AP is not decomposed.")
    return summary, rows


def score_change(base, control, threshold):
    """Signed changes separate systematic shifts from absolute sensitivity."""
    if not len(base):
        return dict(tokens=0, mean_delta=None, mean_absolute_delta=None, mean_logit_delta=None,
                    lost_alarms=0, gained_alarms=0)
    clipped_base = np.clip(base, 1e-6, 1 - 1e-6)
    clipped_control = np.clip(control, 1e-6, 1 - 1e-6)
    logit_delta = np.log(clipped_control / (1 - clipped_control)) - np.log(clipped_base / (1 - clipped_base))
    return dict(tokens=len(base), mean_delta=float(np.mean(control - base)),
                mean_absolute_delta=float(np.mean(abs(control - base))),
                mean_logit_delta=float(np.mean(logit_delta)),
                lost_alarms=int(((base > threshold) & (control <= threshold)).sum()),
                gained_alarms=int(((base <= threshold) & (control > threshold)).sum()))


def compare_control(samples, key, threshold):
    """Every baseline/control metric uses exactly the same finite observations."""
    available = [sample for sample in samples if key in sample]
    base, masks = flatten(available)
    control = np.concatenate([sample[key] for sample in available])
    common = np.isfinite(base) & np.isfinite(control)
    result = dict(responses=len(available), common_tokens=int(common.sum()),
                  available_tokens=len(base), rankings={}, changes={})
    for positive in POSITIVE_ROLES:
        selected = common & (masks[positive] | masks["normal"])
        baseline = ranking(masks[positive][selected], base[selected], threshold)
        altered = ranking(masks[positive][selected], control[selected], threshold)
        delta = {metric: altered[metric] - baseline[metric]
                 if baseline[metric] is not None and altered[metric] is not None else None
                 for metric in ("auroc", "ap")}
        result["rankings"][positive] = dict(baseline=baseline, control=altered, delta=delta)
    for role, mask in masks.items():
        selected = common & mask
        result["changes"][role] = score_change(base[selected], control[selected], threshold)
    return result


def rewiring_summary(samples):
    """Changed RR endpoints divided by RR edges, not diluted by unchanged RP."""
    rows = []
    for sample in samples:
        if "rewire_in" not in sample["diagnostic"]:
            continue
        change = sample["diagnostic"]["rewire_in"]
        rr_edges = int(sample["structure_in_rr"].sum())
        rows.append(dict(id=sample["id"], changed_edges=change["changed_edges"],
                         all_edges=change["edges"], rr_edges=rr_edges))
    changed = sum(row["changed_edges"] for row in rows)
    all_edges = sum(row["all_edges"] for row in rows)
    rr_edges = sum(row["rr_edges"] for row in rows)
    return dict(responses=len(rows), changed_edges=changed, all_edges=all_edges, rr_edges=rr_edges,
                changed_fraction_all=changed / all_edges if all_edges else None,
                changed_fraction_rr=changed / rr_edges if rr_edges else None,
                per_response=rows,
                note="Original divisor, target indegree, RP edges and edge vectors are held fixed; only saved rewiring seeds are inspected.")


def boundary_changes(samples, threshold):
    """Adjacent score changes at N/H boundaries; no smoothing or span expansion."""
    collected = {name: [] for name in ("NN", "NH", "HH", "HN")}
    for sample in samples:
        labels = sample["gold"].astype(bool)
        scores = sample["score"]
        for name, left, right in (("NN", False, False), ("NH", False, True),
                                  ("HH", True, True), ("HN", True, False)):
            selected = (labels[:-1] == left) & (labels[1:] == right)
            pairs = np.stack((scores[:-1][selected], scores[1:][selected]), axis=1)
            collected[name].extend(pairs.tolist())
    result = {}
    for name, pairs in collected.items():
        values = np.asarray(pairs).reshape(-1, 2)
        result[name] = score_change(values[:, 0], values[:, 1], threshold)
    return result


def onset_rows(samples, threshold, controls):
    """Export every unique onset, not only the examples supporting a hypothesis."""
    rows = []
    for sample in samples:
        roles = token_roles(sample)
        for position in np.flatnonzero(sample["onset"]):
            start, end = sample["offsets"][position]
            row = dict(id=sample["id"], source_id=sample["source_id"], token=int(position),
                       role="first_error" if roles["first_error"][position] else "later_onset",
                       token_text=sample["response"][start:end],
                       context=sample["response"][max(0, start - 40):end + 80],
                       score=float(sample["score"][position]),
                       hit=bool(sample["score"][position] > threshold))
            for key in controls:
                value = sample[key][position] if key in sample else np.nan
                row[key] = float(value) if np.isfinite(value) else None
                row[key + "_delta"] = float(value - row["score"]) if np.isfinite(value) else None
            rows.append(row)
    return rows


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def diagnose(samples, settings, output, completed_only=False):
    """Write separate reports, preserving model, predictions and calibration."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    threshold = float(settings["threshold"]["value"])
    controls = sorted({key for sample in samples for key in sample if key.startswith("score_")})
    within, answers = answer_rankings(samples, threshold)
    result = dict(threshold=settings["threshold"], completed_only=completed_only,
                  responses=len(samples), input_records=[sample["id"] for sample in samples],
                  raw_annotation_spans=sum(len(sample["spans"]) for sample in samples),
                  unique_onset_tokens=sum(int(sample["onset"].sum()) for sample in samples),
                  context_rankings=context_rankings(samples, threshold),
                  answer_rankings=within,
                  controls={key: compare_control(samples, key, threshold) for key in controls},
                  rewiring=rewiring_summary(samples),
                  adjacent_transitions=boundary_changes(samples, threshold),
                  notes=["All masks use gold labels only for post-hoc attribution; no new detector is fitted.",
                         "Frozen perturbations are sensitivity tests, not retrained ablation or LLM interventions.",
                         "Prefix sites include gold onsets and a uniform grid; their metrics are subset diagnostics.",
                         "No test-fitted threshold, score normalization, label expansion or missing-score padding.",
                         "Within-answer AP and pooled AP have different prevalence; compare alongside counts."])
    (output / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(output / "answers.csv", answers)
    write_csv(output / "onsets.csv", onset_rows(samples, threshold, controls))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", help="default: <predictions>/saved_diagnostics")
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.predictions)
    output = Path(args.output) if args.output else root / "saved_diagnostics"
    if output.resolve() in (root.resolve(), (root / "samples").resolve()):
        raise ValueError("write diagnostics to a separate directory, not the original result directory")
    samples, settings = read_predictions(root, args.completed_only)
    result = diagnose(samples, settings, output, args.completed_only)
    print(json.dumps(result["answer_rankings"], ensure_ascii=False))
    for role in ("first_error", "later_onset", "continuation"):
        print(json.dumps(dict(role=role, **result["context_rankings"][role]), ensure_ascii=False))
    print("Saved diagnostics:", output)


if __name__ == "__main__":
    main()
