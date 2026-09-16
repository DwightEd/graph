"""Small, read-only helpers for the saved CHARM audit schema."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def load_predictions(root, completed_only=False):
    """Read only scores and metadata, never the large saved embedding member."""
    root = Path(root)
    settings = json.loads((root / "prediction_settings.json").read_text())
    files = sorted((root / "samples").glob("*.npz"))
    if not completed_only:
        names = json.loads((root / "predictions.json").read_text())
        files = [root / "samples" / Path(name).name for name in names]
    if not files:
        raise ValueError("No completed prediction files: " + str(root))
    samples = []
    required = {"gold", "onset", "spans", "offsets", "response", "score"}
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            sample = json.loads(str(saved["record_json"]))
            sample.update({key: saved[key] for key in saved.files
                           if key in required or key.startswith(("score_", "structure_"))})
        sample["response"] = str(sample["response"])
        sample["id"] = str(sample["id"])
        sample["source_id"] = str(sample["source_id"])
        sample["prediction_file"] = str(path)
        validate_sample(sample)
        samples.append(sample)
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate answer IDs would duplicate evaluation tokens")
    expected = set(map(str, settings["records"]))
    actual = {sample["id"] for sample in samples}
    if actual - expected:
        raise ValueError("Saved answer IDs are outside the requested prediction scope")
    if not completed_only and actual != expected:
        raise ValueError("Prediction manifest does not cover the requested records; use --completed-only for a labeled partial audit")
    return samples, settings


def merged_spans(sample):
    """Merge token-overlapping annotations, but preserve adjacent annotations."""
    result = []
    for start, end in sorted(map(tuple, sample["spans"])):
        start, end = int(start), int(end)
        if result and start < result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def membership(sample):
    result = np.full(len(sample["gold"]), -1, dtype=int)
    for index, (start, end) in enumerate(merged_spans(sample)):
        result[start:end] = index
    return result


def validate_sample(sample):
    """Scientific alignment checks belong here, not in every calculation."""
    count = len(sample["gold"])
    if not count or sample["offsets"].shape != (count, 2):
        raise ValueError("Empty or misaligned answer: " + sample["id"])
    if not np.isin(sample["gold"], [0, 1]).all():
        raise ValueError("Labels must be binary")
    for key in ["score", "onset"] + [k for k in sample if k.startswith("score_")]:
        if np.asarray(sample[key]).shape != (count,):
            raise ValueError("Misaligned " + key + ": " + sample["id"])
    if not np.isfinite(sample["score"]).all():
        raise ValueError("Original score must cover every saved token")
    for start, end in sample["spans"]:
        if not 0 <= start < end <= count:
            raise ValueError("Invalid half-open token span")
    gold = sample["gold"].astype(bool)
    if not np.array_equal(membership(sample) >= 0, gold):
        raise ValueError("Saved spans and token labels disagree")
    expected = np.zeros(count, bool)
    for start, _ in sample["spans"]:
        expected[int(start)] = True
    if not np.array_equal(expected, sample["onset"].astype(bool)):
        raise ValueError("Saved onset mask disagrees with annotation starts")


def roles(sample):
    gold = sample["gold"].astype(bool)
    first = np.zeros(len(gold), bool)
    before = np.zeros(len(gold), bool)
    if gold.any():
        position = np.flatnonzero(gold)[0]
        first[position] = True
        before[:position] = True
    return dict(first_error=first, later_onset=sample["onset"].astype(bool) & ~first,
                continuation=gold & ~sample["onset"].astype(bool),
                normal_clean=~gold & (not gold.any()), normal_before=~gold & before,
                normal_after=~gold & ~before & gold.any())


def metric(gold, score, threshold, weights=None):
    gold, score = np.asarray(gold, bool), np.asarray(score, float)
    finite = np.isfinite(score)
    gold, score = gold[finite], score[finite]
    weights = np.ones(len(gold)) if weights is None else np.asarray(weights)[finite]
    keep = weights > 0
    gold, score, weights = gold[keep], score[keep], weights[keep]
    alarm = score > threshold
    tp, fn, fp, tn = [float(weights[mask].sum()) for mask in
                       (gold & alarm, gold & ~alarm, ~gold & alarm, ~gold & ~alarm)]
    return dict(tokens=len(gold), positive_tokens=int(gold.sum()), negative_tokens=int((~gold).sum()),
                prevalence=fraction(tp + fn, tp + fn + fp + tn),
                auroc=float(roc_auc_score(gold, score, sample_weight=weights)) if (tp + fn) and (fp + tn) else None,
                ap=float(average_precision_score(gold, score, sample_weight=weights)) if tp + fn else None,
                tp=tp, fn=fn, fp=fp, tn=tn, recall=fraction(tp, tp + fn),
                fpr=fraction(fp, fp + tn), precision=fraction(tp, tp + fp),
                accuracy=fraction(tp + tn, tp + fn + fp + tn))


def win_credit(positive, negative):
    """Per-positive AUROC credit; exact ties count one half."""
    negative = np.sort(negative)
    if not len(negative):
        return np.full(len(positive), np.nan)
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return (lower + upper) / (2 * len(negative))


def intervals(mask):
    differences = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(differences == 1), np.flatnonzero(differences == -1)))


def finite_mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None
