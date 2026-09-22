"""Read optional annotations only after all label-free scores have been saved."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from state_audit.storage import read_arrays, read_json, write_json


def ranking(labels, scores):
    positive = int(labels.sum())
    both = 0 < positive < len(labels)
    return {
        "tokens": len(labels), "positives": positive,
        "auroc": float(roc_auc_score(labels, scores)) if both else None,
        "ap": float(average_precision_score(labels, scores)) if positive else None,
    }


def load_evaluation(output, annotations):
    settings = read_json(output / "settings.json")
    labels_by_id = read_json(annotations)
    labels, onsets, firsts, scores = [], [], [], []
    for index, response in enumerate(settings["responses"]):
        saved = read_arrays(output / "responses" / f"{index:04d}" / "scores.npz")
        annotation = labels_by_id[response["id"]]
        values = np.asarray(annotation["labels"], dtype=np.int64)
        if not np.array_equal(annotation["token_ids"], saved["token_id"]):
            raise ValueError(f"{response['id']}: annotation token IDs differ from scored tokens")
        if len(values) != len(saved["risk"]) or not np.isin(values, [0, 1]).all():
            raise ValueError(f"{response['id']}: labels must align one-to-one and be binary")
        onset = values.astype(bool) & ~np.r_[False, values[:-1].astype(bool)]
        first = np.zeros(len(values), dtype=bool)
        if values.any():
            first[np.flatnonzero(values)[0]] = True
        labels.append(values)
        onsets.append(onset)
        firsts.append(first)
        scores.append(np.column_stack([saved[name] for name in ("risk", "direct_risk")]))
    return tuple(np.concatenate(part) for part in (labels, onsets, firsts, scores))


def evaluate(output, annotations):
    labels, onset, first, scores = load_evaluation(output, annotations)
    normal = labels == 0
    continuation = labels.astype(bool) & ~onset
    groups = {
        "all_error": (np.ones(len(labels), dtype=bool), labels),
        "span_onset_vs_normal": (normal | onset, onset),
        "first_error_vs_normal": (normal | first, first),
        "continuation_vs_normal": (normal | continuation, continuation),
    }
    result = {"labels_used_for_scoring": False, "threshold_calibrated": False, "methods": {}}
    for column, method in enumerate(("support_graph", "direct_prompt")):
        result["methods"][method] = {
            name: ranking(target[mask], scores[mask, column])
            for name, (mask, target) in groups.items()
        }
    write_json(output / "evaluation.json", result)
    return result
