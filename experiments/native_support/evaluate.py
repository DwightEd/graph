"""Read optional annotations only after all label-free scores have been saved."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from state_audit.storage import read_arrays, read_json, write_json


def ranking(labels, scores, weights=None):
    positive = int(labels.sum())
    both = 0 < positive < len(labels)
    return {
        "tokens": len(labels), "positives": positive, "negatives": len(labels) - positive,
        "prevalence": float(positive / len(labels)) if len(labels) else None,
        "weighted_prevalence": float(np.average(labels, weights=weights)) if len(labels) else None,
        "auroc_status": "available" if both else "requires_both_classes",
        "auroc": float(roc_auc_score(labels, scores, sample_weight=weights)) if both else None,
        "ap": float(average_precision_score(labels, scores, sample_weight=weights)) if positive else None,
    }


def evaluation_records(output, annotations, score_root, columns):
    settings = read_json(output / "settings.json")
    labels_by_id = read_json(annotations)
    records = []
    for index, response in enumerate(settings["responses"]):
        saved = read_arrays(score_root / "responses" / f"{index:04d}" / "scores.npz")
        annotation = labels_by_id[response["id"]]
        if not np.array_equal(annotation["token_ids"], saved["token_id"]):
            raise ValueError(f"{response['id']}: annotation token IDs differ from scored tokens")
        if "source_id" in annotation and annotation["source_id"] != response["source_id"]:
            raise ValueError(f"{response['id']}: annotation source ID differs")
        values, onset, first, valid = annotation_targets(annotation, len(saved["token_id"]), response["id"])
        records.append({
            "id": response["id"], "source_id": response["source_id"],
            "labels": values[valid], "onsets": onset[valid], "firsts": first[valid],
            "target": np.flatnonzero(valid), "response_length": len(valid),
            "scores": np.column_stack([saved[name][valid] for name in columns]),
        })
    return records


def load_evaluation(output, annotations):
    records = evaluation_records(output, annotations, output, ("risk", "direct_risk"))
    return tuple(np.concatenate([r[name] for r in records]) for name in ("labels", "onsets", "firsts", "scores"))


def annotation_targets(annotation, count, identity):
    values = np.asarray(annotation["labels"], dtype=np.int64)
    if values.shape != (count,) or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{identity}: labels must align one-to-one and be binary")
    onset = values.astype(bool) & ~np.r_[False, values[:-1].astype(bool)]
    onset = np.asarray(annotation.get("span_onsets", onset), dtype=bool)
    valid = np.asarray(annotation.get("valid_tokens", np.ones(count)), dtype=bool)
    if onset.shape != values.shape or valid.shape != values.shape or (onset & ~values.astype(bool)).any():
        raise ValueError(f"{identity}: invalid onset/coverage alignment")
    first = np.zeros(count, dtype=bool)
    if (values.astype(bool) & valid).any():
        first[np.flatnonzero(values.astype(bool) & valid)[0]] = True
    return values, onset, first, valid


def unavailable_evaluation(output, annotations):
    result = {
        "status": "unavailable", "reason": "missing_token_annotations",
        "annotations_path": str(annotations), "auroc": None, "ap": None,
        "message": "No real token annotations were found. The four default resampled prefixes "
        "have no inherited official labels. Omit --annotations to use output/annotations.json "
        "if already prepared. Otherwise use --dataset to prepare a small official-answer "
        "pilot; annotations.json is then generated automatically. No scores were changed.",
    }
    write_json(output / "evaluation_status.json", result)
    return result


def evaluate(output, annotations=None):
    annotations = output / "annotations.json" if annotations is None else annotations
    if not annotations.is_file():
        return unavailable_evaluation(output, annotations)
    labels, onset, first, scores = load_evaluation(output, annotations)
    normal = labels == 0
    continuation = labels.astype(bool) & ~onset
    groups = {
        "all_error": (np.ones(len(labels), dtype=bool), labels),
        "span_onset_vs_normal": (normal | onset, onset),
        "first_error_vs_normal": (normal | first, first),
        "continuation_vs_normal": (normal | continuation, continuation),
    }
    settings = read_json(output / "settings.json")
    result = {
        "status": "evaluated", "annotations_path": str(annotations),
        "labels_used_for_scoring": False, "threshold_calibrated": False, "methods": {},
        "cohort": settings.get("cohort", {"selection": "supplied_responses"}),
    }
    for column, method in enumerate(("support_graph", "direct_prompt")):
        result["methods"][method] = {
            name: ranking(target[mask], scores[mask, column])
            for name, (mask, target) in groups.items()
        }
    write_json(output / "evaluation.json", result)
    write_json(output / "evaluation_status.json", {"status": "evaluated", "file": "evaluation.json"})
    return result
