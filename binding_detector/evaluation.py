"""Post-freeze metrics: never import this module in a feature/matching stage."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def label_views(offsets, spans, text_length):
    """Gold scopes are evaluation masks, never detector inputs.

    Continuation excludes EVERY annotated span onset. First error means the
    first positive token in the entire answer, not the first token of each span.
    """
    offsets = np.asarray(offsets)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("integer [tokens,2] response-relative offsets required")
    if (np.any(offsets < 0) or np.any(offsets[:, 1] < offsets[:, 0])
            or np.any(offsets[:, 1] > text_length) or np.any(np.diff(offsets[:, 0]) < 0)):
        raise ValueError("invalid response offsets")
    error = np.zeros(len(offsets), bool)
    onset = error.copy()
    for span in spans:
        start, end = span["start"], span["end"]
        if not 0 <= start < end <= text_length:
            raise ValueError("invalid annotation span")
        hit = (offsets[:, 0] < end) & (offsets[:, 1] > start) & (offsets[:, 1] > offsets[:, 0])
        if not hit.any():
            raise ValueError("annotation not covered by tokenization")
        error |= hit
        onset[np.flatnonzero(hit)[0]] = True
    first = np.zeros(len(error), bool)
    until, after = np.ones(len(error), bool), first.copy()
    if error.any():
        t = np.flatnonzero(error)[0]
        first[t] = True
        until[t + 1:] = False
        after[t + 1:] = True
    all_tokens = np.ones(len(error), bool)
    return {"all_error": (error, all_tokens), "span_onset_full_stream": (onset, all_tokens),
            "span_onset_vs_normal": (onset, ~error | onset),
            "first_error_full_stream": (first, all_tokens),
            "first_error_until_first": (first, until),
            "continuation_vs_normal": (error & ~onset, ~onset),
            "strict_post_first": (error, after)}


def ranking(y, score, sources):
    """Pooled and equal-source metrics. +/-inf preserve their ranking, NaN missing."""
    y, score, sources = np.asarray(y, bool), np.asarray(score, float), np.asarray(sources)
    valid = ~np.isnan(score)
    result = dict(eligible_tokens=len(y), covered_tokens=int(valid.sum()),
                  eligible_positives=int(y.sum()), covered_positives=int(y[valid].sum()),
                  coverage=float(valid.mean()) if len(valid) else None,
                  positive_coverage=float(valid[y].mean()) if y.any() else None)
    y, score, sources = y[valid], score[valid], sources[valid]
    result["infinite_scores"] = int(np.isinf(score).sum())
    if not len(y):
        result.update(pooled=None, source_balanced=None)
        return result
    _, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    ranks = rankdata(score, method="average")
    def measure(w=None):
        return dict(prevalence=float(np.average(y, weights=w)),
                    auroc=float(roc_auc_score(y, ranks, sample_weight=w)) if y.any() and not y.all() else None,
                    ap=float(average_precision_score(y, ranks, sample_weight=w)) if y.any() and not y.all() else None)
    result.update(pooled=measure(), source_balanced=measure(weights))
    return result


def compare_blocks(blocks, primary, controls, bootstrap=200, seed=20260914):
    """Compare on pairwise common coverage; bootstrap whole sources, not tokens."""
    names = sorted({name for b in blocks for name in b["scores"]})
    yy = np.concatenate([b["y"] for b in blocks])
    source = np.concatenate([np.repeat(b["source_id"], len(b["y"])) for b in blocks])
    scores = {name: np.concatenate([b["scores"].get(name, np.full(len(b["y"]), np.nan)) for b in blocks]) for name in names}
    metrics = {name: ranking(yy, score, source) for name, score in scores.items()}
    comparisons = {}
    for control in controls:
        if primary not in scores or control not in scores or primary == control:
            continue
        common = ~np.isnan(scores[primary]) & ~np.isnan(scores[control])
        p, c = scores[primary][common], scores[control][common]
        y, s = yy[common], source[common]
        point = {"primary": ranking(y, p, s), "control": ranking(y, c, s)}
        point["delta"] = {}
        for weight in ("pooled", "source_balanced"):
            a, b = point["primary"][weight], point["control"][weight]
            point["delta"][weight] = {m: a[m] - b[m] if a and b and a[m] is not None and b[m] is not None else None
                                       for m in ("auroc", "ap")}
        sources = np.unique(s)
        clusters = [np.flatnonzero(s == sid) for sid in sources]
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap if len(sources) > 1 else 0):
            chosen = rng.integers(len(clusters), size=len(clusters))
            idx = np.concatenate([clusters[i] for i in chosen])
            # Each duplicated sampled source is a separate equal-weight cluster.
            replicated = np.concatenate([np.repeat(j, len(clusters[i])) for j, i in enumerate(chosen)])
            if y[idx].any() and not y[idx].all():
                a, b = ranking(y[idx], p[idx], replicated), ranking(y[idx], c[idx], replicated)
                draws.append([a[k][m] - b[k][m] for k in ("pooled", "source_balanced") for m in ("auroc", "ap")])
        point["bootstrap"] = dict(valid=len(draws), seed=seed, unit="source",
                                  order=["pooled_auroc", "pooled_ap", "source_balanced_auroc", "source_balanced_ap"],
                                  delta_ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if draws else None)
        comparisons[control] = point
    return dict(metrics=metrics, comparisons=comparisons)


def evaluate_records(records, predictions, annotations, *, split="test", primary="binding_exact_nats",
                     controls=("node_uncertainty", "binding_shuffled_lower_nats"), bootstrap=200):
    """All predictions must already exist on disk when this is called."""
    if bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    chosen = [r for r in records if split == "all" or r["official_split"] == split]
    if not chosen:
        raise ValueError("no records in requested evaluation split")
    ids = {str(r["id"]) for r in chosen}
    gold = {}
    with Path(annotations).open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            if str(raw["id"]) in ids:
                if str(raw["id"]) in gold:
                    raise ValueError("duplicate annotation id")
                gold[str(raw["id"])] = raw
    if set(gold) != ids:
        raise ValueError("missing annotation records")
    scoped = {}
    for r in chosen:
        rid = str(r["id"])
        g = gold[rid]
        if str(g["source_id"]) != str(r["source_id"]) or g["split"] != r["official_split"]:
            raise ValueError("source/split mismatch")
        digest = hashlib.sha256(g["response"].encode()).hexdigest()
        if r["response_sha256"] != digest:
            raise ValueError("response identity mismatch")
        pred = predictions[rid]
        for name, (y, mask) in label_views(r["offsets"], g["labels"], len(g["response"])).items():
            if any(np.shape(v) != np.shape(y) for v in pred.values()):
                raise ValueError("prediction/token length mismatch")
            scoped.setdefault(name, []).append(dict(source_id=str(r["source_id"]), y=y[mask],
                                                    scores={k: v[mask] for k, v in pred.items()}))
    results = {}
    for name, blocks in scoped.items():
        p = primary.get(name, primary["default"]) if isinstance(primary, dict) else primary
        c = controls.get(name, controls["default"]) if isinstance(controls, dict) else controls
        print(f"evaluate {name}: primary={p}, source bootstrap={bootstrap}", flush=True)
        results[name] = compare_blocks(blocks, p, c, bootstrap)
    return dict(scope=dict(responses=len(chosen), sources=len({r["source_id"] for r in chosen}),
                           official_split=split, evaluation_labels_used_for_scores=False,
                           interpretation="post-hoc scopes; not first-error inference inputs"), views=results)
