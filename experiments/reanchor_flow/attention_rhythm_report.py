"""Descriptive, head-resolved rhythm statistics and source-balanced summaries.

Peak convention is OUR diagnostic (strict local maximum + minimum prominence),
not an undocumented claim to reproduce the paper's unspecified peak detector.
FAI is retrospective. Missing future exposure is NaN, never zero influence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def peak_mask(values, *, absolute: float = 0.0, relative: float = 0.10):
    """Strict adjacent prominence: constant plateaus and endpoints are NOT peaks."""
    x = np.asarray(values, dtype=float)
    peaks = np.zeros(x.shape, dtype=bool)
    if x.shape[-1] < 3:
        return peaks
    finite = np.isfinite(x)
    low = np.where(finite, x, np.inf).min(-1, keepdims=True)
    high = np.where(finite, x, -np.inf).max(-1, keepdims=True)
    threshold = np.maximum(absolute, relative * np.maximum(high - low, 0))
    prominence = x[..., 1:-1] - np.maximum(x[..., :-2], x[..., 2:])
    peaks[..., 1:-1] = (np.isfinite(prominence) & (prominence > 0)
                        & (prominence >= threshold))
    return peaks


def bucket_switch(buckets):
    """Old local-majority -> nonlocal-majority rule, on the SAME source rows."""
    b = np.asarray(buckets)
    anchor, local = b[..., :3].sum(-1), b[..., 3]
    total = anchor + local
    fraction = np.divide(anchor, total, out=np.zeros_like(anchor), where=total > 0)
    flip = np.zeros(anchor.shape, dtype=bool)
    flip[..., 1:] = ((total[..., :-1] > 0) & (fraction[..., :-1] < .5)
                    & (fraction[..., 1:] >= .5)
                    & (anchor[..., 1:] > anchor[..., :-1])
                    & (local[..., 1:] < local[..., :-1]))
    return flip


def pair_alignment(waad_peaks, fai_peaks, eligible, lo=0, hi=1):
    """All local/global head pairs, not an average attention representation.

    Denominator is FAI peaks, as in paper Table 2. Null is the EXACT expectation
    for count-preserving uniform permutation over the same eligible positions.
    It is a descriptive occupancy null, not causal or a stationarity guarantee.
    No event count is forced. An empty FAI peak set returns NaN.
    """
    w, f = np.asarray(waad_peaks, bool), np.asarray(fai_peaks, bool)
    valid = np.asarray(eligible, bool)
    covered = np.zeros_like(w)
    for lag in range(lo, hi + 1):
        if lag == 0:
            covered |= w
        elif lag < w.shape[-1]:
            covered[:, lag:] |= w[:, :-lag]
    covered &= valid[None]
    f = f & valid[None]
    denominator = f.sum(-1)
    numerator = covered.astype(float) @ f.astype(float).T
    observed = np.divide(numerator, denominator[None],
                         out=np.full(numerator.shape, np.nan), where=denominator[None] > 0)
    occupancy = covered.sum(-1) / valid.sum() if valid.any() else np.full(len(w), np.nan)
    expected = np.broadcast_to(occupancy[:, None], observed.shape).copy()
    expected[:, denominator == 0] = np.nan
    return observed, expected, denominator


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan, float),
                     where=np.asarray(denominator) > 0)


def analyze_rhythm(trace: dict, *, waad_prominence: float = 0.5,
                   fai_relative_prominence: float = 0.10) -> dict:
    """Use paper response-row coordinates; prediction-label joins are separate."""
    waad = np.asarray(trace["waad"])[..., 1:]  # response rows P..N-1
    msg = np.asarray(trace["message_waad"])[..., 1:]
    fai = np.asarray(trace["fai"])[..., 1:]    # response source P..N-1
    wp = peak_mask(waad, absolute=waad_prominence, relative=0)
    fp = peak_mask(fai, relative=fai_relative_prominence)
    old_a = bucket_switch(trace["attention_buckets"])[..., 1:]
    old_m = bucket_switch(trace["message_buckets"])[..., 1:]
    # Ask if the bucket rule detects at q or q+/-1; not a factual missed-reanchor label.
    caught = old_m.copy()
    caught[..., 1:] |= old_m[..., :-1]
    caught[..., :-1] |= old_m[..., 1:]
    local, global_ = trace["local_heads"], trace["global_heads"]
    heads = int(np.prod(waad.shape[:2]))
    flat_w, flat_f = wp.reshape(heads, -1), fp.reshape(heads, -1)
    eligible = trace["fai_count"][1:] > 0
    # Peaks require valid neighbors, otherwise boundary truncation becomes a peak.
    eligible[:1] = False
    eligible[-1:] = False
    if len(eligible) > 2:
        eligible[1:-1] &= ((trace["fai_count"][1:-2] > 0)
                          & (trace["fai_count"][3:] > 0))
    observed, expected, counts = pair_alignment(flat_w[local], flat_f[global_], eligible)
    # A separately reported full-horizon result removes variable FAI exposure.
    full = eligible & trace["fai_full_horizon"][1:]
    full_observed, full_expected, _ = pair_alignment(flat_w[local], flat_f[global_], full)
    return {
        "waad_peaks": wp, "fai_peaks": fp,
        "bucket_attention_switch": old_a, "bucket_message_switch": old_m,
        "waad_peak_count": wp.sum(-1), "fai_peak_count": fp.sum(-1),
        "waad_peaks_missed_by_bucket": (wp & ~caught).sum(-1),
        "bucket_missed_fraction": _ratio((wp & ~caught).sum(-1), wp.sum(-1)),
        "attention_message_waad_mae": np.abs(waad - msg).mean(-1),
        "coupling_observed": observed, "coupling_uniform_null": expected,
        "coupling_lift": observed - expected, "coupling_anchor_count": counts,
        "coupling_full_horizon_lift": full_observed - full_expected,
        "coupling_eligible_positions": np.array(eligible.sum()),
        "coupling_full_horizon_positions": np.array(full.sum()),
    }


def matched_label_gap(values, labels, response_index):
    """Same sample AND log2 position bin H-minus-N. Labels do not select heads.

    values [...,predicted_token]; known labels 0/1. Equal nonempty mixed bins.
    This controls coarse position only, NOT token type or full confounding.
    """
    labels = np.asarray(labels)
    bins = np.floor(np.log2(np.asarray(response_index) + 1)).astype(int)
    gaps = []
    for b in np.unique(bins):
        clean, hall = (bins == b) & (labels == 0), (bins == b) & (labels == 1)
        if clean.any() and hall.any():
            gaps.append(values[..., hall].mean(-1) - values[..., clean].mean(-1))
    return np.mean(gaps, axis=0) if gaps else np.full(values.shape[:-1], np.nan)


def label_observations(trace, labels):
    """q -> q+1 labels; exclude final paper row, NOT the first predictor row."""
    labels = np.asarray(labels)
    if len(labels) != len(trace["row_position"]) - 1:
        raise ValueError("labels must cover every captured predicted response token")
    rows = trace["row_position"][:-1] + 1 - int(trace["response_start"])
    return {name + "_matched_h_minus_n": matched_label_gap(
        np.asarray(trace[name])[..., :-1], labels, rows)
        for name in ("distance", "waad", "message_waad")}


def _finite_mean(x, axis=0):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    return _ratio(np.where(valid, x, 0).sum(axis), valid.sum(axis))


def source_bootstrap(entries, key: str, *, repetitions=500, seed=2026):
    """Equal samples within source, then equal sources. No one-sample fake CI."""
    sources = {}
    for item in entries:
        sources.setdefault(str(item["source_id"]), []).append(np.asarray(item[key], float))
    if not sources:
        return {"sources": 0, "mean": None, "ci95": None}
    data = np.stack([_finite_mean(np.stack(rows)) for rows in sources.values()])
    mean = _finite_mean(data)
    count = np.isfinite(data).sum(0)
    if len(data) < 3 or repetitions < 1:
        return {"sources": len(data), "valid_sources": count, "mean": mean, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = np.stack([_finite_mean(data[rng.integers(len(data), size=len(data))])
                      for _ in range(repetitions)])
    # Only compute quantiles for entries with enough real independent sources.
    lower, upper = np.full(mean.shape, np.nan), np.full(mean.shape, np.nan)
    for index in np.ndindex(mean.shape):
        values = draws[(slice(None), *index)]
        if count[index] >= 3 and np.isfinite(values).any():
            lower[index], upper[index] = np.quantile(values[np.isfinite(values)], [.025, .975])
    return {"sources": len(data), "valid_sources": count, "mean": mean,
            "ci95": np.stack((lower, upper))}


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def plot_sample(trace, audit, path, title=""):
    """Individual raw heads, same query/source coordinates; no pooled main view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "attention_maps" not in trace:
        return
    selected, rows = trace["map_heads"], trace["map_query_position"]
    heads, p = trace["waad"].shape[1], int(trace["response_start"])
    columns = len(selected)
    fig, axes = plt.subplots(4, columns, figsize=(5 * columns, 12), squeeze=False)
    for column, flat in enumerate(selected):
        l, h = divmod(int(flat), heads)
        raw = trace["attention_maps"][column]
        # This is an exact crop, not a renormalization or a sparse reconstruction.
        left, right = max(0, int(rows[0]) - 32), min(raw.shape[-1], int(rows[-1]) + 1)
        image = axes[0, column].imshow(raw[:, left:right], aspect="auto", origin="upper",
                                      extent=(left - .5, right - .5, rows[-1] + .5, rows[0] - .5))
        fig.colorbar(image, ax=axes[0, column], fraction=.04)
        axes[0, column].set_title(f"L{l} H{h}; mean distance={trace['distance_mean'][l,h]:.2f}")
        axes[0, column].set_xlabel("absolute source position (exact crop)")
        axes[0, column].set_ylabel("query position")
        slot = rows - (p - 1)
        axes[1, column].plot(rows, trace["waad"][l, h, slot], label="attention WAAD")
        axes[1, column].plot(rows, trace["message_waad"][l, h, slot], label="message-weighted WAAD")
        valid = rows >= p
        peak = audit["waad_peaks"][l, h, rows[valid] - p]
        axes[1, column].scatter(rows[valid][peak], trace["waad"][l, h, slot[valid]][peak], marker="x")
        axes[1, column].set_ylabel("clipped lookback distance")
        axes[1, column].legend(fontsize=8)
        bucket = trace["attention_buckets"][l, h, slot]
        for k, name in enumerate(("evidence", "other prompt", "older response", "recent response")):
            axes[2, column].plot(rows, bucket[:, k], label=name)
        axes[2, column].set_ylabel("attention mass; old bucket view")
        axes[2, column].legend(fontsize=7, ncol=2)
        axes[3, column].plot(rows, trace["fai"][l, h, slot], label="FAI (offline)")
        axes[3, column].set_ylabel("future incoming attention")
        axes[3, column].set_xlabel("absolute source position; NOT predictor label")
        axes[3, column].legend(fontsize=8)
    if "token_text" in trace:
        for axis in axes[0]:
            xmin, xmax = axis.get_xlim()
            positions = np.linspace(max(0, int(xmin + .5)), min(len(trace["token_text"])-1, int(xmax-.5)), 8, dtype=int)
            axis.set_xticks(positions)
            axis.set_xticklabels([f"{i}: {str(trace['token_text'][i]).strip()[:10]}" for i in positions],
                                 rotation=55, ha="right", fontsize=7)
    fig.suptitle(title + "\nIndividual-head observations; peaks are candidates, not factual mechanisms.")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    # Full source panel preserves prompt stripes hidden by the near-diagonal crop.
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 4), squeeze=False)
    for column, flat in enumerate(selected):
        l, h = divmod(int(flat), heads)
        axes[0, column].imshow(trace["attention_maps"][column], aspect="auto", origin="upper",
                              extent=(-.5, len(trace["token_ids"]) - .5, rows[-1] + .5, rows[0] - .5))
        axes[0, column].axvline(p - .5, linestyle="--")
        axes[0, column].set_title(f"L{l} H{h}: all source positions")
        axes[0, column].set_xlabel("source; dashed line = response start")
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".full_sources.png"), dpi=130)
    plt.close(fig)
    if "paper_group_maps" in trace:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for k, name in enumerate(("local", "global")):
            axes[k].imshow(trace["paper_group_maps"][k], aspect="auto", origin="upper")
            axes[k].set_title(name + " group MEAN: paper-reference panel ONLY")
        fig.tight_layout()
        fig.savefig(Path(path).with_suffix(".paper_reference.png"), dpi=130)
        plt.close(fig)


def event_examples(trace, audit):
    """Text inspection for displayed heads/window; never selects statistical rows."""
    if "map_heads" not in trace:
        return []
    p, heads = int(trace["response_start"]), trace["waad"].shape[1]
    text = trace.get("token_text", np.asarray([str(t) for t in trace["token_ids"]]))
    shown = set(map(int, trace["map_query_position"]))
    examples = []
    for flat in trace["map_heads"]:
        l, h = divmod(int(flat), heads)
        for relative in np.flatnonzero(audit["waad_peaks"][l, h]):
            q = p + int(relative)
            if q not in shown:
                continue
            slot = q - (p - 1)
            source = int(trace["winner_position"][l, h, slot])
            examples.append({"layer": l, "head": h, "query_position": q,
                             "query_token": str(text[q]),
                             "prediction_position": q+1 if q+1 < len(text) else None,
                             "predicted_token": str(text[q+1]) if q+1 < len(text) else None,
                             "strongest_attention_source": source, "source_token": str(text[source]),
                             "source_lag": q-source, "waad": float(trace["waad"][l,h,slot]),
                             "bucket_switch": bool(audit["bucket_message_switch"][l,h,relative]),
                             "status": "observational peak, not confirmed re-anchor"})
    return examples
