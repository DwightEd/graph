"""Label contrasts and complete entry/reuse contingency tables on physical heads.

Token labels describe predictions q->q+1. Carrier labels describe b itself.
The two coordinates are never silently exchanged. Every horizon is recomputed
from saved full history matrices; empty support is missing, never a negative.
"""
from __future__ import annotations

import re
import numpy as np

HORIZONS = ((1, 16), (17, 64), (65, 0), (1, 0))
STATES = ("no_entry_no_reuse", "entry_no_reuse", "no_entry_reuse", "entry_reuse")


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def mean(a, axis=0):
    a = np.asarray(a, float)
    valid = np.isfinite(a)
    return ratio(np.where(valid, a, 0).sum(axis), valid.sum(axis))


def token_classes(text):
    def kind(s):
        s = str(s).strip()
        if not s:
            return 0
        if re.search(r"\d", s):
            return 1
        return 2 if any(c.isalpha() for c in s) else 3
    return np.array([kind(t) for t in text])


def matched_positions(labels, classes, valid, max_gap=32, onset_radius=None):
    """Nearest same-class N controls without reuse, within the same response.

    Onset controls additionally require a complete, known, all-normal window.
    Unknown/special tokens break eligible windows instead of being relabeled N.
    """
    labels, valid = np.asarray(labels), np.asarray(valid, bool)
    h = np.flatnonzero((labels == 1) & valid)
    n = np.flatnonzero((labels == 0) & valid)
    if onset_radius is not None:
        r = onset_radius
        h = np.array([i for i in h if i > 0 and labels[i - 1] == 0 and valid[i - 1]
                      and r <= i < len(labels) - r and valid[i-r:i+r+1].all()], int)
        n = np.array([i for i in n if r <= i < len(labels) - r
                      and valid[i-r:i+r+1].all() and (labels[i-r:i+r+1] == 0).all()], int)
    available, pairs = set(n.tolist()), []
    for i in h:
        options = [j for j in available if abs(i - j) <= max_gap and classes[i] == classes[j]]
        if options:
            j = min(options, key=lambda j: (abs(i-j), j))
            available.remove(j)
            pairs.append((i, j))
    return np.asarray(pairs, int).reshape(-1, 2)


def bracket_positions(labels, classes, valid, max_gap=32, onset_radius=None):
    """Normal controls on both sides, interpolated to the exact H position.

    This cancels a linear position trend without fitting to labels. Controls
    may recur; uncertainty is computed over independent sources, never tokens.
    """
    h = np.flatnonzero((labels == 1) & valid)
    n = np.flatnonzero((labels == 0) & valid)
    if onset_radius is not None:
        r = onset_radius
        h = np.array([i for i in h if i>0 and labels[i-1]==0 and valid[i-1]
                      and r<=i<len(labels)-r and valid[i-r:i+r+1].all()],int)
        n = np.array([i for i in n if r<=i<len(labels)-r and valid[i-r:i+r+1].all()
                      and (labels[i-r:i+r+1]==0).all()],int)
    matches=[]
    for i in h:
        same=n[(classes[n]==classes[i]) & (np.abs(n-i)<=max_gap)]
        left,right=same[same<i],same[same>i]
        if len(left) and len(right):
            matches.append((i,int(left[-1]),int(right[0])))
    return np.asarray(matches,int).reshape(-1,3)


def control_coordinates(pairs):
    """Accept simple pairs for a reference calculation, triples for the audit."""
    h,left=pairs[:,0],pairs[:,1]
    right=pairs[:,2] if pairs.shape[1]==3 else left
    w=(right-h)/(right-left) if pairs.shape[1]==3 else np.ones(len(pairs))
    return h,left,right,w


def matched_difference(values,pairs):
    h,left,right,w=control_coordinates(pairs)
    return values[...,h] - w*values[...,left] - (1-w)*values[...,right]


def read_metrics(trace):
    """All arrays returned as [layer,head,predicted response token]."""
    ordinary = trace["ordinary_mass"]
    masses = trace["mass"]
    metrics = {"special_mass": masses[..., 0],
               "evidence_mass": masses[..., 1],
               "evidence_share": ratio(masses[..., 1], ordinary),
               "prompt_share": ratio(masses[..., 1:3].sum(-1), ordinary),
               "far_history_share": ratio(masses[..., 3], ordinary),
               "local_history_share": ratio(masses[..., 4], ordinary),
               "self_share": ratio(masses[..., 5], ordinary),
               "message_evidence_share": ratio(trace["message_mass"][..., 1], trace["message_ordinary_mass"]),
               "message_prompt_share": ratio(trace["message_mass"][..., 1:3].sum(-1), trace["message_ordinary_mass"])}
    for i, scale in enumerate(trace["distance_scales"]):
        suffix = f"clip{int(scale)}" if scale else "full"
        metrics["distance_" + suffix] = trace["distance"][..., i]
        metrics["message_distance_" + suffix] = trace["message_distance"][..., i]
    metrics['control_unfiltered_waad10'] = trace['distance_with_special'][..., 1]
    metrics['control_unfiltered_distance'] = trace['distance_with_special'][..., -1]
    for name in ("entropy_normalized", "entropy", "change_tv", "top1", "repeat_mass", "head_norm", "head_margin",
                 "target_literal_evidence_mass", "target_literal_history_mass"):
        metrics[name] = trace[name]
    metrics["topk_past_mass"] = trace["top_attention"][..., 0, :].sum(-1)
    units = trace["unit_mass"]
    if units.shape[-1]:
        p = ratio(units, units.sum(-1, keepdims=True))
        metrics["evidence_unit_entropy"] = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)
        metrics["evidence_unit_top1"] = np.where(np.isfinite(p).any(-1), np.where(np.isfinite(p), p, 0).max(-1), np.nan)
    else:
        metrics["evidence_unit_entropy"] = np.full(ordinary.shape, np.nan)
        metrics["evidence_unit_top1"] = np.full(ordinary.shape, np.nan)
    rows = trace["row_position"]
    ordinary_token = ~trace["special_mask"]
    available = np.cumsum(ordinary_token)[rows]
    prompt = np.sum(ordinary_token[:int(trace["response_start"])])
    evidence = np.cumsum(trace["evidence_mask"])[rows]
    metrics["prompt_excess_uniform"] = metrics["prompt_share"] - ratio(prompt, available)
    metrics["evidence_excess_uniform"] = metrics["evidence_share"] - ratio(evidence, available)
    for source in ("evidence", "far_history"):
        values = metrics[source + "_share"]
        metrics[source + "_gain"] = np.concatenate((np.full_like(values[..., :1], np.nan), np.diff(values)), axis=-1)
    # A structural shift control, independent of WAAD peaks. Threshold sweep is
    # reported separately; continuous gains above remain the primary quantities.
    local = metrics["local_history_share"] + metrics["self_share"]
    local_delta = np.concatenate((np.full_like(local[..., :1], np.nan), np.diff(local)), axis=-1)
    for threshold in (.05, .10, .20):
        gain = metrics["evidence_gain"]
        valid = np.isfinite(gain) & np.isfinite(local_delta)
        metrics[f"entry_rate_{threshold:.2f}"] = np.where(valid, (gain >= threshold) & (local_delta <= -threshold), np.nan)
    invalid_query = trace["special_mask"][rows]
    result = {}
    for name, value in metrics.items():
        value = np.array(value[..., :-1], dtype=np.float32, copy=True)
        value[..., invalid_query[:-1]] = np.nan
        if name.endswith("gain") or name.startswith("entry_rate") or name == "change_tv":
            previous_special = np.r_[False, invalid_query[:-2]]
            value[..., previous_special] = np.nan
        result[name] = value
    return result


def incoming_reads(history, ordinary_mass, trace, horizons=HORIZONS):
    """Exact ordinary-source incoming attention per carrier, with exposure.

    history [H,R,R] includes the initial predictor P-1. Readers q must be
    ordinary and predict an observed ordinary token. Carriers are P..N-1.
    Full finite horizons are exposed separately from shortened end windows.
    """
    rows = trace["row_position"]
    special = trace["special_mask"]
    qvalid = ~special[rows]
    qvalid[-1] = False
    qvalid[:-1] &= ~special[rows[:-1] + 1]
    lag = rows[:, None] - rows[None, 1:]
    weights = ratio(history[..., 1:], ordinary_mass[..., None])
    choices = np.cumsum(~special)[rows]
    uniform = ratio(1, choices)
    means, enrichments, counts, full = [], [], [], []
    for lo, hi in horizons:
        eligible = (lag >= lo) & qvalid[:, None]
        if hi:
            eligible &= lag <= hi
        eligible &= ~special[rows[None, 1:]]
        valid = eligible[None] & np.isfinite(weights)
        total = np.where(valid, weights, 0).sum(1)
        count = valid.sum(1)
        means.append(ratio(total, count))
        enrichments.append(ratio(total, np.where(valid, uniform[None, :, None], 0).sum(1)))
        counts.append(count)
        full.append((rows[1:] + hi <= rows[-1] - 1) if hi else np.ones(len(rows)-1, bool))
    return {"fai": np.stack(means, axis=-1), "enrichment": np.stack(enrichments, axis=-1),
            "count": np.stack(counts, axis=-1), "full": np.stack(full, axis=-1)}


def carrier_entry(trace):
    mass = trace["mass"]
    clean = trace["ordinary_mass"]
    evidence = ratio(mass[..., 1], clean)
    local = ratio(mass[..., 4:6].sum(-1), clean)
    delta, local_delta = np.diff(evidence), np.diff(local)
    # b uses its own query row, not the predictor b-1 used by read_metrics.
    valid = np.isfinite(delta) & np.isfinite(local_delta)
    rows = trace["row_position"]
    valid &= ~trace["special_mask"][rows[1:]][None, None]
    valid &= ~trace["special_mask"][rows[:-1]][None, None]
    return np.where(valid, (delta >= .10) & (local_delta <= -.10), np.nan), delta


def joint_tables(entry, reuse, labels, valid_positions, heads_per_layer):
    """All four states for every legal writer->reader, including failed cases.

    Return N/H tables and a same-response, matched carrier contrast. No peak
    intersection selects the denominator. Matrix entries keep physical heads.
    """
    e = entry.reshape(-1, entry.shape[-1])
    r = reuse.reshape(-1, reuse.shape[-1])
    ids = np.arange(len(e))
    deeper = ids[:, None] // heads_per_layer < ids[None] // heads_per_layer
    output = []
    for label in (0, 1):
        mask = (np.asarray(labels) == label) & valid_positions
        ev, rv = np.isfinite(e) & mask, np.isfinite(r) & mask
        e1, r1 = ev & (e > 0), rv & (r > 0)
        e0, r0 = ev & ~e1, rv & ~r1
        denom = ev.astype(np.float32) @ rv.astype(np.float32).T
        tables = [ratio(a.astype(np.float32) @ b.astype(np.float32).T, denom)
                  for a, b in ((e0, r0), (e1, r0), (e0, r1), (e1, r1))]
        output.append(np.where(deeper, np.stack(tables), np.nan))
    return np.stack(output).astype(np.float32)


def matched_joint_gap(entry, reuse, pairs, heads_per_layer):
    e, r = entry.reshape(-1, entry.shape[-1]), reuse.reshape(-1, reuse.shape[-1])
    shape = (4, len(e), len(r))
    if not len(pairs):
        return np.full(shape, np.nan, np.float32)
    hi, ni, right, w = control_coordinates(pairs)
    ev = np.isfinite(e[:, hi]) & np.isfinite(e[:, ni]) & np.isfinite(e[:,right])
    rv = np.isfinite(r[:, hi]) & np.isfinite(r[:, ni]) & np.isfinite(r[:,right])
    denominator = ev.astype(np.float32) @ rv.astype(np.float32).T
    values = []
    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        h = (ev & (e[:, hi] == a)).astype(np.float32) @ (rv & (r[:, hi] == b)).astype(np.float32).T
        n = ((ev & (e[:, ni] == a))*w).astype(np.float32) @ (rv & (r[:, ni] == b)).astype(np.float32).T
        n += ((ev & (e[:, right] == a))*(1-w)).astype(np.float32) @ (rv & (r[:, right] == b)).astype(np.float32).T
        values.append(ratio(h - n, denominator))
    ids = np.arange(len(e))
    deeper = ids[:, None] // heads_per_layer < ids[None] // heads_per_layer
    return np.where(deeper, np.stack(values), np.nan).astype(np.float32)


def target_chain_tables(trace, history_archive, labels, pairs, valid, progress=None):
    """Exhaustive two-hop *structural* evidence paths ending at labeled targets.

    C_ij(q) = sum_{P <= b < q} A_j(q,b)/(1-special_mass_j(q)) * E_i(b),
    with writer layer i < reader layer j and E_i(b) ordinary evidence share.
    This composes attention coefficients, not transported factual content.
    A within-16-position-block rotation of E is a separate provenance control.
    """
    layers, heads, rows = trace["ordinary_mass"].shape
    f, t = layers * heads, rows - 1
    evidence = ratio(trace["mass"][..., 1], trace["ordinary_mass"])[..., 1:].reshape(f, t)
    carrier_valid = ~trace["special_mask"][int(trace["response_start"]):]
    evidence[:, ~carrier_valid] = 0
    shifted = evidence.copy()
    for begin in range(0, t, 16):
        indices = np.flatnonzero(carrier_valid[begin:begin+16]) + begin
        if len(indices) > 1:
            shifted[:, indices] = evidence[:, np.roll(indices, len(indices)//2)]
    out = {"raw": np.full((2, f, f), np.nan, np.float32),
           "matched": np.full((f, f), np.nan, np.float32),
           "excess_matched": np.full((f, f), np.nan, np.float32)}
    qvalid = valid & ~trace["special_mask"][trace["row_position"][:-1]]
    evalid = np.isfinite(evidence)
    safe = np.where(evalid, evidence, 0).astype(np.float32)
    shifted = np.where(np.isfinite(shifted), shifted, 0).astype(np.float32)
    for l in range(1, layers):
        if progress is not None:
            progress(f'two-hop pairs: reader layer {l + 1}/{layers}')
        a = history_archive[f"L{l}"][:, :-1, 1:]
        lag = trace["row_position"][:-1, None] - trace["row_position"][None, 1:]
        a = ratio(a, trace["ordinary_mass"][l, :, :-1, None])
        a = np.where((lag > 0) & carrier_valid[None], a, 0)
        writer_stop, readers = l * heads, slice(l * heads, (l+1) * heads)
        for y in (0, 1):
            weights = mean(a[:, qvalid & (labels == y)], axis=1)
            out["raw"][y, :writer_stop, readers] = safe[:writer_stop] @ weights.T
        selected = pairs[np.all(qvalid[pairs],axis=1)] if len(pairs) else pairs
        if len(selected):
            # Query index is the penultimate axis; align controls before reducing.
            h,left,right,w = control_coordinates(selected)
            weights = mean(a[:,h] - w[None,:,None]*a[:,left] - (1-w)[None,:,None]*a[:,right],axis=1)
            out["matched"][:writer_stop, readers] = safe[:writer_stop] @ weights.T
            out["excess_matched"][:writer_stop, readers] = (safe[:writer_stop] - shifted[:writer_stop]) @ weights.T
        # Missing writer measurements must not be silently interpreted as zero.
        used = np.any(np.nan_to_num(a) > 0, axis=1)
        missing = (~evalid[:writer_stop]).astype(np.float32) @ used.astype(np.float32).T > 0
        for name in ("matched", "excess_matched"):
            out[name][:writer_stop, readers] = np.where(missing, np.nan, out[name][:writer_stop, readers])
        out["raw"][:, :writer_stop, readers] = np.where(missing, np.nan, out["raw"][:, :writer_stop, readers])
    return out


def by_correction(p):
    """Benjamini-Yekutieli FDR, allowing arbitrary dependence between heads."""
    p = np.asarray(p, float)
    flat = p.ravel()
    indices = np.flatnonzero(np.isfinite(flat))
    output = np.full(flat.shape, np.nan)
    if len(indices):
        order = indices[np.argsort(flat[indices])]
        m = len(order)
        factor = np.sum(1 / np.arange(1, m + 1))
        adjusted = flat[order] * m * factor / np.arange(1, m + 1)
        output[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return output.reshape(p.shape)


class Moments:
    """Streaming source-level estimates; no token/head pseudo-replication."""
    def __init__(self):
        self.count = self.total = self.square = None

    def add(self, value):
        value = np.asarray(value, float)
        valid = np.isfinite(value)
        if self.total is None:
            self.total, self.square = np.zeros(value.shape), np.zeros(value.shape)
            self.count = np.zeros(value.shape, np.int32)
        self.total += np.where(valid, value, 0)
        self.square += np.where(valid, value * value, 0)
        self.count += valid

    def merge(self, other):
        if other.total is None:
            return
        if self.total is None:
            self.total, self.square, self.count = other.total.copy(), other.square.copy(), other.count.copy()
        else:
            self.total += other.total
            self.square += other.square
            self.count += other.count

    def finish(self, inference=False):
        from scipy.stats import t
        average = ratio(self.total, self.count)
        result = {"mean": average.astype(np.float32), "sources": self.count}
        if inference:
            variance = ratio(np.maximum(self.square - self.count * average ** 2, 0), self.count - 1)
            se = np.sqrt(ratio(variance, self.count))
            # Student intervals/p-values across source-level paired differences.
            # Constant effects have zero empirical variance; do not invent p=0.
            valid = (self.count >= 3) & (se > 0)
            p = np.where(valid, 2 * t.sf(np.abs(ratio(np.abs(average), se)), self.count - 1), np.nan)
            width = np.where(valid, t.ppf(.975, self.count - 1) * se, np.nan)
            result.update(ci95=np.stack((average-width, average+width)).astype(np.float32),
                          p=p.astype(np.float32), q_by=by_correction(p).astype(np.float32))
        return result


def sample_contrasts(trace, labels, metrics, match_window=32, onset_radius=8):
    start = int(trace["response_start"])
    labels = np.asarray(labels)
    if len(labels) != len(trace["token_ids"]) - start:
        raise ValueError("labels do not cover the captured response")
    valid = np.isin(labels, (0, 1)) & ~trace["special_mask"][start:]
    classes = token_classes(trace["token_text"][start:])
    pairs = bracket_positions(labels, classes, valid, match_window)
    onsets = bracket_positions(labels, classes, valid, match_window, onset_radius)
    nearest = matched_positions(labels, classes, valid, match_window)
    values = np.stack(list(metrics.values()))
    raw = np.stack([mean(values[..., valid & (labels == y)], -1) for y in (0, 1)])
    paired = mean(matched_difference(values,pairs), -1)
    offsets = np.arange(-onset_radius, onset_radius + 1)
    onset = np.full((*values.shape[:-1], len(offsets)), np.nan)
    onset_raw = np.full((2, *values.shape[:-1], len(offsets)), np.nan)
    if len(onsets):
        hi,left,right,w=control_coordinates(onsets)
        h_values=values[...,hi[:,None]+offsets]
        normal=w[:,None]*values[...,left[:,None]+offsets]+(1-w[:,None])*values[...,right[:,None]+offsets]
        onset=mean(h_values-normal,-2)
        onset_raw=np.stack((mean(normal,-2),mean(h_values,-2)))
    known = labels[valid]
    answer_class = "unknown" if not len(known) else ("positive" if (known == 1).any() else "negative")
    # A partly unknown answer is not certified wholly normal.
    if answer_class == "negative" and ((labels < 0) & ~trace["special_mask"][start:]).any():
        answer_class = "unknown"
    slope = []
    x = np.arange(len(labels), dtype=float) / max(1, len(labels)-1)
    for y in (0, 1):
        v = np.isfinite(values) & valid & (labels == y)
        center = x - ratio((v*x).sum(-1), v.sum(-1))[..., None]
        slope.append(ratio((np.where(v, values, 0)*center).sum(-1), np.where(v, center**2, 0).sum(-1)))
    return {"raw": raw, "matched": paired, 'nearest_control_matched':mean(matched_difference(values,nearest),-1),
            "onset": onset, 'onset_raw': onset_raw, "slope": np.stack(slope),
            "answer": mean(values[..., valid], -1), "answer_class": answer_class,
            "pairs": pairs, "onset_pairs": onsets, "valid": valid,
            "normal_tokens": int(((labels == 0) & valid).sum()),
            "hallucinated_tokens": int(((labels == 1) & valid).sum()),
            "special_targets": int(trace["special_mask"][start:].sum()),
            "unknown_tokens": int(((labels < 0) & ~trace["special_mask"][start:]).sum())}
