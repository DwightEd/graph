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
    full_fai = np.where(trace["fai_full_horizon"][None, None, 1:], fai, np.nan)
    full_peaks = peak_mask(full_fai, relative=fai_relative_prominence).reshape(heads, -1)
    full_observed, full_expected, full_counts = pair_alignment(flat_w[local], full_peaks[global_], full)
    # Primary carrier audit includes ALL heads. Span-ranked local/global sets
    # above are a paper-style descriptive control, not a functional head filter.
    head_ids = np.arange(heads)
    same, same_null, same_counts = pair_alignment(flat_w, full_peaks, full, hi=0)
    deeper = head_ids[:, None] // waad.shape[1] < head_ids[None] // waad.shape[1]
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
        "coupling_full_horizon_anchor_count": full_counts,
        "coupling_write_heads": local, "coupling_read_heads": global_,
        "same_carrier_write_heads": head_ids, "same_carrier_read_heads": head_ids,
        "same_carrier_deeper_observed": np.where(deeper, same, np.nan),
        "same_carrier_deeper_null": np.where(deeper, same_null, np.nan),
        "same_carrier_deeper_lift": np.where(deeper, same - same_null, np.nan),
        "same_carrier_anchor_count": same_counts,
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
            gaps.append(_finite_mean(values[..., hall], axis=-1) - _finite_mean(values[..., clean], axis=-1))
    return _finite_mean(np.stack(gaps)) if gaps else np.full(values.shape[:-1], np.nan)


def label_observations(trace, labels):
    """q -> q+1 labels; exclude final paper row, NOT the first predictor row."""
    labels = np.asarray(labels)
    if len(labels) != len(trace["row_position"]) - 1:
        raise ValueError("labels must cover every captured predicted response token")
    rows = trace["row_position"][:-1] + 1 - int(trace["response_start"])
    result = {name + "_matched_h_minus_n": matched_label_gap(
        np.asarray(trace[name])[..., :-1], labels, rows)
        for name in ("distance", "waad", "message_waad")}
    result["waad_peak_matched_h_minus_n"] = matched_label_gap(
        peak_mask(trace["waad"][..., :-1], absolute=.5, relative=0).astype(float), labels, rows)
    # FAI belongs to carrier b, whereas the above reads belong to predictor b-1.
    result["fai_carrier_matched_h_minus_n"] = matched_label_gap(
        trace["fai"][..., 1:], labels, rows)
    result.update(prompt_history_trends(trace, labels))
    result.update(nonhallucinated_tokens=int((labels == 0).sum()),
                  hallucinated_tokens=int((labels == 1).sum()), unknown_tokens=int((labels < 0).sum()))
    return result


def _slope(values, x, selected):
    """OLS on actual response progress, retaining every layer/head coordinate."""
    valid = np.isfinite(values) & np.asarray(selected)[None, None]
    count = valid.sum(-1)
    mean_x = _ratio((valid * x).sum(-1), count)
    centered = x - mean_x[..., None]
    numerator = (np.where(valid, values, 0) * centered).sum(-1)
    denominator = (np.where(valid, centered, 0) ** 2).sum(-1)
    return np.where(count >= 3, _ratio(numerator, denominator), np.nan)


def prompt_history_trends(trace, labels=None):
    """Test drift directly; opportunity correction is a uniform-source control.

    Normal-token subsets of mixed answers and fully nonhallucinated answers
    are distinct populations. Neither is assumed in advance to drift.
    """
    buckets = trace["attention_buckets"][..., :-1, :]
    prompt, history = buckets[..., :2].sum(-1), buckets[..., 2:].sum(-1)
    positions = np.asarray(trace["row_position"][:-1])
    x = np.arange(len(positions)) / max(1, len(positions) - 1)
    opportunity = int(trace["response_start"]) / (positions + 1)
    values = {"prompt": prompt, "history": history, "prompt_excess_uniform": prompt - opportunity}
    masks = {"all": np.ones(len(positions), bool)}
    if labels is not None:
        masks.update(nonhallucinated=labels == 0, hallucinated=labels == 1,
                     fully_nonhallucinated=np.full(len(labels), bool(np.all(labels == 0))))
    return {f"{name}_slope_{group}": _slope(value, x, mask)
            for group, mask in masks.items() for name, value in values.items()}


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
    flat = data.reshape(len(data), -1)
    finite = np.isfinite(flat).astype(float)
    filled = np.where(np.isfinite(flat), flat, 0)
    draws = []
    for begin in range(0, repetitions, 32):
        weights = rng.multinomial(len(data), np.full(len(data), 1 / len(data)),
                                  size=min(32, repetitions - begin))
        draws.append(_ratio(weights @ filled, weights @ finite))
    draws = np.concatenate(draws).reshape((repetitions, *mean.shape))
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


def summarize_head_pairs(entries, output, destination):
    """Align physical head IDs across samples, then average within each source.

    Stream one source at a time; do not retain [sample,head,head]. Missing or
    layer-ineligible pairs stay missing. No averaging over heads, no head filter
    based on hallucination labels, and no per-pair significance claim.
    """
    from tqdm.auto import tqdm

    sources = {}
    for entry in entries:
        sources.setdefault(entry["source_id"], []).append(entry)
    with np.load(output / entries[0]["path"], allow_pickle=False) as stored:
        layers, heads = stored["distance_mean"].shape
    shape = (layers * heads,) * 2
    total, positive = np.zeros(shape), np.zeros(shape, np.int32)
    count = np.zeros(shape, np.int32)
    for source_entries in tqdm(sources.values(), desc="head-pair sources", unit="source", leave=False):
        value, samples = np.zeros(shape), np.zeros(shape, np.int32)
        for entry in source_entries:
            with np.load((output / entry["path"]).with_suffix(".audit.npz"), allow_pickle=False) as stored:
                rows, columns = stored["same_carrier_write_heads"], stored["same_carrier_read_heads"]
                lift = stored["same_carrier_deeper_lift"]
            index = np.ix_(rows, columns)
            valid = np.isfinite(lift)
            value[index] += np.where(valid, lift, 0)
            samples[index] += valid
        mean = _ratio(value, samples)
        valid = np.isfinite(mean)
        total += np.where(valid, mean, 0)
        count += valid
        positive += valid & (mean > 0)
    result = {"mean_lift": _ratio(total, count), "valid_sources": count,
              "positive_source_fraction": _ratio(positive, count),
              "layers": np.array(layers), "heads": np.array(heads)}
    np.savez_compressed(destination, **result)
    return result


from .attention_rhythm_plot import plot_sample, plot_timeline


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


def summarize_rhythm(entries, output, *, bootstrap=500, seed=2026, evaluated=False):
    from tqdm.auto import tqdm
    from .attention_rhythm_plot import plot_population

    keys = ("waad_peak_rate", "bucket_missed_fraction", "attention_message_waad_mae",
            "prompt_slope_all", "history_slope_all", "prompt_excess_uniform_slope_all")
    if evaluated:
        keys += ("waad_matched_h_minus_n", "message_waad_matched_h_minus_n", "distance_matched_h_minus_n",
                 "waad_peak_matched_h_minus_n", "fai_carrier_matched_h_minus_n")
        keys += tuple(f"{signal}_slope_{subset}" for signal in ("prompt", "history", "prompt_excess_uniform")
                      for subset in ("nonhallucinated", "hallucinated", "fully_nonhallucinated"))
    report = {
        "scope": "raw attention rhythms, native two-hop examples and direct-logit accounting",
        "labels_used_for_capture": False, "labels_used_for_comparison": evaluated,
        "detection_metrics_run": False,
        "not_tested": ["factual content identity", "causal mediation through the carrier",
                       "missed-entry / failed-integration / overwrite / silent-readout classification",
                       "a mechanism-based hallucination detector"],
        "head_roles": "sample-specific span ranks for descriptive pairing; not permanent semantic roles",
        "coupling_null": "count-preserving uniform placement; not an autocorrelation-preserving test",
        "limitations": ["FAI uses future observed response and is offline",
                        "peak rules are explicit audit conventions; conditional CIs are not FDR-corrected",
                        "coarse position matching does not control lexical or generator differences",
                        "two-hop examples are budget-selected witnesses, not representative causal effects",
                        "positive signed readout favors the observed token, which may itself be hallucinated"],
        "groups": {},
    }
    groups = [(split, task) for split in sorted({entry["split"] for entry in entries})
              for task in ("ALL", "QA", "Summary", "Data2txt")]
    for split, task in tqdm(groups, desc="rhythm cohorts", unit="group"):
        chosen = [e for e in entries if e["split"] == split and (task == "ALL" or e["task_type"] == task)]
        if not chosen:
            continue
        group = {"samples": len(chosen), "sources": len({e["source_id"] for e in chosen}),
                 "response_tokens": sum(e["response_tokens"] for e in chosen),
                 "full_response_tokens": sum(e["full_response_tokens"] for e in chosen),
                 "native_relay_examples": sum(e["relay_examples"] for e in chosen),
                 "detail_samples": sum(bool(e["plot"]) for e in chosen)}
        if evaluated:
            for label in ("nonhallucinated_tokens", "hallucinated_tokens", "unknown_tokens"):
                group[label] = sum(e[label] for e in chosen)
        for key in tqdm(keys, desc=f"{split}/{task} source intervals", leave=False, unit="measure"):
            group[key] = source_bootstrap(chosen, key, repetitions=bootstrap, seed=seed)
        name = f"{split}_{task}"
        pair_path = output / f"head_pairs_{name}.npz"
        pairs = summarize_head_pairs(chosen, output, pair_path)
        group["head_pairs"] = pair_path.name
        report["groups"][f"{split}/{task}"] = group
        plot_population(name, group, pairs, output, evaluated)
        tqdm.write(f"{split}/{task:9s} samples={group['samples']} "
                   f"tokens={group['response_tokens']}/{group['full_response_tokens']} "
                   f"sources={group['sources']} native_relay_examples={group['native_relay_examples']}")
    save_json(output / "summary.json", report)
    lines = ["# 原始 attention 与原生载体审计", "",
             "本报告检查逐 head 结构、正常轨迹趋势及有限两跳实例。没有训练检测器，没有机制检测 AUROC。", "",
             "| split/task | 样本 | source | token/完整 token | 正常 token | 幻觉 token | 两跳实例 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, group in report["groups"].items():
        lines.append(f"| {name} | {group['samples']} | {group['sources']} | "
                     f"{group['response_tokens']}/{group['full_response_tokens']} | "
                     f"{group.get('nonhallucinated_tokens', '未连接标签')} | "
                     f"{group.get('hallucinated_tokens', '未连接标签')} | {group['native_relay_examples']} |")
    lines += ["", "先打开 `gallery.html` 看完整回答曲线、原始来源图和两跳实例。", "",
              "`summary.json` 保留每个 layer/head 的趋势与 H−N 差异及来源区间；`head_pairs_*.npz` "
              "按实际 head ID 保存同一载体、严格递增层序的配对结果与可用来源数。", "",
              "正常 token 的趋势与完全正常回答的趋势分别报告。prompt 占比下降也可能来自可见 history "
              "数量增长，需同时检查减去均匀来源占比后的对照趋势。", "",
              "两跳实例的真实残差、全部 head code、MLP 写入及带符号读出在原 NPZ 的 `relay_*` 字段。"
              "它们显示运算与结构，并未验证某项事实被正确传递，也不把模长当功能贡献。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _gallery(entries, output)
    return report


def _gallery(entries, output):
    from html import escape
    from urllib.parse import quote

    rows = []
    for entry in entries:
        path = Path(entry["path"])
        links = []
        for suffix, label in ((".timeline.png", "完整回答"), (".png", "逐 head 原图"),
                              (".full_sources.png", "全部来源列"), (".relay.png", "原生两跳"),
                              (".relays.json", "两跳坐标"), (".events.json", "事件文本"),
                              (".npz", "NPZ")):
            target = path.with_suffix(suffix)
            if (output / target).is_file():
                links.append(f'<a href="{quote(str(target), safe="/")}">{label}</a>')
        identity = f"{entry['split']}/{entry['task_type']}/{entry['sample_id']}"
        rows.append(f"<tr><td>{escape(identity)}</td><td>{entry['response_tokens']}</td>"
                    f"<td>{entry['relay_examples']}</td><td>{' · '.join(links)}</td></tr>")
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐 head 机制审计</title>
<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#203344}
table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dbe3e9;text-align:left}
a{color:#17619d}input{padding:10px;width:350px}p{line-height:1.6}</style>
<h1>原始 attention 与原生载体审计</h1><p>逐 head 曲线覆盖完整回答；原图与两跳实例只在预先选定的展示样本采集。
红色标注在事后连接，未用于选 head 或位置。两跳结构和正的读出贡献不等于事实证据被正确使用。</p>
<p><a href="summary.md">审计说明</a> · <a href="summary.json">逐 head 总体统计</a></p>
<input id="filter" placeholder="筛选 split、task 或样本 ID"><table><thead><tr><th>样本</th><th>response tokens</th>
<th>两跳实例</th><th>产物</th></tr></thead><tbody>"""
    html += "\n".join(rows) + """</tbody></table><script>
document.getElementById('filter').addEventListener('input', function() {
const query = this.value.toLowerCase();
document.querySelectorAll('tbody tr').forEach(row => {row.hidden = !row.textContent.toLowerCase().includes(query);});
});</script></html>"""
    (output / "gallery.html").write_text(html, encoding="utf-8")
