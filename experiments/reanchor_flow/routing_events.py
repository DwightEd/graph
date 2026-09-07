"""Train-selected, head-resolved event audit of captured routing trajectories.

This is a descriptive audit, separate from the label-free anomaly detector.
Correctness labels choose audit heads on train; test never changes that list.
Event centers use current/past scores only. Future offsets and matched controls
are for visualization, never an online detector input. A winning source-unit ID
is a structural continuity proxy, not evidence lineage or causal acceptance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import save_json
from .scan_dataset import BUCKET_NAMES

PROFILE_NAMES = (*BUCKET_NAMES, "log_total_transport", "reanchor_score", "same_evidence_unit")
GROUP_NAMES = ("nonhallucinated", "hallucinated")


def _mean(values, axis=0):
    array = np.asarray(values, dtype=float)
    count = np.isfinite(array).sum(axis=axis)
    return np.divide(
        np.nansum(array, axis=axis), count,
        out=np.full(np.shape(count), np.nan), where=count > 0,
    )


def _json_array(values):
    array = np.asarray(values, dtype=object)
    array[~np.isfinite(np.asarray(values, dtype=float))] = None
    return array.tolist()


def _summary(values):
    """Pointwise source-cluster intervals, explicitly not simultaneous bands."""

    values = np.asarray(values, dtype=float)
    mean = _mean(values)
    count = np.isfinite(values).sum(axis=0)
    variance = np.divide(
        np.nansum((values - mean) ** 2, axis=0), count - 1,
        out=np.full(mean.shape, np.nan), where=count > 1,
    )
    half_width = 1.96 * np.sqrt(variance / np.maximum(count, 1))
    return {
        "mean": _json_array(mean), "source_count": count.tolist(),
        "lower": _json_array(mean - half_width),
        "upper": _json_array(mean + half_width),
    }


def position_adjusted_gap(score, labels, response_index):
    """Within-sample H-N, pairing labels inside causal log2 position strata.

    Only strata containing both annotated groups contribute. Each stratum has
    equal weight; neither response length nor a future token affects its bin.
    The result retains the layer and head axes.
    """

    bins = np.floor(np.log2(np.asarray(response_index) + 1)).astype(int)
    differences = []
    for position_bin in np.unique(bins):
        normal = (bins == position_bin) & (labels == 0)
        hallucinated = (bins == position_bin) & (labels == 1)
        if normal.any() and hallucinated.any():
            differences.append(score[..., hallucinated].mean(-1) - score[..., normal].mean(-1))
    return np.mean(differences, axis=0) if differences else None


def causal_event_centers(score, threshold, *, refractory=4):
    """Threshold crossings with a left-only refractory period (prefix stable)."""

    above = (np.asarray(score) >= threshold) & (np.asarray(score) > 0)
    crossings = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    centers = []
    previous = -refractory - 1
    for row in crossings:
        if row - previous > refractory:
            centers.append(int(row))
            previous = row
    return np.asarray(centers, dtype=int)


def matched_event_controls(score, labels, centers, *, radius=16, exclusion=4):
    """Pair each event with one unused nearby, same-label, zero-switch row."""

    positions = np.arange(len(score))
    eligible = np.asarray(score) == 0
    for center in centers:
        eligible &= np.abs(positions - center) > exclusion
    pairs = []
    for center in centers:
        if labels[center] not in (0, 1):
            continue
        candidates = np.flatnonzero(
            eligible & (labels == labels[center]) & (np.abs(positions - center) <= radius)
        )
        if len(candidates):
            control = int(candidates[np.argmin(np.abs(candidates - center))])
            pairs.append((int(center), control))
            eligible[control] = False
    return pairs


def _window_profile(transport, score, evidence_unit, center, offsets):
    rows = center + offsets
    valid = (rows >= 0) & (rows < len(score))
    profile = np.full((len(PROFILE_NAMES), len(offsets)), np.nan)
    masses = np.asarray(transport[rows[valid]], dtype=float)
    total = masses.sum(-1)
    fractions = np.divide(masses, total[:, None], out=np.zeros_like(masses), where=total[:, None] > 0)
    profile[:4, valid] = fractions.T
    profile[4, valid] = np.log(total + 1e-12)
    profile[5, valid] = score[rows[valid]]
    center_unit = evidence_unit[center]
    if center_unit >= 0:
        observed = evidence_unit[rows[valid]] >= 0
        continuity = np.where(observed, evidence_unit[rows[valid]] == center_unit, np.nan)
        profile[6, valid] = continuity
    return profile


class RoutingEventAudit:
    """Freeze a few train heads, then compare their same-position event windows."""

    def __init__(self, *, max_heads=3, before=8, after=12):
        self.max_heads = max_heads
        self.offsets = np.arange(-before, after + 1)
        self.selection = {}

    def discover(self, scans, labels):
        """Use train labels only for audit selection, never for detector fitting."""

        if scans.split != "train":
            raise ValueError("event heads must be discovered on train scans")
        source_gaps = defaultdict(lambda: defaultdict(list))
        for record in tqdm(scans.records, desc="event train head selection", unit="sample"):
            sample = scans.load(record.sample_id, fields=("reanchor_score",))
            gap = position_adjusted_gap(sample["reanchor_score"], labels.load(sample), sample.response_index)
            if gap is not None:
                source_gaps[sample.task_type][sample.source_id].append(gap)
        self.selection = {record.task_type: [] for record in scans.records}
        for task, sources in source_gaps.items():
            source_values = np.stack([np.mean(values, axis=0) for values in sources.values()])
            gap = source_values.mean(0)
            ranked = np.argsort(-np.abs(gap), axis=None, kind="stable")
            for index in ranked[:self.max_heads]:
                layer, head = np.unravel_index(index, gap.shape)
                if gap[layer, head] == 0:
                    continue
                values = source_values[:, layer, head]
                self.selection[task].append({
                    "layer": int(layer), "head": int(head),
                    "train_position_adjusted_gap": float(gap[layer, head]),
                    "train_paired_sources": len(values),
                    "train_source_sign_agreement": float(np.mean(np.sign(values) == np.sign(gap[layer, head]))),
                    "selection_uses_train_labels": True,
                })
        # Only selected heads' positive scores are retained for this weighted
        # quantile: at most a few values per token, never all heads' trajectories.
        positives = defaultdict(list)
        weights = scans.sample_weights()
        for record in tqdm(scans.records, desc="event train thresholds", unit="sample"):
            heads = self.selection.get(record.task_type, [])
            if not heads:
                continue
            sample = scans.load(record.sample_id, fields=("reanchor_score",))
            for selected in heads:
                key = (record.task_type, selected["layer"], selected["head"])
                score = sample["reanchor_score"][key[1], key[2]]
                positive = score[score > 0]
                if len(positive):
                    positives[key].append((positive, weights[record.sample_id] / len(positive)))
        for task, heads in self.selection.items():
            for selected in heads:
                values = positives[task, selected["layer"], selected["head"]]
                score = np.concatenate([value for value, _ in values])
                weight = np.concatenate([np.full(len(value), item_weight) for value, item_weight in values])
                order = np.argsort(score)
                quantile_index = np.searchsorted(np.cumsum(weight[order]), 0.9 * weight.sum())
                selected["threshold"] = float(score[order[min(quantile_index, len(order) - 1)]])
                selected["threshold_positive_quantile"] = 0.9
        return self

    def _collect(self, scans, labels, pairs_file):
        profiles = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        counts = defaultdict(lambda: {name: {"events": 0, "matched_pairs": 0, "absolute_offset_sum": 0} for name in GROUP_NAMES})
        for record in tqdm(scans.records, desc=f"event {scans.split} profiles", unit="sample"):
            heads = self.selection.get(record.task_type, [])
            if not heads:
                continue
            sample = scans.load(record.sample_id, fields=(
                "reanchor_score", "reanchor_bucket_transport", "reanchor_bucket_source_unit_id",
                "reanchor_bucket_source_position",
            ))
            annotation = labels.load(sample)
            for selected in heads:
                layer, head = selected["layer"], selected["head"]
                key = (record.task_type, layer, head)
                score = sample["reanchor_score"][layer, head]
                centers = causal_event_centers(score, selected["threshold"])
                pairs = matched_event_controls(score, annotation, centers)
                transport = sample["reanchor_bucket_transport"][layer, head]
                evidence_unit = sample["reanchor_bucket_source_unit_id"][layer, head, :, 0]
                pair_index = dict(pairs)
                for center in centers:
                    control = pair_index.get(int(center))
                    pairs_file.write(json.dumps({
                        "sample_id": sample.sample_id, "source_id": sample.source_id,
                        "task_type": sample.task_type, "layer": layer, "head": head,
                        "predictor_position": int(sample.row_position[center]),
                        "predicted_token_position": int(sample.row_position[center]) + 1,
                        "response_index": int(sample.response_index[center]),
                        "label": int(annotation[center]), "score": float(score[center]),
                        "threshold": selected["threshold"],
                        "control_response_index": int(sample.response_index[control]) if control is not None else None,
                        "winning_source_position_by_bucket": sample["reanchor_bucket_source_position"][layer, head, center].tolist(),
                        "winning_source_unit_by_bucket": sample["reanchor_bucket_source_unit_id"][layer, head, center].tolist(),
                    }) + "\n")
                for group, name in enumerate(GROUP_NAMES):
                    counts[key][name]["events"] += int(np.sum(annotation[centers] == group))
                    group_pairs = [(event, control) for event, control in pairs if annotation[event] == group]
                    if not group_pairs:
                        continue
                    counts[key][name]["matched_pairs"] += len(group_pairs)
                    counts[key][name]["absolute_offset_sum"] += sum(abs(event - control) for event, control in group_pairs)
                    event_values, control_values = [], []
                    for event, control in group_pairs:
                        event_values.append(_window_profile(transport, score, evidence_unit, event, self.offsets))
                        control_values.append(_window_profile(transport, score, evidence_unit, control, self.offsets))
                    # Pair first, sample second, source third. Long answers and
                    # many model answers for one source do not acquire more weight.
                    profiles[key][name][record.source_id].append(np.stack((
                        _mean(event_values), _mean(control_values),
                        _mean(np.asarray(event_values) - np.asarray(control_values)),
                    )))
        return profiles, counts

    def run(self, scans, labels, output, *, plot=True):
        """Stream scans into source-balanced profiles of matched event/control pairs."""

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "event_pairs.jsonl").open("w") as pairs_file:
            profiles, counts = self._collect(scans, labels, pairs_file)
        report = {
            "schema": 1, "split": scans.split, "status": "exploratory_structural_audit",
            "selection": self.selection, "offsets": self.offsets.tolist(),
            "profile_names": list(PROFILE_NAMES), "tasks": {},
            "event_index": "event_pairs.jsonl", "source_bucket_order": list(BUCKET_NAMES),
            "interpretation": {
                "heads": "Selected only on train by absolute source-balanced within-position-bin H-N gap; heads remain separate.",
                "events": "Current/past threshold crossings; four-row left refractory. Future offsets are descriptive only.",
                "controls": "Unused zero-switch rows in the same sample and center label, within 16 token positions; exclude four rows around events.",
                "profiles": "Only matched event/control pairs; equal samples within source, then equal sources.",
                "intervals": "Pointwise 95% normal-approximation source-cluster intervals; no multiple-comparison or selection correction.",
                "evidence": "Bucket names and winning unit continuity do not establish factual evidence, integration, causal acceptance, or lineage through a hub.",
            },
        }
        for task, heads in self.selection.items():
            report["tasks"][task] = []
            for selected in heads:
                key = (task, selected["layer"], selected["head"])
                entry = {**selected, "groups": {}}
                grouped_sources = {}
                for name in GROUP_NAMES:
                    sources = {source: _mean(values) for source, values in profiles[key][name].items()}
                    grouped_sources[name] = sources
                    statistics = counts[key][name]
                    paired_count = statistics["matched_pairs"]
                    group_report = {
                        "event_count": statistics["events"], "matched_pair_count": paired_count,
                        "source_count": len(sources),
                        "mean_absolute_control_offset": statistics["absolute_offset_sum"] / paired_count if paired_count else None,
                    }
                    if sources:
                        values = np.stack(list(sources.values()))
                        for index, profile_name in enumerate(("event", "control", "event_minus_control")):
                            group_report[profile_name] = _summary(values[:, index])
                    entry["groups"][name] = group_report
                paired_sources = sorted(set(grouped_sources[GROUP_NAMES[0]]) & set(grouped_sources[GROUP_NAMES[1]]))
                entry["paired_hallucinated_minus_nonhallucinated_sources"] = len(paired_sources)
                if paired_sources:
                    differences = np.stack([
                        grouped_sources[GROUP_NAMES[1]][source][2] - grouped_sources[GROUP_NAMES[0]][source][2]
                        for source in paired_sources
                    ])
                    entry["paired_event_minus_control_group_gap"] = _summary(differences)
                report["tasks"][task].append(entry)
                if plot:
                    entry["figure"] = self._plot(task, entry, output)
        save_json(output / "event_audit.json", report)
        return report

    def _plot(self, task, entry, output):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True)
        for metric, ax in enumerate(axes.flat[:len(PROFILE_NAMES)]):
            for group, color in zip(GROUP_NAMES, ("#2563eb", "#dc2626")):
                data = entry["groups"][group]
                for kind, linestyle in (("event", "-"), ("control", "--")):
                    if kind not in data:
                        continue
                    mean = np.asarray(data[kind]["mean"], dtype=float)[metric]
                    lower = np.asarray(data[kind]["lower"], dtype=float)[metric]
                    upper = np.asarray(data[kind]["upper"], dtype=float)[metric]
                    ax.plot(self.offsets, mean, color=color, linestyle=linestyle, label=f"{group} {kind}")
                    if kind == "event":
                        ax.fill_between(self.offsets, lower, upper, color=color, alpha=.12)
            ax.axvline(0, color="0.5", linewidth=.8)
            ax.set_title(PROFILE_NAMES[metric].replace("_", " "), fontsize=10)
            ax.set_xlabel("Token offset from center")
            ax.grid(alpha=.15)
        axes.flat[7].axis("off")
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[7].legend(handles, legend_labels, loc="upper left", fontsize=9)
        counts = "; ".join(
            f"{name}: {entry['groups'][name]['matched_pair_count']} pairs / {entry['groups'][name]['source_count']} sources"
            for name in GROUP_NAMES
        )
        fig.suptitle(f"{task} L{entry['layer']}H{entry['head']} | train-selected structural event audit\n{counts}", fontsize=12)
        fig.text(.5, .01, "Future offsets are descriptive. Shading: pointwise source-cluster intervals. Continuity is not causal evidence acceptance.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .93))
        filename = f"head_event_{task}_L{entry['layer']}H{entry['head']}.png"
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        return filename
