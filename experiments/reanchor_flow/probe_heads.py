"""Explain a saved supervised readout, then screen heads on training sources.

This is a supervised diagnostic of the existing linear model. Parameter norms
are sensitivities, not empirical importance. Position-matched contributions are
associations, not causal effects or the performance of a sparse detector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .artifacts import save_json
from .routing_events import position_adjusted_gap
from .routing_probe import SupervisedRoutingProbe
from .routing_transition import ILR_BASIS, RoutingSequence
from .scan_dataset import BUCKET_NAMES, ScanDataset, ScanLabelStore


@dataclass(frozen=True)
class HeadContributions:
    current: np.ndarray  # [row, layer, head], standardized current-state terms
    difference: np.ndarray  # [row, layer, head], standardized adjacent differences
    context: np.ndarray  # [row], the routing probe's own position/context terms
    intercept: float

    @property
    def per_head(self):
        return self.current + self.difference

    @property
    def score(self):
        return self.per_head.sum(axis=(1, 2)) + self.context + self.intercept


class HeadReadout:
    """Exact additive decomposition; each head retains its own contribution."""

    def __init__(self, probe: SupervisedRoutingProbe):
        self.probe = probe
        self.shape = tuple(probe.state_shape)
        self.width = int(np.prod(self.shape))

    def decompose(self, sequence: RoutingSequence) -> HeadContributions:
        if tuple(sequence.state.shape[1:]) != self.shape:
            raise ValueError("probe and scan layer/head state dimensions differ")
        name = "supervised_routing"
        rows = len(sequence.state)
        current = np.empty((rows, *self.shape[:2]))
        difference = np.empty_like(current)
        context = np.empty(rows)
        for start in range(0, rows, 256):
            batch = np.arange(start, min(start + 256, rows))
            features = self.probe._features(sequence, batch)[name]
            terms = ((features - self.probe.mean[name]) / self.probe.scale[name]
                     * self.probe.coefficient[name])
            current[batch] = terms[:, :self.width].reshape(-1, *self.shape).sum(-1)
            difference[batch] = terms[:, self.width:2 * self.width].reshape(
                -1, *self.shape).sum(-1)
            context[batch] = terms[:, 2 * self.width:].sum(-1)
        return HeadContributions(current, difference, context, float(self.probe.intercept[name]))

    def parameter_summary(self) -> list[dict]:
        """Rank standardized coefficient norms, with original-coordinate slopes.

        Source slopes operate on log smoothed fractions. The current term holds
        delta fixed; changing current with previous fixed instead uses the sum
        of current and delta slopes. Contrasts are not causal interventions.
        """
        name = "supervised_routing"
        coefficient = np.asarray(self.probe.coefficient[name])[:2 * self.width]
        standardized = coefficient.reshape(2, *self.shape)
        slope = (coefficient / self.probe.scale[name][:2 * self.width]).reshape(
            2, *self.shape)
        norms = np.sqrt(np.square(standardized).sum(axis=(0, 3)))
        order = np.argsort(-norms.ravel(), kind="stable")
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(1, len(order) + 1)
        entries = []
        for layer, head in np.ndindex(self.shape[:2]):
            entry = {"layer": layer, "head": head,
                     "parameter_rank": int(rank.reshape(norms.shape)[layer, head]),
                     "standardized_coefficient_norm": float(norms[layer, head])}
            for component, values, weights in zip(
                ("current", "difference"), slope[:, layer, head],
                standardized[:, layer, head], strict=True,
            ):
                entry[component] = {
                    "standardized_coefficient_norm": float(np.linalg.norm(weights)),
                    "log_fraction_contrast": dict(zip(
                        BUCKET_NAMES, (values[:3] @ ILR_BASIS).tolist(), strict=True)),
                    "log_total_slope": float(values[3]),
                }
            immediate = slope[:, layer, head].sum(0)
            entry["fixed_previous"] = {
                "log_fraction_contrast": dict(zip(
                    BUCKET_NAMES, (immediate[:3] @ ILR_BASIS).tolist(), strict=True)),
                "log_total_slope": float(immediate[3]),
            }
            entries.append(entry)
        return entries


class SourceGaps:
    """Equal samples within a source, then equal sources; preserve head axes."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.sources = {}
        self.samples = 0

    def add(self, source_id: str, gap: np.ndarray | None):
        if gap is None:
            return
        value = np.asarray(gap, dtype=float)
        if value.shape != self.shape or not np.isfinite(value).all():
            raise ValueError("source gap has incompatible shape or nonfinite values")
        if source_id not in self.sources:
            self.sources[source_id] = [np.zeros(self.shape), 0]
        self.sources[source_id][0] += value
        self.sources[source_id][1] += 1
        self.samples += 1

    def summary(self):
        count = len(self.sources)
        values = np.asarray([total / n for total, n in self.sources.values()])
        return {
            "paired_sources": count, "paired_samples": self.samples,
            "mean_gap": values.mean(0) if count else np.full(self.shape, np.nan),
            "positive_sources": (values > 0).sum(0) if count else np.zeros(self.shape, int),
            "positive_source_fraction": (values > 0).mean(0) if count else np.full(self.shape, np.nan),
        }


def freeze_selection(calibration: SourceGaps, *, limit=20) -> list[dict]:
    """Order by positive held-out-train total gap; never inspect test scores."""
    summary = calibration.summary()
    gap = summary["mean_gap"][0]  # [component, layer, head], component 0 = total
    eligible = np.flatnonzero(np.isfinite(gap) & (gap > 0))
    order = eligible[np.argsort(-gap.ravel()[eligible], kind="stable")][:limit]
    return [
        {"layer": int(index // gap.shape[1]), "head": int(index % gap.shape[1]),
         "calibration_mean_gap": float(gap.ravel()[index])}
        for index in order
    ]


def _collect(scans, labels, readouts, groups):
    """Stream each selected scan once; retain only source-level head summaries."""
    result = {group: {task: SourceGaps((3, *model.shape[:2]))
                     for task, model in readouts.items()} for group in set(groups.values())}
    for record in tqdm(scans.records, desc=f"{scans.split} head contributions", unit="sample"):
        group = groups.get(record.source_id)
        if group is None:
            continue
        scan = scans.load(record.sample_id, fields=("reanchor_bucket_transport",))
        contributions = readouts[record.task_type].decompose(RoutingSequence.from_scan(scan))
        target = labels.load(scan)
        current = position_adjusted_gap(
            np.moveaxis(contributions.current, 0, -1), target, scan.response_index)
        difference = position_adjusted_gap(
            np.moveaxis(contributions.difference, 0, -1), target, scan.response_index)
        gap = None if current is None else np.stack((current + difference, current, difference))
        result[group][record.task_type].add(record.source_id, gap)
    return result


def _head_statistics(summary, layer, head):
    def number(value):
        return float(value) if np.isfinite(value) else None

    return {
        "paired_sources": summary["paired_sources"],
        "paired_samples": summary["paired_samples"],
        **{component: {
            "mean_gap": number(summary["mean_gap"][i, layer, head]),
            "positive_sources": int(summary["positive_sources"][i, layer, head]),
            "positive_source_fraction": number(summary["positive_source_fraction"][i, layer, head]),
        } for i, component in enumerate(("total", "current", "difference"))},
    }


def build_report(readouts, fit, calibration, test, selected):
    report = {}
    for task, readout in readouts.items():
        summaries = {name: values[task].summary() for name, values in (
            ("fit", fit), ("calibration", calibration), ("test", test))}
        entries = readout.parameter_summary()
        calibration_gap = summaries["calibration"]["mean_gap"][0]
        eligible = np.flatnonzero(np.isfinite(calibration_gap))
        ordered = eligible[np.argsort(-calibration_gap.ravel()[eligible], kind="stable")]
        calibration_rank = {int(index): rank + 1 for rank, index in enumerate(ordered)}
        selected_rank = {(item["layer"], item["head"]): i + 1
                         for i, item in enumerate(selected[task])}
        for entry in entries:
            layer, head = entry["layer"], entry["head"]
            entry["selected_rank"] = selected_rank.get((layer, head))
            entry["calibration_gap_rank"] = calibration_rank.get(layer * readout.shape[1] + head)
            entry["contribution_gap"] = {
                name: _head_statistics(summary, layer, head)
                for name, summary in summaries.items()}
        report[task] = {"selected_heads": selected[task], "heads": entries}
    return report


def run_audit(analysis, *, scans=None, cache=None):
    analysis = Path(analysis)
    partition = json.loads((analysis / "source_partition.json").read_text())
    frozen = json.loads((analysis / "frozen_detector.json").read_text())
    capture = Path(scans or frozen["capture_root"])
    train, test = ScanDataset(capture / "train"), ScanDataset(capture / "test")
    fit_ids, calibration_ids = (set(map(str, partition[key])) for key in (
        "fit_source_ids", "calibration_source_ids"))
    test_ids = {record.source_id for record in test.records}
    if fit_ids & calibration_ids or (fit_ids | calibration_ids) & test_ids:
        raise ValueError("fit, calibration and test sources must be disjoint")
    tasks = sorted({record.task_type for record in test.records})
    readouts = {task: HeadReadout(SupervisedRoutingProbe.load(
        analysis / "models" / f"{task}_supervised_probe.npz")) for task in tasks}
    output = analysis / "head_audit"
    output.mkdir(exist_ok=True)
    train_groups = {source: "fit" for source in fit_ids}
    train_groups.update({source: "calibration" for source in calibration_ids})
    labels = ScanLabelStore(train, dataset_root=Path(cache) / "train" if cache else None)
    training = _collect(train, labels, readouts, train_groups)
    selected = {task: freeze_selection(training["calibration"][task]) for task in tasks}
    selection = {
        "schema": 1, "labels_used": True,
        "rule": "top 20 positive position-matched head contribution gaps on calibration train sources",
        "selection_budget_is_not_validated_sparse_detector": True,
        "fit_source_ids": sorted(fit_ids), "calibration_source_ids": sorted(calibration_ids),
        "selected_heads": selected,
    }
    save_json(output / "frozen_head_selection.json", selection)
    # All head choices are on disk before opening test labels or test scan data.
    labels = ScanLabelStore(test, dataset_root=Path(cache) / "test" if cache else None)
    testing = _collect(test, labels, readouts, {source: "test" for source in test_ids})["test"]
    report = {
        "schema": 1, "labels_used": True, "head_averaging": False,
        "interpretation": "supervised additive readout diagnostic; not causal head importance",
        "position_control": "within-sample H-N in log2(response_index+1) bins; equal bins, samples, sources",
        "slope_semantics": "current holds difference fixed; fixed_previous adds current and difference slopes",
        "limitations": [
            "coefficient norms are sensitivity, not observed contribution",
            "correlated heads can compensate; contributions are not independent causal effects",
            "head selection uses training labels and does not create an unsupervised detector",
            "coarse position bins leave residual position confounding",
            "only within-sample bins containing both annotated groups enter gaps",
        ],
        "tasks": build_report(readouts, training["fit"], training["calibration"], testing, selected),
    }
    save_json(output / "head_report.json", report)
    for task in tasks:
        print(f"{task}: calibration-selected heads (zero-based): "
              + ", ".join(f"L{x['layer']}H{x['head']}" for x in selected[task]))
    print(f"Head contribution report: {output / 'head_report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--scans", type=Path, help="override original train/test scan root")
    parser.add_argument("--cache", type=Path, help="override train/test research dataset root")
    args = parser.parse_args()
    run_audit(args.analysis, scans=args.scans, cache=args.cache)


if __name__ == "__main__":
    main()
