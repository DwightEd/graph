"""Post-hoc cohort comparisons on saved, head-resolved mechanism observations.

The census uses each sample's complete preselection scan, never the union of
selected-target prefixes. Means first average annotated tokens within a sample
and retain every (layer, head); the paired effect then compares both token groups
within the same sample. These are descriptive effects, not causal estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import save_json

COHORT_REPORT_NAME = "cohort_summary.json"
BUCKETS = ("prompt_evidence", "other_prompt", "remote_response", "recent_local")
GROUPS = ("annotated_nonhallucinated", "annotated_hallucinated")
STRUCTURAL_METRICS = (*BUCKETS, "local_to_long_switch_rate")
MIN_PLOT_SAMPLES = 3


def _json_array(value: np.ndarray) -> list:
    array = np.asarray(value, dtype=object)
    array[~np.isfinite(np.asarray(value, dtype=float))] = None
    return array.tolist()


def _masked_mean(value: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    count = valid.sum(axis=axis)
    return np.divide(
        np.where(valid, value, 0).sum(axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=float),
        where=count > 0,
    )


@dataclass
class HeadCohort:
    """Sample-balanced moments and within-sample effects, with heads intact."""

    metric_names: tuple[str, ...]
    samples: list[np.ndarray] = field(default_factory=list)

    def add(self, group_means: np.ndarray) -> None:
        # [annotated group, metric, layer, head]; no reduction over heads.
        if group_means.ndim != 4 or group_means.shape[:2] != (
            2,
            len(self.metric_names),
        ):
            raise ValueError("cohort observations need [group,metric,layer,head] axes")
        if self.samples and self.samples[0].shape != group_means.shape:
            raise ValueError("cohort samples use different layer/head architectures")
        self.samples.append(group_means)

    def summary(self) -> dict:
        if not self.samples:
            return {"sample_count": 0, "metrics": {}}
        values = np.stack(self.samples)
        valid = np.isfinite(values)
        means = _masked_mean(values, valid, axis=0)
        counts = valid.sum(axis=0)
        paired = values[:, 1] - values[:, 0]
        paired_valid = np.isfinite(paired)
        differences = _masked_mean(paired, paired_valid, axis=0)
        result = {}
        for index, name in enumerate(self.metric_names):
            result[name] = {
                GROUPS[group]: {
                    "mean": _json_array(means[group, index]),
                    "sample_count": counts[group, index].tolist(),
                }
                for group in range(2)
            }
            result[name]["within_sample_hallucinated_minus_nonhallucinated"] = {
                "mean": _json_array(differences[index]),
                "sample_count": paired_valid[:, index].sum(axis=0).tolist(),
            }
        return {"sample_count": len(values), "metrics": result}


def scan_observation(scan, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Join q to the annotation of q+1 and summarize the full scan by sample."""

    rows = np.asarray(scan["route_row_position"], dtype=int)
    response_start = int(np.asarray(scan["response_start"]).item())
    relative = rows + 1 - response_start
    keep = (relative >= 0) & (relative < len(labels))
    if np.any(np.diff(rows) <= 0):
        raise ValueError("cohort scan rows must be ordered and unique")
    bucket_names = tuple(np.asarray(scan["reanchor_bucket_name"]).astype(str))
    if bucket_names != BUCKETS:
        raise ValueError("cohort scan has a different source bucket order")
    transport = np.asarray(scan["reanchor_bucket_transport"], dtype=float)
    score = np.asarray(scan["reanchor_score"], dtype=float)
    if transport.shape != (*score.shape, 4) or score.shape[2] != len(rows):
        raise ValueError("scan transport/score must share [layer,head,row] axes")
    total = transport.sum(axis=-1)
    fractions = np.divide(
        transport,
        total[..., None],
        out=np.full_like(transport, np.nan),
        where=total[..., None] > 0,
    )
    adjacent = np.r_[False, np.diff(rows) == 1]
    # The first response query may have no preceding captured response row.
    # Its zero switch is unknown rather than evidence of a non-event.
    switch_valid = adjacent[None, None, :] & (total > 0)
    switch_valid[..., 1:] &= total[..., :-1] > 0
    switch = np.where(switch_valid, score > 0, np.nan)
    metrics = np.concatenate((fractions, switch[..., None]), axis=-1)
    metrics = metrics[:, :, keep, :].transpose(3, 0, 1, 2)
    row_labels = np.asarray(labels)[relative[keep]]
    grouped = []
    for group in range(2):
        valid = np.isfinite(metrics) & (row_labels == group)[None, None, None, :]
        grouped.append(_masked_mean(metrics, valid, axis=-1))
    coverage = {
        "full_response_tokens": len(labels),
        "annotated_response_tokens": int(np.sum(np.isin(labels, (0, 1)))),
        "scanned_response_tokens": int(keep.sum()),
        "scanned_nonhallucinated_tokens": int(np.sum(row_labels == 0)),
        "scanned_hallucinated_tokens": int(np.sum(row_labels == 1)),
        "unannotated_scanned_tokens": int(np.sum(~np.isin(row_labels, (0, 1)))),
        "full_nonhallucinated_tokens": int(np.sum(labels == 0)),
        "full_hallucinated_tokens": int(np.sum(labels == 1)),
        "full_response_covered": len(relative[keep]) == len(labels),
        "mixed_annotation_sample": bool(np.any(labels == 0) and np.any(labels == 1)),
        "mixed_scanned_sample": bool(
            np.any(row_labels == 0) and np.any(row_labels == 1)
        ),
    }
    return np.stack(grouped), coverage


def _coverage_summary(rows: list[dict]) -> dict:
    count_fields = (
        "full_response_tokens",
        "annotated_response_tokens",
        "scanned_response_tokens",
        "scanned_nonhallucinated_tokens",
        "scanned_hallucinated_tokens",
        "unannotated_scanned_tokens",
        "full_nonhallucinated_tokens",
        "full_hallucinated_tokens",
    )
    result = {name: sum(row[name] for row in rows) for name in count_fields}
    result.update(
        scanned_samples=len(rows),
        full_response_covered_samples=sum(row["full_response_covered"] for row in rows),
        mixed_annotation_samples=sum(row["mixed_annotation_sample"] for row in rows),
        mixed_scanned_samples=sum(row["mixed_scanned_sample"] for row in rows),
    )
    result["response_token_coverage"] = (
        result["scanned_response_tokens"] / result["full_response_tokens"]
        if result["full_response_tokens"]
        else None
    )
    return result


def _functional_observations(output: Path, manifest: dict, target_rows: list[dict]):
    """Read immediate action only at the audited query, not historical rows."""

    labels = {
        (row["sample_id"], row["query_position"]): row["hallucination_label"]
        for row in target_rows
    }
    samples = {}
    for entry in tqdm(
        manifest["audits"].values(), desc="cohort target diagnostics", unit="target"
    ):
        with np.load(output / entry["result"], allow_pickle=False) as artifact:
            sample_id = str(artifact["dataset_sample_id"].item())
            query = int(artifact["query_position"].item())
            label = labels[(sample_id, query)]
            if label not in (0, 1):
                continue
            sample = samples.setdefault(
                sample_id,
                {"task_type": str(artifact["task_type"].item()), "groups": [[], []]},
            )
            slot = int(np.flatnonzero(artifact["route_row_position"] == query)[0])
            action = np.asarray(artifact["reanchor_bucket_downstream_action"], float)
            observation = {"action": action[:, :, slot, :].transpose(2, 0, 1)}
            if "route_stage_position" in artifact:
                stage_slot = int(
                    np.flatnonzero(artifact["route_stage_position"] == query)[0]
                )
                for name in (
                    "route_module_vector_cosine",
                    "route_module_functional_agreement",
                ):
                    if name in artifact:
                        value = np.asarray(artifact[name], float)
                        observation[name] = (
                            value if value.ndim == 1 else value[:, stage_slot]
                        )
            observation["root_value_effect"] = float(
                artifact["root_value_effect"].item()
            )
            sample["groups"][label].append(observation)
    return samples


def _functional_summary(samples: dict) -> dict:
    moments = HeadCohort(BUCKETS)
    profiles = {
        name: [[], []]
        for name in ("route_module_vector_cosine", "route_module_functional_agreement")
    }
    root_effects = [[], []]
    target_counts = [0, 0]
    for sample in samples.values():
        shape = next(
            item["action"].shape for group in sample["groups"] for item in group
        )
        action = np.full((2, *shape), np.nan)
        for group, observations in enumerate(sample["groups"]):
            target_counts[group] += len(observations)
            if observations:
                values = np.stack([item["action"] for item in observations])
                action[group] = _masked_mean(values, np.isfinite(values), axis=0)
            for name in profiles:
                values = [item[name] for item in observations if name in item]
                if values:
                    values = np.stack(values)
                    profiles[name][group].append(
                        _masked_mean(values, np.isfinite(values), axis=0)
                    )
            values = [
                item["root_value_effect"]
                for item in observations
                if np.isfinite(item["root_value_effect"])
            ]
            if values:
                root_effects[group].append(float(np.mean(values)))
        moments.add(action)
    profile_summary = {}
    for name, groups in profiles.items():
        profile_summary[name] = {}
        for group, values in enumerate(groups):
            array = np.asarray(values)
            profile_summary[name][GROUPS[group]] = {
                "sample_count": len(values),
                "mean_by_layer": (
                    _json_array(_masked_mean(array, np.isfinite(array), axis=0))
                    if len(values)
                    else []
                ),
            }
    return {
        "selected_targets": dict(zip(GROUPS, target_counts, strict=True)),
        "bucket_immediate_action": moments.summary(),
        "module_integration": profile_summary,
        "root_value_effect_sample_means": dict(zip(GROUPS, root_effects, strict=True)),
    }


def _draw_head_comparisons(figure, axes, summary: dict, *, signed: bool) -> None:
    from matplotlib import colormaps
    from matplotlib.ticker import MaxNLocator

    columns = (*GROUPS, "within_sample_hallucinated_minus_nonhallucinated")
    titles = (
        "Annotated nonhallucinated",
        "Annotated hallucinated",
        "Within-sample H - N",
    )
    for row, (metric, values) in enumerate(summary["metrics"].items()):
        absolute = np.array([values[group]["mean"] for group in GROUPS], dtype=float)
        absolute_limit = (
            max(float(np.nanmax(np.abs(absolute))), 1e-8)
            if np.any(np.isfinite(absolute))
            else 1.0
        )
        for column, name in enumerate(columns):
            axis = axes[row, column]
            mean = np.array(values[name]["mean"], dtype=float)
            counts = np.asarray(values[name]["sample_count"])
            mask = ~np.isfinite(mean) | (counts < MIN_PLOT_SAMPLES)
            difference = column == 2
            if difference:
                finite = mean[~mask]
                limit = max(float(np.max(np.abs(finite))), 1e-8) if len(finite) else 1
            else:
                limit = absolute_limit if signed else 1
            diverging = signed or difference
            cmap = colormaps["coolwarm" if diverging else "viridis"].copy()
            cmap.set_bad("#dedede")
            artist = axis.imshow(
                np.ma.masked_array(mean, mask),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-limit if diverging else 0,
                vmax=limit,
            )
            axis.set_xlabel("Head")
            axis.set_ylabel("Layer")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_title(f"{metric.replace('_', ' ')}\n{titles[column]}", fontsize=9)
            figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.02)


def plot_cohort_group(output: Path, task: str, group: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    structural = group["structure"]
    if structural["metrics"]:
        figure, axes = plt.subplots(5, 3, figsize=(15, 19), constrained_layout=True)
        _draw_head_comparisons(figure, axes, structural, signed=False)
        coverage = group["coverage"]
        figure.suptitle(
            f"{task}: all scanned rows, each layer/head retained\n"
            f"{coverage['scanned_samples']} samples; "
            f"{coverage['scanned_response_tokens']}/{coverage['full_response_tokens']} "
            "response tokens. Means weight samples equally.\n"
            f"Gray: fewer than {MIN_PLOT_SAMPLES} contributing samples; paired column "
            "requires both labels in the same sample. Descriptive, not a significance test.",
            fontsize=11,
        )
        path = output / f"cohort_{task}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        files.append(path.name)
    functional = group["functional"]
    if functional["bucket_immediate_action"]["metrics"]:
        figure, axes = plt.subplots(5, 3, figsize=(15, 19), constrained_layout=True)
        _draw_head_comparisons(
            figure, axes[:4], functional["bucket_immediate_action"], signed=True
        )
        colors = ("#287bb5", "#c74844")
        for index, (name, profiles) in enumerate(
            functional["module_integration"].items()
        ):
            axis = axes[4, index]
            for group_index, label in enumerate(GROUPS):
                profile = profiles[label]
                if profile["sample_count"] >= MIN_PLOT_SAMPLES:
                    axis.plot(
                        profile["mean_by_layer"], color=colors[group_index], label=label
                    )
            axis.set_title(name.removeprefix("route_").replace("_", " "), fontsize=9)
            axis.set_xlabel("Layer")
            axis.set_ylim(-1.05, 1.05)
            axis.axhline(0, color="gray", linewidth=0.7)
            if axis.lines and len(axis.lines) > 1:
                axis.legend(fontsize=7)
        axis = axes[4, 2]
        for index, label in enumerate(GROUPS):
            values = functional["root_value_effect_sample_means"][label]
            if values:
                ordered = np.sort(values)
                axis.step(
                    np.r_[ordered[0], ordered],
                    np.arange(len(ordered) + 1) / len(ordered),
                    where="post",
                    label=f"{label} n={len(ordered)}",
                    color=colors[index],
                )
        axis.set_title("Fixed-root cut effect: sample mean ECDF", fontsize=9)
        axis.set_xlabel("Observed-token contrast effect")
        axis.set_ylabel("Fraction of samples")
        axis.set_ylim(0, 1.02)
        if axis.lines:
            axis.legend(fontsize=7)
        figure.suptitle(
            f"{task}: selected-target functional audit\n"
            "Signed gradient-message action supports the recorded token contrast; "
            "source-cut module agreement does not establish factual truth.\n"
            f"Heads remain separate; gray/omitted profiles require {MIN_PLOT_SAMPLES} "
            "samples. Event selection can bias these distributions.",
            fontsize=11,
        )
        path = output / f"cohort_functional_{task}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        files.append(path.name)
    return files


def summarize_cohort(
    output: str | Path,
    manifest: dict,
    label_by_sample: dict[str, np.ndarray],
    *,
    target_rows: list[dict],
    plot: bool = False,
) -> dict:
    """Save per-task full-scan comparisons and selected-target diagnostics."""

    output = Path(output)
    observations = {}
    for sample_id, entry in tqdm(
        manifest["samples"].items(), desc="cohort full scans", unit="sample"
    ):
        if "scan" not in entry:
            continue
        with np.load(output / entry["scan"], allow_pickle=False) as scan:
            if str(scan["dataset_sample_id"].item()) != sample_id:
                raise ValueError("cohort scan belongs to a different sample")
            means, coverage = scan_observation(scan, label_by_sample[sample_id])
            observations[sample_id] = {
                "task_type": str(scan["task_type"].item()),
                "means": means,
                "coverage": coverage,
            }
    functional = _functional_observations(output, manifest, target_rows)
    task_names = sorted(
        {item["task_type"] for item in (*observations.values(), *functional.values())}
        | {entry["task_type"] for entry in manifest["selection"]}
    )
    groups = {}
    for task in ("ALL", *task_names):
        selected = {
            key: value
            for key, value in observations.items()
            if task == "ALL" or value["task_type"] == task
        }
        moments = HeadCohort(STRUCTURAL_METRICS)
        for item in selected.values():
            moments.add(item["means"])
        selected_functional = {
            key: value
            for key, value in functional.items()
            if task == "ALL" or value["task_type"] == task
        }
        groups[task] = {
            "selected_samples": sum(
                task == "ALL" or entry.get("task_type") == task
                for entry in manifest["selection"]
            ),
            "coverage": _coverage_summary(
                [item["coverage"] for item in selected.values()]
            ),
            "structure": moments.summary(),
            "functional": _functional_summary(selected_functional),
        }
    report = {
        "cohort_schema": 1,
        "label_use": "Post-hoc only; labels did not select samples, rows or events.",
        "annotation_semantics": (
            "Label 0 means no annotated hallucination at that recorded response token; "
            "it is not an independent factual truth judgment. Labels join query q to token q+1."
        ),
        "structure_scope": "All saved preselection scan rows; no event-selected target union.",
        "source_semantics": (
            "prompt_evidence denotes declared prompt source units; bucket membership "
            "alone does not show which factual constraint the model used."
        ),
        "functional_scope": "Immediate action and source-cut integration at selected audit targets only.",
        "aggregation": (
            "Tokens average within each sample/label for each exact layer/head; "
            "samples then have equal weight. Paired effects use samples containing both groups. "
            "No head averaging and no token-independent significance tests."
        ),
        "limitations": (
            "Comparisons are descriptive associations. Within-sample comparisons do not "
            "control response position, claim type or dependence between samples from one source; "
            "they do not establish causal mediation or factual evidence acceptance."
        ),
        "minimum_plot_samples": MIN_PLOT_SAMPLES,
        "samples_without_full_scan": len(manifest["samples"]) - len(observations),
        "groups": groups,
        "sample_coverage": {
            sample_id: {"task_type": item["task_type"], **item["coverage"]}
            for sample_id, item in observations.items()
        },
        "plots": [],
    }
    if plot:
        for task, group in tqdm(groups.items(), desc="cohort plots", unit="task"):
            report["plots"].extend(plot_cohort_group(output, task, group))
    save_json(output / COHORT_REPORT_NAME, report)
    return report
