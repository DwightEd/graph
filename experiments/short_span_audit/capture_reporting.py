"""Descriptive paired source statistics; selected targets are not a detector test."""

import argparse
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from state_audit.storage import read_json, write_arrays, write_csv, write_json

SOURCE_METRICS = (
    "route_mass",
    "contribution_positive",
    "contribution_negative",
    "value_energy",
)
READOUT_METRICS = ("logit_entropy", "target_logp", "margin")
GROUP_FIELDS = ("task", "generator", "split", "normal_history", "phase")
NORMAL_HISTORY_STEPS = 15


def answer_directory(output, entry):
    record = entry["record"]
    return output / "answers" / record["split"] / record["task"] / str(record["id"])


def load_targets(directory, entry):
    """Open only the compact summaries, leaving large native edge arrays on disk."""
    values, readouts = {}, []
    record = entry["record"]
    for target in sorted({item["target"] for item in entry["targets"]}):
        path = directory / f"{target:06d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            current = {
                "sources": np.stack(
                    [saved["source_" + name] for name in SOURCE_METRICS]
                ),
                "readouts": np.asarray(
                    [saved[name].item() for name in READOUT_METRICS]
                ),
                "layers": saved["layers"],
            }
            scalars = {name: float(saved[name]) for name in READOUT_METRICS}
        invalid = [
            name
            for name in ("sources", "readouts")
            if not np.isfinite(current[name]).all()
        ]
        if invalid:
            raise FloatingPointError(
                f"{record['id']} target {target}: nonfinite saved {invalid}"
            )
        values[target] = current
        start, end = entry["offsets"][target]
        readouts.append(
            dict(
                id=record["id"],
                source_id=record["source_id"],
                task=record["task"],
                generator=record["generator"],
                split=record["split"],
                target=target,
                token=entry["text"][start:end],
                **scalars,
            )
        )
    return values, readouts


def pair_memberships(entry):
    pairs = defaultdict(lambda: {"error": [], "normal": []})
    for membership in entry["targets"]:
        if membership["pair_id"] is not None:
            pairs[str(membership["pair_id"])][membership["side"]].append(membership)
    return pairs


def phase_measurement(memberships, values, phase):
    selected = [item for item in memberships if phase == "span" or item["offset"] == 0]
    if not selected or any(item["target"] not in values for item in selected):
        return None
    return {
        name: np.mean([values[item["target"]][name] for item in selected], axis=0)
        for name in ("sources", "readouts")
    }


def normal_history(gold_spans, start):
    """A normal interval with recent labelled errors is a recovery control."""
    history_start = max(0, start - NORMAL_HISTORY_STEPS)
    recent_error = any(
        left < start and right > history_start for left, right in gold_spans
    )
    return "recovery" if recent_error else "clean_history"


def measured_pairs(entry, values):
    rows = []
    for pair_id, sides in pair_memberships(entry).items():
        for phase in ("onset", "span"):
            error = phase_measurement(sides["error"], values, phase)
            normal = phase_measurement(sides["normal"], values, phase)
            if error is None or normal is None:
                continue
            metadata = {
                name: entry["record"][name]
                for name in ("id", "source_id", "task", "generator", "split")
            }
            rows.append(
                dict(
                    **metadata,
                    pair_id=pair_id,
                    phase=phase,
                    error=error,
                    normal=normal,
                    normal_history=normal_history(
                        entry["gold"], sides["normal"][0]["span_start"]
                    ),
                )
            )
    return rows


def source_means(pairs, measure):
    """Average pairs within each source first; heads never become replicate samples."""
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["source_id"])].append(pair)
    means = {}
    for side in ("error", "normal"):
        sources = [
            np.mean([pair[side][measure] for pair in group], axis=0)
            for group in grouped.values()
        ]
        means[side] = np.mean(sources, axis=0)
    return means, len(grouped)


def grouped_pairs(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[tuple(pair[name] for name in GROUP_FIELDS)].append(pair)
    return groups


def source_rows(pairs, layers, names):
    rows = []
    for group, selected in grouped_pairs(pairs).items():
        means, count = source_means(selected, "sources")
        base = dict(zip(GROUP_FIELDS, group), pairs=len(selected), sources=count)
        for metric, layer, head, source in np.ndindex(means["error"].shape):
            error = float(means["error"][metric, layer, head, source])
            normal = float(means["normal"][metric, layer, head, source])
            rows.append(
                dict(
                    **base,
                    metric=SOURCE_METRICS[metric],
                    layer=int(layers[layer]),
                    head=head,
                    source_group=names[source],
                    mean_error=error,
                    mean_normal=normal,
                    mean_difference=error - normal,
                )
            )
    return rows


def readout_rows(pairs):
    rows = []
    for group, selected in grouped_pairs(pairs).items():
        means, count = source_means(selected, "readouts")
        base = dict(zip(GROUP_FIELDS, group), pairs=len(selected), sources=count)
        for index, metric in enumerate(READOUT_METRICS):
            error, normal = float(means["error"][index]), float(means["normal"][index])
            rows.append(
                dict(
                    **base,
                    metric=metric,
                    mean_error=error,
                    mean_normal=normal,
                    mean_difference=error - normal,
                )
            )
    return rows


def save_pair_arrays(output, pairs, layers, names):
    metadata = [
        {key: value for key, value in pair.items() if key not in ("error", "normal")}
        for pair in pairs
    ]
    write_json(output / "paired_records.json", metadata)
    arrays = {
        "layers": np.asarray(layers),
        "source_names": np.asarray(names),
        "source_metrics": np.asarray(SOURCE_METRICS),
        "readout_metrics": np.asarray(READOUT_METRICS),
    }
    for side in ("error", "normal"):
        for measure in ("sources", "readouts"):
            arrays[f"{side}_{measure}"] = np.asarray(
                [pair[side][measure] for pair in pairs]
            )
    write_arrays(output / "paired_measurements.npz", **arrays)


def write_tables(output, tokens, pairs, layers, names):
    token_fields = [
        "id",
        "source_id",
        "task",
        "generator",
        "split",
        "target",
        "token",
        *READOUT_METRICS,
    ]
    write_csv(output / "token_readouts.csv", tokens, token_fields)
    common = [*GROUP_FIELDS, "pairs", "sources", "metric"]
    values = ["mean_error", "mean_normal", "mean_difference"]
    write_csv(
        output / "paired_head_sources.csv",
        source_rows(pairs, layers, names),
        common + ["layer", "head", "source_group"] + values,
    )
    write_csv(output / "paired_readouts.csv", readout_rows(pairs), common + values)
    save_pair_arrays(output, pairs, layers, names)


def create_archive(output, destination):
    """Review summaries and exact target identities, without the large edge arrays."""
    filenames = (
        "settings.json",
        "report.json",
        "token_readouts.csv",
        "paired_head_sources.csv",
        "paired_readouts.csv",
        "paired_measurements.npz",
        "paired_records.json",
    )
    paths = [output / name for name in filenames]
    paths.extend(sorted((output / "answers").rglob("answer.json")))
    temporary = destination.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(output))
    temporary.replace(destination)


def write_report(output, cohort):
    settings = read_json(output / "settings.json")
    tokens, pairs, complete_answers = [], [], 0
    for entry in cohort:
        values, readouts = load_targets(answer_directory(output, entry), entry)
        tokens.extend(readouts)
        pairs.extend(measured_pairs(entry, values))
        expected = len({item["target"] for item in entry["targets"]})
        complete_answers += int(expected > 0 and len(values) == expected)
    write_tables(output, tokens, pairs, settings["layers"], settings["source_names"])
    archive = output.parent / "contributions_review.tar.gz"
    summary = {
        "purpose": settings["purpose"],
        "expected_answers": sum(bool(e["targets"]) for e in cohort),
        "completed_answers": complete_answers,
        "expected_targets": sum(
            len({t["target"] for t in e["targets"]}) for e in cohort
        ),
        "completed_targets": len(tokens),
        "completed_pair_phases": len(pairs),
        "observer": settings["model"],
        "source_roles": "prompt_history_only_not_reviewed_evidence",
        "statistical_status": "exploratory_paired_source_means_without_multiple_comparison_testing",
        "future_tokens_used": False,
        "natural_labels_used_for_target_selection": True,
        "detector_evaluation": False,
        "normal_history_steps": NORMAL_HISTORY_STEPS,
        "archive": str(archive),
    }
    write_json(output / "report.json", summary)
    create_archive(output, archive)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(
        write_report(
            args.audit / "contributions", read_json(args.audit / "cohort.json")
        )
    )


if __name__ == "__main__":
    main()
