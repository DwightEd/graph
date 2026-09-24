"""Token metrics by task, generator and official split; no test-time model selection."""

from collections import defaultdict
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.stats import rankdata
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from ..comparison_evaluation import phase_masks
from ..evaluate import annotation_targets, ranking
from .data import annotations
from .scoring import METHODS, PRIMARY


def pair_counts(labels, matrix):
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    pairs = positives * negatives
    if pairs == 0:
        return np.zeros(matrix.shape[1]), 0
    ranks = rankdata(matrix, axis=0, method="average")
    concordant = ranks[labels == 1].sum(0) - positives * (positives + 1) / 2
    return concordant, pairs


def load_records(output, manifest, methods):
    truth = annotations(output, manifest, manifest["records"])
    records = []
    for item in manifest["records"]:
        directory = output / item["directory"]
        saved, response = read_arrays(directory / "scores.npz"), read_json(directory / "response.json")
        if "selected_detector" in methods:
            saved.update(read_arrays(directory / "selected_scores.npz"))
        annotation = truth[item["id"]]
        if str(annotation["source_id"]) != str(item["source_id"]):
            raise ValueError(f"{item['id']}: annotation source identity differs")
        if not np.array_equal(saved["token_id"], annotation["token_ids"]):
            raise ValueError(f"{item['id']}: annotation and scored token identities differ")
        labels, onsets, firsts, valid = annotation_targets(annotation, item["tokens"], item["id"])
        matrix = np.column_stack([saved[name] for name in methods])
        if not np.isfinite(matrix).all():
            raise ValueError(f"{item['id']}: nonfinite scores cannot be silently excluded")
        same_answer, answer_pairs = pair_counts(labels[valid], matrix[valid])
        same_unit, unit_pairs, mixed = np.zeros(len(methods)), 0, 0
        for unit in response["units"]:
            positions = np.arange(unit["start"], unit["stop"])
            positions = positions[valid[positions]]
            concordant, pairs = pair_counts(labels[positions], matrix[positions])
            same_unit += concordant
            unit_pairs += pairs
            mixed += int(pairs > 0)
        records.append(dict(**item, labels=labels[valid], onsets=onsets[valid], firsts=firsts[valid],
            target=np.flatnonzero(valid), scores=matrix[valid], same_answer=same_answer,
            answer_pairs=answer_pairs, same_unit=same_unit, unit_pairs=unit_pairs, mixed_units=mixed))
    return records


def group_metrics(records, methods):
    joined = {name: np.concatenate([row[name] for row in records])
              for name in ("labels", "onsets", "firsts", "scores", "target")}
    lengths = np.concatenate([np.full(len(row["labels"]), row["tokens"]) for row in records])
    phases = phase_masks(joined["labels"], joined["onsets"], joined["firsts"])
    phases.update(front_half=(joined["target"] < lengths / 2, joined["labels"]),
                  back_half=(joined["target"] >= lengths / 2, joined["labels"]))
    answer_pairs = sum(row["answer_pairs"] for row in records)
    unit_pairs = sum(row["unit_pairs"] for row in records)
    same_answer = sum(row["same_answer"] for row in records)
    same_unit = sum(row["same_unit"] for row in records)
    measured = {}
    for column, name in enumerate(methods):
        measured[name] = {phase: ranking(target[mask], joined["scores"][mask, column])
                          for phase, (mask, target) in phases.items()}
        measured[name]["all_error"].update(
            within_answer_auroc=float(same_answer[column] / answer_pairs) if answer_pairs else None,
            within_unit_auroc=float(same_unit[column] / unit_pairs) if unit_pairs else None)
    return dict(answers=len(records), sources=len({row["source_id"] for row in records}),
                mixed_units=sum(row["mixed_units"] for row in records), methods=measured)


def evaluate(args, manifest):
    coverage = read_json(args.output / "coverage.json")
    if coverage["status"] != "complete":
        raise ValueError("Freeze every selected answer before reading evaluation labels")
    selection_path = args.output / "selection.json"
    methods = (*METHODS, "selected_detector") if selection_path.exists() else METHODS
    records = load_records(args.output, manifest, methods)
    groups = defaultdict(list)
    for row in records:
        for task, generator in ((row["task"], row["generator"]), (row["task"], "ALL"), ("ALL", "ALL")):
            groups[task, row["split"], generator].append(row)
    result, table = [], []
    for (task, split, generator), rows in sorted(groups.items()):
        metadata = dict(task=task, split=split, generator=generator)
        measured = group_metrics(rows, methods)
        result.append(dict(**metadata, **measured))
        for method, phases in measured["methods"].items():
            for phase, values in phases.items():
                table.append(dict(**metadata, method=method, phase=phase,
                    **{key: values[key] for key in ("tokens", "positives", "prevalence", "auroc", "ap")},
                    within_answer_auroc=values.get("within_answer_auroc"),
                    within_unit_auroc=values.get("within_unit_auroc")))
    write_json(args.output / "evaluation.json", dict(status="evaluated", primary_candidate=PRIMARY, groups=result))
    write_csv(args.output / "metrics.csv", table, list(table[0]))
    datasets = [row for row in table if row["task"] != "ALL" and row["generator"] == "ALL" and row["phase"] == "all_error"]
    write_csv(args.output / "metrics_by_dataset.csv", datasets, list(table[0]))
    headline = [row for row in datasets if row["split"] == "test" and row["method"] in (PRIMARY, "selected_detector")]
    summary = dict(status="evaluated", primary_candidate=PRIMARY, coverage=coverage,
        test_by_dataset=headline, selection=read_json(selection_path) if selection_path.exists() else None,
        comparison="pooled token metrics within task/generator/split; no averaged unit AUROC")
    write_json(args.output / "summary.json", summary)
    plot(args.output, datasets)
    return summary


def plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["split"] == "test"]
    tasks = sorted({row["task"] for row in selected})
    if not tasks:
        return
    methods = [PRIMARY, "raw_route", "local_route_0.1"]
    if any(row["method"] == "selected_detector" for row in selected):
        methods.append("selected_detector")
    lookup = {(row["task"], row["method"]): row for row in selected}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    width = .8 / len(methods)
    for axis, metric in zip(axes, ("auroc", "ap")):
        for index, method in enumerate(methods):
            values = [lookup[task, method][metric] for task in tasks]
            axis.bar(np.arange(len(tasks)) + (index - (len(methods) - 1) / 2) * width,
                     [np.nan if v is None else v for v in values], width, label=method)
        axis.set(xticks=np.arange(len(tasks)), xticklabels=tasks, ylabel=metric.upper(), ylim=(0, 1))
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "datasets.png", dpi=150)
    plt.close(figure)


def pack(output):
    """Compact aggregate review; original per-answer NPZ and head captures stay on disk."""
    archive = output.with_name(output.name + "_review_light.zip")
    temporary = archive.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file() and ".partial." not in path.name:
                bundle.write(path, path.name)
    temporary.replace(archive)
    return str(archive)
