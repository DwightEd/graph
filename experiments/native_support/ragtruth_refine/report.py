"""Task/generator test evaluation and exact AUC accounting for anchor ties."""

from collections import defaultdict

import numpy as np
from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from ..evaluate import annotation_targets
from ..ragtruth_benchmark.report import group_metrics, pair_counts
from .data import read_annotations, require_scores
from .scoring import ANCHORS, CHANNELS, METHODS, PRIMARY


def evaluation_rows(args, reader, manifest, records, methods, choices):
    truth = read_annotations(reader, manifest, records)
    result = []
    for record in tqdm(records, desc=f"test {records[0]['task']}"):
        saved = read_arrays(args.output / record["directory"] / "scores.npz")
        if choices:
            saved["refined_detector"] = saved[choices[record["task"]]["method"]]
        annotation = truth[record["id"]]
        if not np.array_equal(saved["token_id"], annotation["token_ids"]):
            raise ValueError(f"{record['id']}: evaluation annotation token alignment differs")
        if str(annotation["source_id"]) != str(record["source_id"]):
            raise ValueError(f"{record['id']}: annotation source identity differs")
        labels, onsets, firsts, valid = annotation_targets(annotation, record["tokens"], record["id"])
        matrix = np.column_stack([saved[name] for name in methods])
        if not np.isfinite(matrix).all():
            raise ValueError(f"{record['id']}: nonfinite score")
        same_answer, answer_pairs = pair_counts(labels[valid], matrix[valid])
        same_unit, unit_pairs, mixed = np.zeros(len(methods)), 0, 0
        response = reader.json(record["directory"] + "/response.json")
        for unit in response["units"]:
            positions = np.arange(unit["start"], unit["stop"])
            positions = positions[valid[positions]]
            concordant, pairs = pair_counts(labels[positions], matrix[positions])
            same_unit += concordant
            unit_pairs += pairs
            mixed += int(pairs > 0)
        result.append(dict(**record, labels=labels[valid], onsets=onsets[valid], firsts=firsts[valid],
            target=np.flatnonzero(valid), scores=matrix[valid], same_answer=same_answer,
            answer_pairs=answer_pairs, same_unit=same_unit, unit_pairs=unit_pairs, mixed_units=mixed))
    return result


def tie_accounting(records, methods):
    """AUC changes only inside equal-anchor blocks; different units can share a block."""
    labels = np.concatenate([row["labels"] for row in records])
    matrix = np.concatenate([row["scores"] for row in records])
    total_pairs = int(labels.sum()) * int((labels == 0).sum())
    result = {}
    for anchor in ANCHORS:
        base = matrix[:, methods.index(f"source_{anchor}_unit_mean")]
        names = [f"{anchor}_{channel}_tie" for channel in CHANNELS]
        columns = [methods.index(name) for name in names]
        order = np.argsort(base, kind="stable")
        boundaries = np.r_[0, np.flatnonzero(np.diff(base[order])) + 1, len(order)]
        concordant, tied_pairs = np.zeros(len(names)), 0
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            positions = order[start:stop]
            count, pairs = pair_counts(labels[positions], matrix[positions][:, columns])
            concordant += count
            tied_pairs += pairs
        for index, name in enumerate(names):
            result[name] = dict(anchor=f"source_{anchor}_unit_mean", tied_positive_negative_pairs=tied_pairs,
                all_positive_negative_pairs=total_pairs,
                tied_pair_fraction=tied_pairs / total_pairs if total_pairs else None,
                tie_block_auroc=float(concordant[index] / tied_pairs) if tied_pairs else None,
                predicted_auroc_gain=float((concordant[index] - .5 * tied_pairs) / total_pairs) if total_pairs else None)
    return result


def metric_table(groups):
    rows = []
    for group in groups:
        for method, phases in group["methods"].items():
            for phase, values in phases.items():
                rows.append(dict(task=group["task"], split="test", generator=group["generator"],
                    method=method, phase=phase,
                    **{key: values[key] for key in ("tokens", "positives", "prevalence", "auroc", "ap")},
                    within_answer_auroc=values.get("within_answer_auroc"),
                    within_unit_auroc=values.get("within_unit_auroc")))
    return rows


def plot(output, rows, headline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = sorted({row["task"] for row in rows})
    methods = ["source_local_unit_mean", "source_pair_unit_mean", "source_full_unit_mean", headline]
    lookup = {(row["task"], row["method"]): row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, ("auroc", "ap", "within_unit_auroc")):
        for index, method in enumerate(methods):
            values = [lookup[task, method][metric] for task in tasks]
            axis.bar(np.arange(len(tasks)) + (index - 1.5) * .2,
                     [np.nan if value is None else value for value in values], .2, label=method)
        axis.set(xticks=np.arange(len(tasks)), xticklabels=tasks, ylabel=metric, ylim=(0, 1))
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / "datasets.png", dpi=150)
    plt.close(figure)


def evaluate(args, reader, manifest):
    require_scores(args.output, manifest)
    selection = read_json(args.output / "selection.json") if args.select_on_train else None
    choices = selection["choices"] if selection else {}
    methods = list(METHODS)
    if manifest["previous_selected"]:
        methods.append("previous_selected")
    if choices:
        methods.append("refined_detector")
    groups, diagnostics = [], {}
    for task in args.tasks:
        selected = [row for row in manifest["records"] if row["task"] == task and row["split"] == "test"]
        records = evaluation_rows(args, reader, manifest, selected, methods, choices)
        diagnostics[task] = tie_accounting(records, methods)
        grouped = defaultdict(list)
        for row in records:
            grouped[row["generator"]].append(row)
        grouped["ALL"] = records
        for generator, rows in tqdm(sorted(grouped.items()), desc=f"metrics {task}"):
            groups.append(dict(task=task, split="test", generator=generator, **group_metrics(rows, methods)))
    table = metric_table(groups)
    datasets = [row for row in table if row["generator"] == "ALL" and row["phase"] == "all_error"]
    headline = "refined_detector" if choices else PRIMARY
    reported = {"source_local_unit_mean", "source_full_unit_mean", "source_pair_unit_mean", PRIMARY, headline, "previous_selected"}
    summary = dict(status="evaluated", primary_candidate=headline, fixed_candidate=PRIMARY,
        selection=selection, coverage=read_json(args.output / "coverage.json"),
        test_by_dataset=[row for row in datasets if row["method"] in reported],
        test_status="exploratory_re_evaluation_after_prior_test_analysis")
    write_json(args.output / "evaluation.json", dict(groups=groups))
    write_json(args.output / "tie_accounting.json", diagnostics)
    write_json(args.output / "summary.json", summary)
    write_csv(args.output / "metrics.csv", table, list(table[0]))
    write_csv(args.output / "metrics_by_dataset.csv", datasets, list(datasets[0]))
    plot(args.output, datasets, headline)
    return summary
