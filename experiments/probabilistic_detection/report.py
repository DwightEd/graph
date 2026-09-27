"""Export frozen results as readable tables and standalone scientific figures."""

import argparse
import csv
import json
from pathlib import Path


TASKS = ("QA", "Summary", "Data2txt")
METHODS = ("source_refine", "context_prior", "selected_gaussian", "logistic", "logistic_matched",
           "linear", "squares", "interactions", "conditioned", "trees", "selected_detector")
FIELDS = ("auroc", "ap", "within_answer_auroc", "within_unit_auroc", "token_fpr",
          "token_recall", "normal_answer_false_alarm")


def read(path):
    return json.loads(path.read_text())


def format_number(value):
    return "undefined" if value is None else f"{value:.4f}"


def export(run, output):
    output.mkdir(parents=True, exist_ok=True)
    results = {task: read(run / task / "test_metrics.json") for task in TASKS}
    rows = [dict(task=task, method=method, **{field: results[task][method][field] for field in FIELDS})
            for task in TASKS for method in METHODS]
    with (output / "test_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task", "method", *FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Frozen official-test exploratory revalidation", "",
        "All predictions and development thresholds were frozen before this evaluation. "
        "The official test was exposed in earlier project research; these are not fresh confirmatory results.", "",
        "Three tasks, 450 sources, 2,700 answers, 424,408 valid tokens. "
        "Supervised source-disjoint training; no new LLM forward passes.", ""]
    for task in TASKS:
        selection = read(run / task / "detector_selection.json")
        lines.extend([f"## {task}", "", f"Development-selected detector: `{selection['selected']}`.", "",
            "| Method | AUROC | AP | Within answer | Within unit | Token FPR | Token recall | Normal-answer any-alarm |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for method in METHODS:
            values = [format_number(results[task][method][field]) for field in FIELDS]
            lines.append("| " + " | ".join([method, *values]) + " |")
        lines.extend(["", "Paired source bootstrap (300 draws; exploratory, unadjusted comparisons):", "",
                      "| Candidate − baseline | AUROC difference | 95% interval |", "|---|---:|---|"])
        intervals = read(run / task / "bootstrap.json")
        for pair, value in intervals.items():
            candidate, baseline = pair.split("__")
            difference = results[task][candidate]["auroc"] - results[task][baseline]["auroc"]
            interval = value["auroc_delta_95ci"]
            lines.append(f"| {candidate} − {baseline} | {difference:+.5f} | [{interval[0]:+.5f}, {interval[1]:+.5f}] |")
        lines.append("")
    (output / "RESULT_TABLES.md").write_text("\n".join(lines) + "\n")
    draw(results, output)


def draw(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ("source_refine", "linear", "conditioned", "trees")
    colors = ("#7c8288", "#40a7a2", "#075d63", "#b97139")
    labels = ("Previous source refinement", "Linear readout", "Conditional interactions", "Same-input HGB")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for axis, metric, title in zip(axes, ("auroc", "ap"), ("Token AUROC", "Average precision")):
        for index, (method, color, label) in enumerate(zip(methods, colors, labels)):
            x = np.arange(3) + (index - 1.5) * .12
            values = [results[task][method][metric] for task in TASKS]
            axis.scatter(x, values, color=color, marker=("o", "s", "D", "^")[index], s=45, label=label)
        axis.set_xticks(np.arange(3), TASKS)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle("RAGTruth: source-disjoint supervised detection (exploratory test)", fontsize=11)
    figure.savefig(output / "test_comparison.svg")
    figure.savefig(output / "test_comparison.png", dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)


if __name__ == "__main__":
    main()
