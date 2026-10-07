"""Frozen-head evaluation with entity-pair bootstrap; no head selection on test."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


def paired_interval(values, groups, seed=17):
    """Report uncertainty over held-out entity pairs, retaining all owners/orders."""
    group_means = np.array([np.mean(np.asarray(values)[np.asarray(groups) == group])
                            for group in sorted(set(groups))])
    generator = np.random.default_rng(seed)
    samples = generator.choice(group_means, size=(4000, len(group_means)), replace=True)
    return dict(mean=float(group_means.mean()), ci95=np.quantile(samples.mean(axis=1), [.025, .975]).tolist(),
                pairs=len(group_means))


def selected_readouts(cases, readouts):
    summary, rows = {}, []
    for name, result in readouts.items():
        selected = result["selected"]
        summary[name] = dict(layer=selected // 32, head=selected % 32,
                             ridge=result["ridge"], temperature=result["temperature"], styles={})
        for style, metrics in result["evaluations"].items():
            index = [i for i, case in enumerate(cases) if case["split"] == "test" and case["style"] == style]
            logits = result["scores"][index, selected]
            labels = torch.tensor([cases[i]["label"] for i in index]).float()
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            gain = (np.log(2) - losses.numpy()) / np.log(2)
            summary[name]["styles"][style] = dict(accuracy=float(metrics["accuracy"][selected]),
                auc=float(metrics["auc"][selected]), logloss=float(metrics["loss"][selected]),
                content_baseline_bits=paired_interval(gain, [cases[i]["pair_id"] for i in index]))
            for head_index in range(len(metrics["loss"])):
                rows.append(dict(kind=name, style=style, layer=head_index // 32, head=head_index % 32,
                                 selected=head_index == selected, loss=float(metrics["loss"][head_index]),
                                 accuracy=float(metrics["accuracy"][head_index]), auc=float(metrics["auc"][head_index])))
    return summary, rows


def causal_summary(interventions, selected):
    summary = []
    for layer, head in selected:
        rows = [row for row in interventions if (row["layer"], row["head"]) == (layer, head)]
        entry = dict(layer=layer, head=head, controls={})
        for kind in sorted({row["kind"] for row in rows}):
            condition = [row for row in rows if row["kind"] == kind]
            values = [row["signed_effect"] for row in condition]
            entry["controls"][kind] = dict(**paired_interval(values, [row["pair_id"] for row in condition]),
                positive_fraction=float(np.mean(np.asarray(values) > 0)),
                candidate_flips=sum((row["baseline"] > 0) != (row["margin"] > 0) for row in condition))
        owner = {row["case"]: row for row in rows if row["kind"] == "owner"}
        for kind in ("style", "random"):
            comparison = [row for row in rows if row["kind"] == kind]
            differences = [owner[row["case"]]["signed_effect"] - row["signed_effect"] for row in comparison]
            entry[f"owner_minus_{kind}"] = paired_interval(differences, [row["pair_id"] for row in comparison])
        summary.append(entry)
    return summary


def baseline_summary(cases, observations, tokenizer):
    summary = {}
    seven = tokenizer.encode("7", add_special_tokens=False)[0]
    for split in ("fit", "dev", "test"):
        rows = []
        for case, observation in zip(cases, observations):
            if case["split"] != split:
                continue
            other = tokenizer.encode(case["other"], add_special_tokens=False)[0]
            target = seven if case["label"] else other
            logits = observation["logits"][0]
            rows.append((int(logits.argmax()) == target,
                         float(logits.softmax(-1)[[seven, other]].sum()),
                         bool((logits[seven] > logits[other]).item()) == bool(case["label"])))
        summary[split] = dict(cases=len(rows), greedy_digit_accuracy=float(np.mean([row[0] for row in rows])),
                              mean_candidate_mass=float(np.mean([row[1] for row in rows])),
                              pair_choice_accuracy=float(np.mean([row[2] for row in rows])))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = json.loads((args.output / "cases.json").read_text())
    execution = json.loads((args.output / "execution.json").read_text())
    readouts = torch.load(args.output / "readability.pt", weights_only=False)
    observations = torch.load(args.output / "observations.pt", weights_only=False)
    interventions = json.loads((args.output / "interventions.json").read_text())
    selected = json.loads((args.output / "selection.json").read_text())["selected"]
    tokenizer = AutoTokenizer.from_pretrained(execution["model"], local_files_only=True)
    readability, rows = selected_readouts(cases, readouts)
    result = dict(execution=execution, baseline=baseline_summary(cases, observations, tokenizer),
                  readability=readability, causal=causal_summary(interventions, selected),
                  canaries=json.loads((args.output / "canaries.json").read_text()))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "readouts.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
