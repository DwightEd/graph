"""Channel ablations and fixed-source query swaps on frozen binding observations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.decision_risk_flow.run import load_model
from state_audit.model.adapter import ModelAdapter
from .binding import fit_readability, measure_case
from .binding_analyze import selected_readouts
from .binding_run import cohort_indices, write_json


def query_donor(tokenizer, case):
    """Change only the queried name; all source tokens precede and stay identical."""
    donor = deepcopy(case)
    donor["content"] = case["content"].replace(f"does {case['pair'][0]} have?",
                                              f"does {case['pair'][1]} have?")
    donor["prompt"] = tokenizer.apply_chat_template(
        [{"role": "user", "content": donor["content"]}], add_generation_prompt=True)
    donor["query"] = len(donor["prompt"]) - 1
    donor["id"] += "_query_second"
    donor["label"] = 1 - case["label"]
    assert donor["prompt"][:case["key"] + 1] == case["prompt"][:case["key"] + 1]
    assert donor["query"] == case["query"]
    return donor


def channel_features(cases, observations):
    values = torch.stack([row["value"] for row in observations])
    attention = torch.stack([row["attention"] for row in observations])
    norms = values.norm(dim=-1)
    position = torch.tensor([case["key"] / case["query"] for case in cases])
    position = position[:, None, None].expand_as(norms)
    return dict(direction=values / norms[..., None],
                norm_position=torch.stack((norms.log(), position), dim=-1),
                attention=attention[..., None])


def channel_readouts(cases, features):
    labels = torch.tensor([case["label"] for case in cases])
    fit = cohort_indices(cases, "fit", "owns")
    dev = cohort_indices(cases, "dev", "owns")
    tests = {style: cohort_indices(cases, "test", style) for style in ("owns", "has")}
    results = {}
    for name, channel in features.items():
        results[name] = fit_readability(channel, labels, fit, dev, tests)
        print(f"CONTROL READOUT {name}", flush=True)
    return results


def capture_query_controls(adapter, tokenizer, cases, observations, candidate, output):
    donors, donor_observations = [], []
    max_value_change = 0.
    for index, (case, original) in enumerate(zip(cases, observations)):
        donor = query_donor(tokenizer, case)
        observed, _ = measure_case(adapter, donor, candidate)
        max_value_change = max(max_value_change, (observed["value"] - original["value"]).abs().max().item())
        donors.append(donor)
        donor_observations.append(observed)
        if (index + 1) % 16 == 0:
            print(f"QUERY CONTROL {index + 1}/{len(cases)}", flush=True)
    assert max_value_change == 0
    write_json(output / "query_cases.json", donors)
    torch.save(donor_observations, output / "query_observations.pt")
    write_json(output / "query_canaries.json", dict(source_value_difference=max_value_change,
        cases=len(donors), reconstruction_max=max(row["reconstruction"] for row in donor_observations)))
    return donors, donor_observations


def fit_query_controls(cases, observations, donors, captured, output):
    joined_cases, joined_observations = cases + donors, observations + captured
    # Paired identical V with complementary labels implies no V-only decoder can beat 50%.
    attention = torch.stack([row["attention"] for row in joined_observations])
    values = torch.stack([row["value"] for row in joined_observations])
    features = dict(attention=attention[..., None], edge=values * attention[..., None])
    readouts = channel_readouts(joined_cases, features)
    torch.save(readouts, output / "query_readability.pt")
    summary, _ = selected_readouts(joined_cases, readouts)
    write_json(output / "query_summary.json", dict(value_only_accuracy_bound=.5, **summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-controls", action="store_true")
    parser.add_argument("--cached-query-controls", action="store_true", help="Fit only already captured query pairs")
    args = parser.parse_args()
    cases = json.loads((args.output / "cases.json").read_text())
    observations = torch.load(args.output / "observations.pt", weights_only=False)
    if args.cached_query_controls:
        donors = json.loads((args.output / "query_cases.json").read_text())
        captured = torch.load(args.output / "query_observations.pt", weights_only=False)
        fit_query_controls(cases, observations, donors, captured, args.output)
        print("CACHED QUERY CONTROLS DONE", flush=True)
        return
    features = channel_features(cases, observations)
    readouts = channel_readouts(cases, features)
    torch.save(readouts, args.output / "channel_readability.pt")
    summary, _ = selected_readouts(cases, readouts)
    write_json(args.output / "channel_summary.json", summary)
    if args.query_controls:
        execution = json.loads((args.output / "execution.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(execution["model"], local_files_only=True)
        adapter = ModelAdapter(load_model(execution["model"]))
        candidate = tokenizer.encode("7", add_special_tokens=False)[0]
        donors, captured = capture_query_controls(adapter, tokenizer, cases, observations, candidate, args.output)
        fit_query_controls(cases, observations, donors, captured, args.output)
    print("CONTROLS DONE", flush=True)


if __name__ == "__main__":
    main()
