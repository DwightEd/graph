"""Run content-fixed binding diagnostics on a frozen local Llama observer.

PYTHONPATH=.:teaching/state_audit/src python -m experiments.token_backtrace.binding_run
    --model /path/to/Meta-Llama-3.1-8B-Instruct --output outputs/binding_message_v1
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.decision_risk_flow.run import load_model
from state_audit.model.adapter import ModelAdapter
from .binding import (edge_delta, fit_readability, measure_case, patch_choice,
                      query_gradients)
from .binding_data import build_cases, matched_donor
from .messages import native_trace, patch_message


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def capture_cohort(adapter, cases, candidate, output):
    observations = []
    for index, case in enumerate(cases):
        observation, _ = measure_case(adapter, case, candidate)
        observations.append(observation)
        if (index + 1) % 8 == 0:
            print(f"CAPTURE {index + 1}/{len(cases)}", flush=True)
    torch.save(observations, output / "observations.pt")
    return observations


def cohort_indices(cases, split, style):
    return [i for i, case in enumerate(cases) if case["split"] == split and case["style"] == style]


def measure_readability(cases, observations, output):
    values = torch.stack([row["value"] for row in observations])
    attention = torch.stack([row["attention"] for row in observations])
    edges = values * attention[..., None]
    labels = torch.tensor([case["label"] for case in cases])
    position = torch.tensor([case["key"] / case["query"] for case in cases])
    position = position[:, None, None].expand_as(attention)
    nuisance = torch.stack((position, attention, values.norm(dim=-1).log()), dim=-1)
    fit = cohort_indices(cases, "fit", "owns")
    dev = cohort_indices(cases, "dev", "owns")
    evaluations = {style: cohort_indices(cases, "test", style) for style in ("owns", "has")}
    results = {}
    for name, features in (("value", values), ("edge", edges), ("nuisance", nuisance)):
        results[name] = fit_readability(features, labels, fit, dev, evaluations)
        print(f"READABILITY {name} head={results[name]['selected']}", flush=True)
    torch.save(results, output / "readability.pt")
    return results


def screen_heads(adapter, tokenizer, cases, observations, candidate, output):
    derivatives = []
    for index in cohort_indices(cases, "dev", "owns"):
        case = cases[index]
        other = tokenizer.encode(case["other"], add_special_tokens=False)[0]
        _, trace = measure_case(adapter, case, candidate, gradients=True)
        gradient = query_gradients(trace, other)
        donor = matched_donor(cases, case, change_owner=True)
        delta = edge_delta(observations[index], observations[donor])
        sign = 2 * case["label"] - 1
        derivatives.append(-sign * (gradient * delta).sum(dim=-1))
        print(f"SCREEN {case['id']}", flush=True)
        del trace
    slopes = torch.stack(derivatives)
    ranking = torch.argsort(slopes.mean(dim=0).flatten(), descending=True, stable=True)
    selected = [(int(index // slopes.shape[2]), int(index % slopes.shape[2])) for index in ranking[:4]]
    torch.save(dict(slopes=slopes, selected=selected), output / "screen.pt")
    write_json(output / "selection.json", dict(selected=selected, split="dev", selection="signed edge slope"))
    return selected


def intervention_directions(cases, observations, index, layer, head):
    case = cases[index]
    base = observations[index]
    donor = matched_donor(cases, case, change_owner=True)
    sham = matched_donor(cases, case, change_style=True)
    owner_delta = edge_delta(base, observations[donor])[layer, head]
    style_delta = edge_delta(base, observations[sham])[layer, head]
    generator = torch.Generator().manual_seed(4000 + index * 1024 + layer * 32 + head)
    random = torch.randn(owner_delta.shape, generator=generator)
    random = random / random.norm() * owner_delta.norm()
    return dict(owner=owner_delta, style=style_delta, random=random)


def finite_cohort(adapter, tokenizer, cases, observations, candidate, selected, output):
    rows = []
    for index in cohort_indices(cases, "test", "owns"):
        case = cases[index]
        other = tokenizer.encode(case["other"], add_special_tokens=False)[0]
        logits = observations[index]["logits"][0]
        baseline = float(logits[candidate] - logits[other])
        for layer, head in selected:
            directions = intervention_directions(cases, observations, index, layer, head)
            settings = [(name, delta, 1.) for name, delta in directions.items()]
            if (layer, head) == selected[0]:
                settings += [("owner_half", directions["owner"], .5)]
            for name, delta, alpha in settings:
                result = patch_choice(adapter, case, candidate, other, layer, head, delta, alpha)
                rows.append(dict(case=case["id"], pair_id=case["pair_id"], label=case["label"],
                                 layer=layer, head=head, kind=name, baseline=baseline,
                                 signed_effect=(2 * case["label"] - 1) * (baseline - result["margin"]),
                                 **result))
        print(f"FINITE {case['id']} rows={len(rows)}", flush=True)
        write_json(output / "interventions.json", rows)
    return rows


def scientific_canaries(adapter, cases, observations, candidate, selected):
    index = cohort_indices(cases, "test", "owns")[0]
    case = cases[index]
    layer, head = selected[0]
    delta = torch.zeros_like(observations[index]["value"][layer, head])
    with patch_message(adapter.native, layer, head, case["query"], delta):
        trace = native_trace(adapter.native, case["prompt"], [candidate], gradients=False)
    identity_error = (trace.logits.cpu() - observations[index]["logits"]).abs().max().item()
    layer0_differences = []
    for i, row in enumerate(cases):
        donor = matched_donor(cases, row, change_owner=True)
        difference = (observations[i]["value"][0] - observations[donor]["value"][0]).abs().max()
        layer0_differences.append(float(difference))
    return dict(identity_replay_error=identity_error,
                identity_top_same=int(observations[index]["logits"][0].argmax()) == int(trace.logits[0].argmax()),
                layer0_owner_difference=max(layer0_differences),
                reconstruction_max=max(row["reconstruction"] for row in observations))


def protocol_hashes(output):
    sources = [Path(__file__), Path(__file__).with_name("binding.py"),
               Path(__file__).with_name("binding_data.py"), output / "cases.json"]
    write_json(output / "input_hashes.json", {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="2 fit / 1 dev / 2 test entity pairs")
    parser.add_argument("--reverse-participants", action="store_true", help="Balance both query owners for each entity name")
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    counts = (2, 1, 2) if args.pilot else (12, 4, 8)
    cases = build_cases(tokenizer, counts, args.reverse_participants)
    write_json(args.output / "cases.json", cases)
    protocol_hashes(args.output)
    candidate = tokenizer.encode("7", add_special_tokens=False)[0]
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(args.model))
    observations = capture_cohort(adapter, cases, candidate, args.output)
    measure_readability(cases, observations, args.output)
    selected = screen_heads(adapter, tokenizer, cases, observations, candidate, args.output)
    canaries = scientific_canaries(adapter, cases, observations, candidate, selected)
    write_json(args.output / "canaries.json", canaries)
    assert canaries["identity_replay_error"] < 1e-5 and canaries["identity_top_same"]
    assert canaries["layer0_owner_difference"] == 0 and canaries["reconstruction_max"] < 1e-4
    rows = finite_cohort(adapter, tokenizer, cases, observations, candidate, selected, args.output)
    write_json(args.output / "execution.json", dict(status="DONE", cases=len(cases),
               finite_forwards=len(rows) + 1, gradient_forwards=len(cohort_indices(cases, "dev", "owns")),
               capture_forwards=len(cases), seconds=time.time() - started,
               peak_memory=torch.cuda.max_memory_allocated(), model=args.model,
               auxiliary_owner_labels=True, hallucination_labels_used=False))
    print(f"DONE seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
