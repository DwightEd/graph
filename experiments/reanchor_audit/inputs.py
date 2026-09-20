"""Reuse paired contexts, including archives without original attention NPZs."""

import json

import numpy as np
import pandas as pd


def saved_cases(paired_input):
    cases = []
    for path in sorted((paired_input / "pairs").glob("*/reviewed_case.json")):
        case = json.loads(path.read_text())
        for side in ("supported", "unsupported"):
            context_path = path.parent / side / "onset" / "context.json"
            context = json.loads(context_path.read_text())
            cases.append(dict(case=case, side=side, context=context,
                              prompt_length=len(context["prefix_ids"]) - context["position"]))
    return cases


def inventory(paired_input, output):
    cases = saved_cases(paired_input)
    rows = []
    for item in cases:
        case, context = item["case"], item["context"]
        lengths = [len(tokens) for tokens in context["candidates"]]
        rows.append(dict(case_id=case["case_id"], source_id=case["source_id"], side=item["side"],
            prefix_tokens=len(context["prefix_ids"]), prompt_tokens=item["prompt_length"],
            decision_position=context["position"], first_candidate_tokens=lengths[0],
            second_candidate_tokens=lengths[1], history_status=case["history_status"][item["side"]],
            readout_status="different_attribute_slots" if case["case_id"] == "14375_onion_stage"
                else "natural_prefix_and_wording_confounded",
            source_roles="reviewed_local_support", native_graph_measured=False))
    pd.DataFrame(rows).to_csv(output / "inventory.csv", index=False)
    return cases


def compile_probe(item, tokenizer):
    from ..path_conflict.paired_inputs import reviewed_roles

    ids = item["context"]["prefix_ids"]
    roles = reviewed_roles(tokenizer, ids[:item["prompt_length"]], item["case"])
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if np.intersect1d(roles[left], roles[right]).size:
                raise ValueError(f"Reviewed roles overlap at token boundaries: {left}, {right}")
    probe = dict(prefix_ids=ids, candidates=item["context"]["candidates"],
        prompt_length=item["prompt_length"], groups=roles, record_writes=False,
        phase="onset", readout="sequence_margin",
        actual_token=item["context"]["candidates"][int(item["side"] == "unsupported")][0])
    text = [tokenizer.decode([token]) for token in ids]
    return probe, roles, text


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Audit settings changed: {path}; use a new output directory")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def model_path(args):
    if args.model:
        return args.model
    config = json.loads((args.paired_input / "paired_config.json").read_text())
    return config["sampling"]["model"]
