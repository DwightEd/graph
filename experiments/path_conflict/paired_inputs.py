"""Reviewed local claims in saved, same-question resampled answers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import compile_side, quote_tokens, read_run, token_text_offsets, validate_text_cases


PHASES = ("before_claim", "onset", "back_half", "post_claim")
SOURCE_GROUPS = ("evidence", "inapplicable_source", "history", "query_self",
                 "prior_history", "claim_history", "head_total")


def inventory_pairs(directory, cases, output):
    settings, samples, prompts = read_run(directory)
    validate_text_cases(samples, prompts, cases)
    lookup = {(str(row["source_id"]), row["seed"]): row for row in samples}
    if len(lookup) != len(samples) or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Sample (source, seed) and reviewed case_id must be unique")
    rows = []
    for case in cases:
        for side in ("supported", "unsupported"):
            annotation = case[side]
            sample = lookup[case["source_id"], annotation["seed"]]
            rows.append(dict(
                case_id=case["case_id"], source_id=case["source_id"], side=side,
                seed=sample["seed"], trace=sample["trace"], response=sample["response"],
                claim=annotation["target"], trace_available=(Path(directory) / sample["trace"]).is_file(),
                label_scope="reviewed_local_claim", whole_answer_label="unknown",
                evidence_status=case["evidence_status"],
                history_status=case["history_status"][side],
                generator=settings["model"], observer=settings["model"],
            ))
    table = pd.DataFrame(rows)
    table.to_csv(output / "pair_inventory.csv", index=False)
    pd.DataFrame(samples).to_csv(output / "sample_inventory.csv", index=False)
    return settings, samples, table


def claim_trace(directory, record, annotation, include_attention=True):
    with np.load(Path(directory) / record["trace"], allow_pickle=False) as saved:
        trace = {name: saved[name] for name in saved.files
                 if include_attention or name != "attention"}
    prompt = int(trace["prompt_length"])
    pieces = trace["token_text"][prompt:].tolist()
    text = "".join(pieces)
    ends = np.cumsum([len(piece) for piece in pieces])
    offsets = np.column_stack((np.r_[0, ends[:-1]], ends))
    indices = quote_tokens(text, offsets, annotation["target"])
    return trace, (int(indices[0]), int(indices[-1]) + 1)


def phase_positions(span, response_length, special_mask):
    start, stop = span
    positions = dict(before_claim=start - 1, onset=start,
                     back_half=start + max(1, (stop - start) // 2), post_claim=stop)
    return {phase: position for phase, position in positions.items()
            if 0 <= position < response_length and not special_mask[position]
            and (phase != "back_half" or position < stop)}


def reviewed_roles(tokenizer, prompt_ids, case):
    text, offsets = token_text_offsets(tokenizer, prompt_ids)
    roles = {}
    for role, quotes in case["source_roles"].items():
        for quote in quotes:
            if text.count(quote) != 1:
                raise ValueError(f"{case['case_id']}: ambiguous {role} quote: {quote}")
        roles[role] = np.unique(np.concatenate([quote_tokens(text, offsets, quote) for quote in quotes]))
    return roles


def source_groups(prompt_length, prefix_length, roles, span):
    evidence = np.union1d(roles["scope"], roles["supported_value"])
    inapplicable = roles["value_source"]
    if np.intersect1d(evidence, inapplicable).size:
        raise ValueError("Reviewed applicable/inapplicable token masks overlap")
    history = np.arange(prompt_length, prefix_length)
    query_self = np.array([prefix_length - 1])
    prior_history = np.arange(prompt_length, prefix_length - 1)
    start, stop = span
    claim = np.arange(prompt_length + start, min(prompt_length + stop, prefix_length))
    return dict(evidence=evidence, inapplicable_source=inapplicable, history=history,
                query_self=query_self, prior_history=prior_history,
                claim_history=claim, head_total=np.arange(prefix_length))


def compile_pair(directory, tokenizer, samples, case):
    lookup = {(str(row["source_id"]), row["seed"]): row for row in samples}
    result = {}
    for side in ("supported", "unsupported"):
        record = lookup[case["source_id"], case[side]["seed"]]
        trace, span = claim_trace(directory, record, case[side], include_attention=False)
        native = compile_side(directory, tokenizer, record, case[side], case)
        prompt_length = int(trace["prompt_length"])
        roles = reviewed_roles(tokenizer, trace["token_ids"][:prompt_length].tolist(), case)
        response_length = len(trace["chosen_logit"])
        positions = phase_positions(span, response_length, trace["special_mask"][prompt_length:])
        probes = {}
        for phase, position in positions.items():
            prefix = trace["token_ids"][:prompt_length + position].tolist()
            target = int(trace["token_ids"][prompt_length + position])
            if phase == "onset":
                observed = 0 if side == "supported" else 1
                if native["candidates"][observed][0] != target:
                    raise ValueError(f"{case['case_id']}/{side}: candidate/observed token mismatch")
            # Non-onset positions have no invented factual alternative. The
            # duplicated ID only makes NativeRun's unused local lens equal zero.
            candidates = native["candidates"] if phase == "onset" else [[target], [target]]
            probes[phase] = dict(native, prefix_ids=prefix, candidates=candidates,
                groups=source_groups(prompt_length, len(prefix), roles, span),
                record_writes=False, phase=phase, position=position, claim_span=list(span),
                actual_token=target, readout="sequence_margin" if phase == "onset" else "observed_logp")
        result[side] = probes
    left, right = result["supported"]["onset"], result["unsupported"]["onset"]
    np.testing.assert_array_equal(left["prefix_ids"][:left["prompt_length"]],
                                  right["prefix_ids"][:right["prompt_length"]])
    return result
