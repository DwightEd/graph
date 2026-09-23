"""Functional contrast data: finite-bank meaning changes and native message paths."""

import numpy as np
from scipy.special import logsumexp

from state_audit.functional_readout import (
    contrast_effects, contrast_weights, distribution_change, entropy_partition,
)
from state_audit.storage import read_arrays, read_json, write_arrays, write_csv, write_json

CHANNELS = ("head_total", "head_residual", "head_ffn_mediated")


def summarize_bank(directory, source_groups, source_count):
    bank = read_json(directory / "bank.json")
    if bank != read_json(directory / "captured_bank.json"):
        raise ValueError(f"{directory}: candidate bank changed after capture")
    captures = [read_arrays(directory / f"candidate_{index:02d}.npz")
                for index in range(len(bank["candidates"]))]
    for candidate, captured in zip(bank["candidates"], captures):
        if not np.array_equal(candidate["token_ids"], captured["candidate_ids"]):
            raise ValueError(f"{directory}: captured candidate token IDs differ")
    scores = np.array([row["log_probability"].sum(dtype=float) for row in captures])
    previous = np.array([row["pre_ffn_log_probability"].sum(dtype=float) for row in captures])
    names = [candidate["group"] for candidate in bank["candidates"]]
    groups = np.array([list(dict.fromkeys(names)).index(name) for name in names])
    summary = {**entropy_partition(scores, groups), **distribution_change(previous, scores, groups),
        "max_rms_identity_error": max(float(np.abs(row["rms_identity_error"]).max()) for row in captures),
        "max_rms_rounding_error": max(float(np.abs(row["rms_native_rounding_error"]).max()) for row in captures),
        "max_head_reconstruction_error": max(float(row["head_reconstruction_error"].max()) for row in captures),
        "candidate_lengths": [len(row["candidate_ids"]) for row in captures],
        "candidate_log_scores": scores.tolist(), "group_names": list(dict.fromkeys(names))}
    write_json(directory / "readout.json", summary)
    contrasts = []
    for negative in range(1, len(summary["group_names"])):
        weights = contrast_weights(scores, groups, 0, negative)
        name = "observed_vs_" + summary["group_names"][negative]
        contrasts.extend(write_contrast(directory, name, weights, scores, captures, source_groups, source_count))
    # Wording control uses the same observed meaning, without changing truth conditions.
    weights = np.zeros(len(scores))
    weights[:2] = (1, -1)
    contrasts.extend(write_contrast(directory, "observed_wording", weights, scores,
                                    captures, source_groups, source_count))
    return summary, contrasts


def write_contrast(directory, name, weights, scores, captures, source_groups, source_count):
    roots = contrast_effects(weights, [row["prefix_root_sensitivity"] for row in captures])
    prefix_groups = np.asarray(source_groups[:len(roots)])
    arrays = dict(candidate_weights=weights, prefix_root_sensitivity=roots,
                  prefix_group_ids=prefix_groups)
    # Boundary query is identical across branches; later branch coordinates differ.
    for channel in CHANNELS:
        arrays[channel] = contrast_effects(weights, [row[channel][0] for row in captures])
    group_count = source_count + 3
    arrays["root_positive"] = np.bincount(prefix_groups, weights=np.maximum(roots, 0), minlength=group_count)
    arrays["root_negative"] = np.bincount(prefix_groups, weights=np.maximum(-roots, 0), minlength=group_count)
    positive, negative = weights > 0, weights < 0
    arrays["log_odds"] = np.asarray(logsumexp(scores[positive]) - logsumexp(scores[negative]))
    write_arrays(directory / f"contrast_{name}.npz", **arrays)
    rows = []
    for layer in range(arrays["head_total"].shape[0]):
        for head in range(arrays["head_total"].shape[1]):
            for group in range(arrays["head_total"].shape[2]):
                rows.append(dict(contrast=name, layer=layer, head=head, group=group,
                    **{channel: float(arrays[channel][layer, head, group]) for channel in CHANNELS}))
    return rows


def write_observed_tokens(directory, response, bank, identity):
    """Native observed branch only; candidate branches never inherit natural labels."""
    capture = read_arrays(directory / "candidate_00.npz")
    rows = []
    for offset, target in enumerate(range(bank["start"], bank["stop"])):
        rows.append(dict(**identity, target=target,
            token=response["token_text"][response["prompt_length"] + target],
            **{name: float(capture[name][offset]) for name in (
                "rms_direct_margin", "rms_rescale_margin", "rms_margin_delta",
                "rms_identity_error", "rms_native_rounding_error", "pre_ffn_entropy")},
            log_probability=float(capture["log_probability"][offset])))
    return rows


def report(destination, settings):
    units, heads, tokens = [], [], []
    for directory in sorted((destination / "responses").glob("*/unit_*")):
        if not (directory / "complete.json").is_file():
            continue
        identity = read_json(directory / "complete.json")
        response = settings["responses"][identity["response_index"]]
        sources = read_json(directory.parent / "sources.json")
        summary, rows = summarize_bank(directory, sources["group_ids"], len(sources["blocks"]))
        units.append({**identity, **summary})
        heads.extend({**identity, **row} for row in rows)
        tokens.extend(write_observed_tokens(directory, response, read_json(directory / "bank.json"), identity))
    write_json(destination / "units.json", units)
    if heads:
        write_csv(destination / "head_contrasts.csv", heads, list(heads[0]))
        write_csv(destination / "observed_tokens.csv", tokens, list(tokens[0]))
    return dict(completed_units=len(units), observed_tokens=len(tokens),
                detector_evaluation=False, new_auroc=None)
