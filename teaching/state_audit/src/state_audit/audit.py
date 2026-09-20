"""Offline orchestration: measurements first, annotation joins last."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .annotations import label_report, token_spans
from .measurements import measure_heads, reanchor_candidates, source_masks
from .roles import swap_probe
from .storage import read_arrays, read_json, start_stage, write_arrays, write_csv, write_json

HEAD_FIELDS = [
    "target",
    "layer",
    "head",
    "ordinary_mass",
    "evidence_mass",
    "history_mass",
    "other_mass",
    "attention_entropy",
    "evidence_write_norm",
    "history_write_norm",
    "evidence_history_cosine",
]
NODE_FIELDS = [
    "target",
    "layer",
    "head",
    "source",
    "mass",
    "baseline",
    "rise",
    "query",
    "query_token",
    "target_token",
    "source_id",
    "key",
    "key_token",
]
SPAN_FIELDS = [
    "pair",
    "kind",
    "start",
    "end",
    "length",
    "text",
    "pre_onset_observable",
    "pre_onset_queries_observed",
    "onset_observable",
    "nodes_before",
    "node_at_onset",
    "nodes_inside",
    "first_inside_delay",
]
ROLE_FIELDS = [
    "layer",
    "target",
    "head",
    "pair",
    "left_keys",
    "right_keys",
    "positional",
    "symbolic",
    "swapped_mass",
    "relative_contrast",
    "identifiable",
    "before_left",
    "before_right",
    "after_left",
    "after_right",
    "native_max_error",
]


def describe_nodes(nodes: list[dict], trace: dict, answer: dict, layer: int) -> list[dict]:
    sources = source_masks(answer, trace["attention"].shape[-1])
    nodes = [
        node
        for node in nodes
        if answer["response_ids"][node["target"]] not in answer["special_token_ids"]
    ]
    for node in nodes:
        target, head, source = node["target"], node["head"], node["source"]
        keys = np.flatnonzero(sources["ordinary"] & (sources["sources"] == source))
        key = int(keys[np.argmax(trace["attention"][head, target, keys])])
        query = int(trace["queries"][target])
        node.update(
            layer=layer,
            query=query,
            query_token=answer["token_strings"][query],
            target_token=answer["token_strings"][query + 1],
            key=key,
            key_token=answer["token_strings"][key],
            source_id=answer["evidence"][source]["id"],
        )
    return nodes


def audit_layer(
    root: Path, directory: Path, output: Path, layer: int, answer: dict, settings: dict
) -> list[dict]:
    trace = read_arrays(directory / "trace" / f"layer_{layer:03d}.npz")
    weights = read_arrays(root / "weights" / f"layer_{layer:03d}.npz")["output_projection"]
    measured = measure_heads(trace, weights, answer)
    arrays = {key: value for key, value in measured.items() if value is not None}
    write_arrays(output / f"observations_{layer:03d}.npz", **arrays)
    rows = []
    for token, head in np.ndindex(measured["ordinary_mass"].shape):
        row = dict(target=token, layer=layer, head=head)
        row.update({name: float(measured[name][token, head]) for name in HEAD_FIELDS[3:]})
        rows.append(row)
    write_csv(output / f"heads_{layer:03d}.csv", rows, HEAD_FIELDS)
    nodes = reanchor_candidates(
        measured["source_mass"],
        settings["window"],
        settings["minimum_mass"],
        settings["minimum_rise"],
    )
    if settings["roles"]:
        roles = [dict(layer=layer, **row) for row in swap_probe(trace, answer)]
        write_csv(output / f"roles_{layer:03d}.csv", roles, ROLE_FIELDS)
    return describe_nodes(nodes, trace, answer, layer)


def token_rows(answer: dict, readout: dict) -> list[dict]:
    spans = token_spans(answer)
    rows = []
    for index, token in enumerate(answer["response_ids"]):
        query = answer["prompt_length"] + index - 1
        ordinary = token not in answer["special_token_ids"]
        labeled = spans is not None and ordinary
        rows.append(
            dict(
                target=index,
                query=query,
                token_id=token,
                token=answer["token_strings"][query + 1],
                ordinary=ordinary,
                target_logp=float(readout["target_logp"][index]),
                logit_entropy=float(readout["logit_entropy"][index]),
                hallucination=any(a <= index < b for a, b in spans) if labeled else None,
                onset=any(index == a for a, _ in spans) if labeled else None,
            )
        )
    return rows


def audit_sample(root: Path, sample: dict, output: Path, settings: dict) -> dict:
    directory = root / "samples" / f"{sample['index']:06d}"
    answer = read_json(directory / "answer.json")
    complete = read_json(directory / "trace" / "complete.json")
    nodes = []
    for layer in complete["layers"]:
        nodes.extend(audit_layer(root, directory, output, layer, answer, settings))
    write_csv(output / "nodes.csv", nodes, NODE_FIELDS)
    # All observations and candidates above are frozen before this label-dependent join.
    spans, summary = label_report(answer, nodes, settings["window"])
    write_csv(output / "span_links.csv", spans, SPAN_FIELDS)
    rows = token_rows(answer, read_arrays(directory / "trace" / "readout.npz"))
    write_csv(output / "tokens.csv", rows, list(rows[0]))
    summary.update(
        id=answer["id"],
        source_id=answer["source_id"],
        response_tokens=len(rows),
        candidate_rows=len(nodes),
        candidate_tokens=len({row["target"] for row in nodes}),
        reanchor_applicable=bool(answer["evidence"]),
    )
    write_json(output / "summary.json", summary)
    return summary


def audit_run(root: Path, output: Path, settings: dict) -> dict:
    if settings["window"] < 1 or not 0 <= settings["minimum_mass"] <= 1:
        raise ValueError("window >= 1 and minimum_mass in [0, 1] are required")
    if not 0 <= settings["minimum_rise"] <= 1:
        raise ValueError("minimum_rise must be in [0, 1]")
    samples = read_json(root / "run.json")["samples"]
    start_stage(output / "settings.json", settings, resume=True)
    reports = [
        audit_sample(root, sample, output / f"{sample['index']:06d}", settings)
        for sample in tqdm(samples, desc="offline audits")
    ]
    labeled = [row for row in reports if row["label_status"] == "available"]
    errors = sum(row["error_tokens"] for row in labeled)
    continuations = sum(row["continuation_tokens"] for row in labeled)
    summary = dict(
        purpose="observational_audit_not_detector_evaluation",
        samples=reports,
        sources=len({row["source_id"] for row in reports}),
        labeled_answers=len(labeled),
        error_tokens=errors,
        continuation_tokens=continuations,
        continuation_fraction=continuations / errors if errors else None,
    )
    write_json(output / "summary.json", summary)
    return summary
