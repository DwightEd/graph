"""Capture once, then score the complete observed prefix in causal order."""

import sys
from collections import deque
from contextlib import closing
from time import perf_counter

import numpy as np
from state_audit.storage import read_arrays, write_arrays, write_csv, write_json
from tqdm import tqdm

from .inputs import validate_tokenizer
from .routes import ROUTE_SCORES, measure_routes
from .score import prompt_focus, support_step


def pending_targets(response, directory):
    count = len(response["token_ids"]) - response["prompt_length"]
    return [index for index in range(count) if not (directory / f"token_{index:06d}.npz").exists()]


def capture_response(
    model, tokenizer, response, directory, chunk_size, *, query_chunk_size=8, compress_cache=False
):
    from state_audit.native_forward import iter_forward_traces

    validate_tokenizer(response, tokenizer)
    count = len(response["token_ids"]) - response["prompt_length"]
    pending = pending_targets(response, directory)
    if not pending:
        return
    iterator = iter_forward_traces(
        model, response["token_ids"], response["prompt_length"], pending,
        tokenizer.all_special_ids, prefill_chunk_size=chunk_size, query_chunk_size=query_chunk_size,
    )
    started = perf_counter()
    capture_seconds, write_seconds = 0.0, 0.0
    with closing(iterator):
        for index in tqdm(pending, desc=response["id"], leave=False):
            arrays = next(iterator)
            capture_seconds += float(arrays["capture_seconds"])
            writing = perf_counter()
            write_arrays(directory / f"token_{index:06d}.npz", compressed=compress_cache, **arrays)
            write_seconds += perf_counter() - writing
    timing = {
        "captured_tokens": len(pending), "reused_tokens": count - len(pending),
        "query_chunk_size": query_chunk_size, "compressed": compress_cache,
        "capture_seconds": capture_seconds, "write_seconds": write_seconds,
        "wall_seconds": perf_counter() - started,
    }
    write_json(directory / "capture_timing.json", timing)
    tqdm.write(f"{response['id']}: captured {len(pending)}, reused {count - len(pending)}, "
               f"capture {capture_seconds:.1f}s, write {write_seconds:.1f}s", file=sys.stderr)


def score_response(
    response, directory, *, destination=None, evidence_mask=None, with_routes=False,
    focus_width=8, focus_history=3, read_rise=0.1,
):
    prompt = response["prompt_length"]
    count = len(response["token_ids"]) - prompt
    support, fingerprints, rows, parents = [], [], [], []
    recent_attention = deque(maxlen=focus_history)
    focus_arrays = []
    route_arrays, route_history = [], deque(maxlen=3)
    for target in range(count):
        arrays = read_arrays(directory / f"token_{target:06d}.npz")
        verify_position(arrays, response, target)
        metrics, fingerprint, weights = support_step(arrays, prompt, support, fingerprints)
        attention = arrays["attention"].copy()
        attention[..., arrays["group_ids"] == 2] = 0
        if evidence_mask is not None:
            attention[..., :prompt] *= evidence_mask
        focus = prompt_focus(attention, prompt, list(recent_attention), focus_width)
        if with_routes:
            route_scores, details, anchors = measure_routes(
                arrays, prompt, evidence_mask, route_history, focus, focus_width
            )
            metrics.update(route_scores)
            route_arrays.append(details)
            route_history.append(anchors)
        row = token_row(response, target, arrays, metrics, focus, len(recent_attention), read_rise)
        support.append(metrics["support"])
        fingerprints.append(fingerprint)
        parents.append(weights)
        rows.append(row)
        focus_arrays.append(focus)
        recent_attention.append(attention[..., :prompt].copy())
    save_scores(destination or directory, rows, fingerprints, parents, focus_arrays, route_arrays)
    return rows


def verify_position(arrays, response, target):
    position = response["prompt_length"] + target
    if int(arrays["query"]) != position - 1 or int(arrays["target"]) != target:
        raise ValueError(f"{response['id']}: cache prediction-row mismatch at {target}")
    if int(arrays["observed_id"]) != response["token_ids"][position]:
        raise ValueError(f"{response['id']}: cache target token mismatch at {target}")
    if arrays["attention"].shape[-1] != position:
        raise ValueError(f"{response['id']}: cache contains missing or future keys at {target}")


def token_row(response, target, arrays, metrics, focus, prior_rows, read_rise):
    position = response["prompt_length"] + target
    return dict(
        response_id=response["id"], source_id=response["source_id"], target=target,
        query=int(arrays["query"]), token_id=int(arrays["observed_id"]),
        token=response["token_text"][position], competitor_id=int(arrays["competitor_id"]),
        **metrics,
        focus_prior_rows=prior_rows,
        focus_max_gain=float(np.nanmax(focus["focus_gain"])) if prior_rows else "",
        read_event_heads=int((focus["focus_gain"] >= read_rise).sum()) if prior_rows else "",
    )


def save_scores(directory, rows, fingerprints, parents, focuses, routes=()):
    offsets = np.concatenate(([0], np.cumsum([len(parent) for parent in parents])))
    arrays = {
        name: np.asarray([row[name] for row in rows])
        for name in ("target", "query", "token_id", "risk", "direct_risk", "inherited_risk")
    }
    arrays.update(
        fingerprints=np.asarray(fingerprints), history_indptr=offsets,
        history_indices=np.concatenate([np.arange(len(parent)) for parent in parents]),
        history_weights=np.concatenate(parents),
    )
    for name in ("focus_start", "focus_mass", "focus_gain"):
        arrays[name] = np.stack([focus[name] for focus in focuses])
    if routes:
        for name in (*ROUTE_SCORES, "entropy", "surprisal", "negative_margin"):
            arrays[name] = np.asarray([row[name] for row in rows])
        for name in routes[0]:
            arrays[name] = np.stack([row[name] for row in routes])
    write_arrays(directory / "scores.npz", **arrays)
    write_csv(directory / "tokens.csv", rows, list(rows[0]))
    write_json(directory / "complete.json", {"tokens": len(rows), "complete": True})
