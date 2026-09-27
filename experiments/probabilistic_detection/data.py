"""Reuse native scalar traces; official labels never enter feature construction."""

import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from experiments.native_support.dual_state.scoring import window_mean
from experiments.native_support.evaluate import annotation_targets
from experiments.native_support.message_carriers.token_representation import aggregate_units
from experiments.native_support.ragtruth_benchmark.data import annotations
from experiments.native_support.ragtruth_benchmark.selection import development_sources


CONTEXT = ("local_unit", "full_unit", "relative_position", "log_position",
           "log_answer_length", "log_prompt_length")
OBSERVATIONS = ("local_deviation", "full_deviation", "with_full_logp", "with_local_logp",
                "route", "attention", "entropy", "route_window", "entropy_window",
                "route_deviation", "entropy_deviation")
BASELINES = ("source_first", "source_refine", "local_unit", "full_unit",
             "raw_route", "route_window", "entropy")


def source_order(source):
    return hashlib.sha256(f"42:{source}".encode()).hexdigest()


def select_records(manifest, task, split, train_limit=None, dev_limit=None):
    development = set(development_sources(manifest["records"], task))
    records = [r for r in manifest["records"] if r["task"] == task and r["split"] == split]
    partitions = {}
    for name in ("fit", "dev") if split == "train" else ("test",):
        sources = {r["source_id"] for r in records
                   if split == "test" or (r["source_id"] in development) == (name == "dev")}
        ordered = sorted(sources, key=source_order)
        limit = train_limit if name == "fit" else dev_limit if name == "dev" else None
        partitions[name] = set(ordered[:limit])
    selected = [dict(r, partition=part) for r in records
                for part, sources in partitions.items() if r["source_id"] in sources]
    return selected


def answer_features(observed, native, response, prompt_length, window=16):
    units = response["units"]
    count = len(response["answer_ids"])
    if [u["start"] for u in units] != [0, *[u["stop"] for u in units[:-1]]] or units[-1]["stop"] != count:
        raise ValueError("Saved units must partition the original answer")
    for values in (observed, native):
        if not np.array_equal(values["token_id"], response["answer_ids"]):
            raise ValueError("Native trace token IDs do not match the saved response")
    local = aggregate_units(observed["source_local"], units)
    full = aggregate_units(observed["source_full"], units)
    route, entropy = observed["raw_route"], observed["entropy"]
    position = np.arange(count)
    context = np.column_stack((local, full, position / max(1, count - 1),
        np.log1p(position), np.full(count, np.log1p(count)),
        np.full(count, np.log1p(prompt_length))))
    route_window = window_mean(route, window, offline=True)
    features = np.column_stack((observed["source_local"] - local,
        observed["source_full"] - full, native["full"], native["local"],
        route, observed["raw_attention"], entropy, route_window,
        window_mean(entropy, window, offline=True), route - aggregate_units(route, units),
        entropy - aggregate_units(entropy, units)))
    if not np.isfinite(context).all() or not np.isfinite(features).all():
        raise ValueError("Native feature cache contains nonfinite measurements")
    unit_index = np.empty(count, dtype=np.int32)
    for index, unit in enumerate(units):
        unit_index[unit["start"]:unit["stop"]] = index
    return context, features, unit_index


def valid_tokens(response):
    offsets = np.asarray(response["offsets"])
    ordinary = ~np.isin(response["answer_ids"], response["special_ids"])
    return ordinary & (offsets[:, 1] > offsets[:, 0])


def baseline_scores(root, refine, record, observed, response, refine_method):
    count = record["tokens"]
    first = read_arrays(root / record["directory"] / "selected_scores.npz")
    if not np.array_equal(first["token_id"], observed["token_id"]):
        raise ValueError("Source-first baseline token alignment differs")
    refined = np.full(count, np.nan)
    path = refine / record["directory"] / "scores.npz"
    if path.exists():
        values = read_arrays(path)
        if not np.array_equal(values["token_id"], observed["token_id"]):
            raise ValueError("Refinement baseline token alignment differs")
        refined = values[refine_method]
    units = response["units"]
    return np.column_stack((first["selected_detector"], refined,
        aggregate_units(observed["source_local"], units),
        aggregate_units(observed["source_full"], units), observed["raw_route"],
        window_mean(observed["raw_route"], 16, offline=True), observed["entropy"]))


def load_answer(root, refine, record, refine_method):
    directory = root / record["directory"]
    response = read_json(directory / "response.json")
    if len(response["answer_ids"]) != record["tokens"]:
        raise ValueError("Manifest token count differs from saved response")
    observed = read_arrays(directory / "observations.npz")
    native = read_arrays(directory / "with_source.npz")
    source = read_json(root / record["source_file"])
    context, features, unit_index = answer_features(observed, native, response,
                                                  len(source["prompt_with_source"]))
    valid = valid_tokens(response)
    baselines = baseline_scores(root, refine, record, observed, response, refine_method)
    arrays = dict(context=context[valid], observations=features[valid],
        baselines=baselines[valid], target=np.flatnonzero(valid),
        token_id=observed["token_id"][valid], unit_index=unit_index[valid])
    return arrays, response, valid


def prepare_task(root, refine, output, task, split, train_limit=None, dev_limit=None):
    manifest = read_json(root / "manifest.json")
    records = select_records(manifest, task, split, train_limit, dev_limit)
    choice = read_json(refine / "selection.json")["choices"][task]["method"]
    truth = annotations(root, manifest, records) if split == "train" else None
    source_ids = sorted({r["source_id"] for r in records})
    source_lookup = {identity: index for index, identity in enumerate(source_ids)}
    batches, metadata, cursor = [], [], 0
    for answer_index, record in enumerate(tqdm(records, desc=f"prepare {task}/{split}")):
        arrays, response, valid = load_answer(root, refine, record, choice)
        count = len(arrays["target"])
        arrays["source_index"] = np.full(count, source_lookup[record["source_id"]], np.int32)
        arrays["answer_index"] = np.full(count, answer_index, np.int32)
        arrays["development"] = np.full(count, record["partition"] == "dev", bool)
        if truth is not None:
            label, onset, first, annotated_valid = annotation_targets(
                truth[record["id"]], record["tokens"], record["id"])
            if not np.array_equal(valid, annotated_valid):
                raise ValueError("Annotation and feature valid-token masks differ")
            arrays.update(labels=label[valid], onsets=onset[valid], firsts=first[valid])
        metadata.append(dict(record, packed_start=cursor, packed_stop=cursor + count))
        batches.append(arrays)
        cursor += count
    packed = {name: np.concatenate([row[name] for row in batches]) for name in batches[0]}
    write_arrays(output / f"{task}_{split}.npz", **packed)
    write_json(output / f"{task}_{split}.json", dict(records=metadata, sources=source_ids,
        task=task, split=split, context=CONTEXT, observations=OBSERVATIONS, baselines=BASELINES,
        source_cache=str(root.resolve()), refine_cache=str(refine.resolve()),
        official_manifest=manifest["model"], labels_loaded=truth is not None,
        train_source_limit=train_limit, dev_source_limit=dev_limit, window=16, offline=True))


def load_pack(output, task, split):
    return read_arrays(output / f"{task}_{split}.npz"), read_json(output / f"{task}_{split}.json")


def source_weights(source_index):
    _, inverse, counts = np.unique(source_index, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / weights.sum())


def evaluation_labels(root, pack, metadata):
    """Called only after all test predictions are frozen."""
    manifest = read_json(root / "manifest.json")
    truth = annotations(root, manifest, metadata["records"])
    joined = {name: [] for name in ("labels", "onsets", "firsts")}
    for row in metadata["records"]:
        annotation = truth[row["id"]]
        label, onset, first, valid = annotation_targets(annotation, row["tokens"], row["id"])
        region = slice(row["packed_start"], row["packed_stop"])
        if not np.array_equal(np.asarray(annotation["token_ids"])[valid], pack["token_id"][region]):
            raise ValueError("Frozen predictions and official annotation tokens differ")
        for name, value in zip(joined, (label, onset, first)):
            joined[name].append(value[valid])
    return {name: np.concatenate(values) for name, values in joined.items()}
