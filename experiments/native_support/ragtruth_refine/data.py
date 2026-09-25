"""Read original scalar caches; portable reviews carry aligned annotations separately."""

from collections import defaultdict

import numpy as np
from tqdm import tqdm
from state_audit.storage import read_json, write_arrays, write_json

from ..ragtruth_benchmark.data import annotations
from ..ragtruth_benchmark.selection import development_sources
from .scoring import features, fit_scales, score_answer


def select_records(manifest, tasks, select_on_train):
    records, development = [], {}
    for task in tasks:
        test = [row for row in manifest["records"] if row["task"] == task and row["split"] == "test"]
        if not test:
            raise ValueError(f"{task}: input contains no test answers; explicitly choose available --tasks")
        records.extend(test)
        if select_on_train:
            if "development_sources" in manifest:
                sources = manifest["development_sources"][task]
            else:
                sources = development_sources(manifest["records"], task)
            train = [row for row in manifest["records"] if row["task"] == task and row["split"] == "train"
                     and row["source_id"] in sources]
            if {row["source_id"] for row in train} != set(sources):
                raise ValueError(f"{task}: portable cache lacks development sources")
            if set(sources) & {row["source_id"] for row in test}:
                raise ValueError(f"{task}: train/test sources overlap")
            development[task] = sources
            records.extend(train)
    return records, development


def read_observed(reader, record, previous_selected):
    directory = record["directory"]
    observed = reader.arrays(directory + "/observations.npz")
    if previous_selected and "previous_selected" not in observed:
        previous = reader.arrays(directory + "/selected_scores.npz")
        if not np.array_equal(previous["token_id"], observed["token_id"]):
            raise ValueError(f"{record['id']}: previous detector token alignment differs")
        observed["previous_selected"] = previous["selected_detector"]
    return observed


def load_row(reader, record, previous_selected, window):
    response = reader.json(record["directory"] + "/response.json")
    observed = read_observed(reader, record, previous_selected)
    if not np.array_equal(observed["token_id"], response["answer_ids"]):
        raise ValueError(f"{record['id']}: measurement and response token identities differ")
    count = record["tokens"]
    for name in ("source_local", "source_full", "raw_route", "raw_attention", "entropy"):
        if observed[name].shape != (count,) or not np.isfinite(observed[name]).all():
            raise ValueError(f"{record['id']}: {name} is incomplete or nonfinite")
    units = response["units"]
    if [unit["start"] for unit in units] != [0, *[unit["stop"] for unit in units[:-1]]] or units[-1]["stop"] != count:
        raise ValueError(f"{record['id']}: unit intervals must partition the answer")
    return dict(record=record, response=response, observed=observed, features=features(observed, units, window))


def score_all(args, reader, manifest):
    groups = defaultdict(list)
    for record in manifest["records"]:
        groups[record["task"], record["split"]].append(record)
    for (task, split), records in sorted(groups.items()):
        paths = [args.output / row["directory"] / "scores.npz" for row in records]
        if args.resume and all(path.exists() for path in paths):
            continue
        rows = [load_row(reader, row, manifest["previous_selected"], args.window)
                for row in tqdm(records, desc=f"read {task}/{split}")]
        scales = fit_scales(rows)
        for row, path in tqdm(list(zip(rows, paths)), desc=f"refine {task}/{split}"):
            if args.resume and path.exists():
                continue
            write_arrays(path, **score_answer(row, scales))
    coverage = dict(status="complete", scored_answers=len(manifest["records"]),
        scored_tokens=sum(row["tokens"] for row in manifest["records"]), labels_used_for_fixed_scores=False)
    write_json(args.output / "coverage.json", coverage)
    return coverage


def read_annotations(reader, manifest, records):
    original = reader.json("manifest.json")
    if original.get("portable_refinement_cache"):
        saved = reader.json("annotations.json")
        return {row["id"]: saved[row["id"]] for row in records}
    return annotations(reader.path, original, records)


def require_scores(output, manifest):
    if read_json(output / "coverage.json")["status"] != "complete":
        raise ValueError("Freeze all selected answers before selection or evaluation")
    for record in manifest["records"]:
        if not (output / record["directory"] / "scores.npz").exists():
            raise ValueError(f"{record['id']}: frozen score file is missing")
