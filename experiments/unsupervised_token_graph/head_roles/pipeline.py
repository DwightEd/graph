"""Identical label-free banks/readouts for all frozen head-mask ablations."""

from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..fixed_graph.reference import (
    apply_calibration, binary_spans, calibrate, fit_reference, group_records,
    novelty_distance, sample_reference_rows,
)
from ..head_geometry.pipeline import read_json, selected_observations
from ..offline_span.data import write_json


PRIMARY = "contrast__drop_positional"


def load_masks(args):
    layout = read_json(args.output / "observations/manifest.json")["channels"]
    with np.load(args.output / "profile/priors.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["channels"], layout):
            raise ValueError("Prior and cache physical heads differ")
        names = ["all", "drop_positional", "drop_symbolic"]
        names.extend(f"random_{index}" for index in range(args.random_controls))
        names.extend(f"symbolic_random_{index}" for index in range(args.random_controls))
        return {name: saved[name].copy() for name in names}


def masked_contrast(observations, keep):
    selected = keep.reshape(observations.shape[1:3])
    output = np.zeros_like(observations)
    for layer, heads in enumerate(selected):
        values = observations[:, layer, heads]
        output[:, layer, heads] = values - values.mean(axis=1, keepdims=True)
    return output


def representations(observations, masks):
    result = {}
    for name, keep in masks.items():
        contrast = masked_contrast(observations, keep)
        for basis, matrix in (("raw", observations), ("contrast", contrast)):
            matrix = matrix.reshape(len(matrix), -1, matrix.shape[-1])
            result[f"{basis}__{name}"] = matrix[:, keep].reshape(len(matrix), -1)
    return result


def score_values(observations, masks, references):
    if not len(observations):
        return {name: np.empty(0) for name in references}
    values = representations(observations, masks)
    return {name: novelty_distance(matrix, references[name]) for name, matrix in values.items()}


def calibrate_group(args, rows, masks, references):
    collected = defaultdict(list)
    sources = []
    for row in tqdm(rows, desc="calibrate masked banks", leave=False):
        with np.load(args.output / "observations" / row["file"], allow_pickle=False) as saved:
            values = saved["observations"][saved["coverage"]]
        scores = score_values(values, masks, references)
        for name, values in scores.items():
            collected[name].extend(values)
        sources.extend([row["source_id"]] * len(values))
    if not sources:
        raise ValueError("No calibration tokens")
    return {name: calibrate(np.asarray(values), sources, args.quantile)
            for name, values in collected.items()}


def fit(args):
    manifest = read_json(args.output / "observations/manifest.json")
    roles = read_json(args.output / "profile/manifest.json")["roles"]
    masks = load_masks(args)
    train = [row for row in manifest["records"] if row["split"] == "train"]
    root = args.output / "reference"
    root.mkdir(exist_ok=True)
    groups = {}
    for number, (group, rows) in enumerate(sorted(group_records(train).items())):
        directory = root / str(number)
        directory.mkdir(exist_ok=True)
        if args.resume and (directory / "complete.json").exists():
            groups["|".join(group)] = read_json(directory / "complete.json")
            continue
        reference = [row for row in rows if roles[row["source_id"]] == "reference"]
        calibration = [row for row in rows if roles[row["source_id"]] == "calibration"]
        if not reference or not calibration:
            raise ValueError("Every task/generator requires reference and calibration sources")
        chosen = sample_reference_rows(args.output / "observations", reference,
                                      args.bank_size, args.tokens_per_source, args.seed)
        values = selected_observations(args.output / "observations", chosen)
        arrays = representations(values, masks)
        banks = {name: fit_reference(matrix, args.neighbors) for name, matrix in arrays.items()}
        controls = calibrate_group(args, calibration, masks, banks)
        flat = {f"{name}___{key}": value for name, bank in banks.items() for key, value in bank.items()}
        np.savez_compressed(directory / "bank.npz", **flat)
        result = dict(directory=str(number), calibration=controls, bank_rows=chosen)
        write_json(directory / "complete.json", result)
        groups["|".join(group)] = result
    write_json(root / "settings.json", dict(groups=groups, roles=roles, labels_used=False))


def load_banks(directory):
    result = defaultdict(dict)
    with np.load(directory / "bank.npz", allow_pickle=False) as saved:
        for field in saved.files:
            name, key = field.split("___")
            result[name][key] = saved[field]
    return dict(result)


def score_answer(args, row, masks, references, controls, path):
    with np.load(args.output / "observations" / row["file"], allow_pickle=False) as saved:
        coverage = saved["coverage"]
        scores = score_values(saved["observations"][coverage], masks, references)
        arrays = {key: saved[key].copy() for key in
                  ("coverage", "token_ids", "offsets", "prompt_length", "retained_mass",
                   "head_observed", "conditional_defined")}
    for name, values in scores.items():
        score = np.full(len(coverage), np.nan, np.float32)
        score[coverage] = apply_calibration(values, controls[name])
        arrays[name] = score
        arrays[name + "__alarm"] = score > controls[name]["threshold"]
        arrays[name + "__spans"] = binary_spans(arrays[name + "__alarm"])
    arrays["position"] = np.where(coverage, np.arange(len(coverage)), np.nan)
    arrays["window_count"] = coverage.astype(int)
    if not len(arrays["offsets"]):
        del arrays["offsets"]
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def score(args):
    records = read_json(args.output / "observations/manifest.json")["records"]
    settings = read_json(args.output / "reference/settings.json")
    masks = load_masks(args)
    root = args.output / "predictions"
    root.mkdir(exist_ok=True)
    loaded, saved = {}, []
    for row in tqdm([row for row in records if row["split"] == "test"], desc="freeze prior ablations"):
        group = row["task"] + "|" + row["generator"]
        configured = settings["groups"][group]
        if group not in loaded:
            loaded[group] = load_banks(args.output / "reference" / configured["directory"])
        path = root / row["file"]
        if not (args.resume and path.exists()):
            score_answer(args, row, masks, loaded[group], configured["calibration"], path)
        saved.append(row)
    methods = list(next(iter(settings["groups"].values()))["calibration"])
    write_json(root / "freeze.json", dict(records=saved, methods=methods,
               primary=PRIMARY, complete=True, labels_used=False))
