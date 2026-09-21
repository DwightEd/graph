"""Fit on mixed TRAIN sources; calibrate separately; freeze all TEST scores."""

from collections import defaultdict
import json

import numpy as np
from tqdm import tqdm

from ..fixed_graph.reference import (
    apply_calibration, binary_spans, calibrate, fit_reference, group_records,
    novelty_distance, sample_reference_rows, split_sources,
)
from ..offline_span.data import write_json
from .geometry import ENERGIES, PRIMARY, embed, fit_geometry


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def selected_files(selected):
    result = defaultdict(list)
    for row_number, (file, position, source) in enumerate(selected):
        result[file].append((row_number, position))
    return result


def selected_observations(root, selected):
    values = [None] * len(selected)
    for file, coordinates in selected_files(selected).items():
        with np.load(root / file, allow_pickle=False) as saved:
            observations = saved["observations"]
        for row_number, position in coordinates:
            values[row_number] = observations[position]
    return np.asarray(values, np.float64)


def embeddings_from_file(path, model, args, positions=None):
    with np.load(path, allow_pickle=False) as saved:
        layers = np.unique(saved["channels"][:, 0])
        return embed(saved["observations"], saved["coverage"], layers,
                     model, args, positions)


def reference_embeddings(root, selected, model, args):
    arrays = {}
    files = selected_files(selected)
    for file, coordinates in tqdm(files.items(), desc="reference geometry", leave=False):
        indices, positions = np.asarray(coordinates).T
        values, _, _ = embeddings_from_file(root / file, model, args, positions)
        for name, matrix in values.items():
            if name not in arrays:
                arrays[name] = np.empty((len(selected), matrix.shape[1]), np.float32)
            arrays[name][indices] = matrix
    return arrays


def score_file(path, model, references, args):
    embeddings, positions, counts = embeddings_from_file(path, model, args)
    scores = {}
    for name, values in embeddings.items():
        scores[name] = np.full(len(counts), np.nan, np.float32)
        if len(positions):
            if name.split("__")[1] in ENERGIES:
                scores[name][positions] = np.mean(values ** 2, axis=1)
            else:
                scores[name][positions] = novelty_distance(values, references[name])
    return scores, positions, counts, embeddings


def group_calibration(root, records, model, references, args):
    collected = {name: [] for name in references}
    sources = []
    for row in tqdm(records, desc="unlabelled calibration", leave=False):
        scores, positions, _, _ = score_file(root / row["file"], model, references, args)
        for name in collected:
            collected[name].extend(scores[name][positions])
        sources.extend([row["source_id"]] * len(positions))
    if not sources:
        raise ValueError("No covered calibration tokens for this task/generator")
    return {name: calibrate(np.asarray(values), sources, args.quantile)
            for name, values in collected.items()}


def fit_group(root, rows, roles, directory, args):
    fitting = [row for row in rows if roles[row["source_id"]] == "reference"]
    calibration = [row for row in rows if roles[row["source_id"]] == "calibration"]
    if not fitting or not calibration:
        raise ValueError("Every task/generator needs reference and calibration sources")
    selected = sample_reference_rows(root, fitting, args.bank_size,
                                     args.tokens_per_source, args.seed)
    observations = selected_observations(root, selected)
    model = fit_geometry(observations, args.signal_indices, args.ridge)
    arrays = reference_embeddings(root, selected, model, args)
    references = {name: (dict(energy=np.array(True)) if name.split("__")[1] in ENERGIES
                        else fit_reference(values, args.neighbors))
                  for name, values in arrays.items()}
    controls = group_calibration(root, calibration, model, references, args)
    np.savez_compressed(directory / "geometry.npz", **model)
    bank = {f"{name}___{key}": value for name, reference in references.items()
            for key, value in reference.items()}
    np.savez_compressed(directory / "bank.npz", **bank)
    result = dict(directory=directory.name, calibration=controls, bank_rows=selected,
                  reference_sources=sorted({source for _, _, source in selected}),
                  calibration_sources=sorted({row["source_id"] for row in calibration}))
    write_json(directory / "complete.json", result)
    return result


def fit(args):
    root = args.output / "observations"
    records = read_json(root / "manifest.json")["records"]
    roles = split_sources(records, args.seed)
    train = [row for row in records if row["split"] == "train"]
    output = args.output / "reference"
    output.mkdir(exist_ok=True)
    settings = dict(labels_used=False, roles=roles, groups={})
    for number, (group, rows) in enumerate(sorted(group_records(train).items())):
        directory = output / str(number)
        directory.mkdir(exist_ok=True)
        if args.resume and (directory / "complete.json").exists():
            configured = read_json(directory / "complete.json")
        else:
            configured = fit_group(root, rows, roles, directory, args)
        settings["groups"]["|".join(group)] = configured
    write_json(output / "settings.json", settings)
    write_json(output / "complete.json", dict(complete=True, labels_used=False))


def load_group(directory):
    with np.load(directory / "geometry.npz", allow_pickle=False) as saved:
        model = {key: saved[key] for key in saved.files}
    references = defaultdict(dict)
    with np.load(directory / "bank.npz", allow_pickle=False) as saved:
        for field in saved.files:
            name, key = field.split("___")
            references[name][key] = saved[field]
    return model, dict(references)


def score_answer(args, row, model, references, calibration, destination):
    source = args.output / "observations" / row["file"]
    scores, positions, counts, embeddings = score_file(source, model, references, args)
    arrays = dict(window_count=counts)
    for name, values in scores.items():
        arrays[name] = apply_calibration(values, calibration[name]).astype(np.float32)
        alarm = arrays[name] > calibration[name]["threshold"]
        arrays[name + "__alarm"] = alarm
        arrays[name + "__spans"] = binary_spans(alarm)
        if args.save_embeddings:
            arrays[name + "__embedding"] = embeddings[name]
    with np.load(source, allow_pickle=False) as saved:
        for key in ("coverage", "token_ids", "offsets", "prompt_length", "retained_mass",
                    "head_observed", "conditional_defined"):
            arrays[key] = saved[key]
    if not len(arrays["offsets"]):
        del arrays["offsets"]
    arrays["embedding_positions"] = positions
    # Keep each physical head's residual energy for audit, before scalar readout.
    layers, heads, features = model["marginal_root"].shape[:3]
    for method in ENERGIES:
        vectors = embeddings["all__" + method].reshape(len(positions), layers, heads, features)
        arrays[method + "__per_head"] = np.mean(vectors ** 2, axis=-1)
    arrays["position"] = np.arange(row["tokens"], dtype=np.float32)
    arrays["position"][~arrays["coverage"]] = np.nan
    partial = destination.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(destination)


def score(args):
    records = read_json(args.output / "observations/manifest.json")["records"]
    records = [row for row in records if row["split"] == "test"]
    settings = read_json(args.output / "reference/settings.json")
    output = args.output / "predictions"
    output.mkdir(exist_ok=True)
    loaded, saved = {}, []
    for number, row in enumerate(tqdm(records, desc="freeze geometry scores", unit="answer")):
        key = row["task"] + "|" + row["generator"]
        configured = settings["groups"][key]
        if key not in loaded:
            loaded[key] = load_group(args.output / "reference" / configured["directory"])
        path = output / f"{number:06d}.npz"
        if not (args.resume and path.exists()):
            model, references = loaded[key]
            score_answer(args, row, model, references, configured["calibration"], path)
        saved.append(dict(row, file=path.name))
    methods = list(next(iter(settings["groups"].values()))["calibration"])
    write_json(output / "freeze.json", dict(records=saved, methods=methods,
               complete=True, labels_used=False, primary="all__" + PRIMARY))
