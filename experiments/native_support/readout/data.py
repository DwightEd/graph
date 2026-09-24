"""Read immutable observable caches; validate token identities at the boundary."""

from contextlib import closing
import csv
import io

import numpy as np
from tqdm import tqdm

from ..choice_cache import CaptureReader
from ..evaluate import annotation_targets
from .features import ROLES, SCALARS, OPERATORS, context_features, head_features, temporal_features

BASELINES = ("raw_route", "observable_route", "route_offline_mean", "raw_attention", "entropy")
CHOICE_FIELDS = ("local_source_deficit", "lineage_source_deficit", "local_opposition", "lineage_opposition")


def operator_features(rows, response_id, count):
    selected = [row for row in rows if row["response_id"] == response_id]
    layers = sorted({int(row["layer"]) for row in selected})
    keys = {(int(row["target"]), int(row["layer"])) for row in selected}
    expected = {(target, layer) for target in range(count) for layer in layers}
    if keys != expected or len(selected) != len(expected) or not layers:
        raise ValueError(f"{response_id}: operator rows missing or duplicated")
    values = np.empty((count, len(layers), len(OPERATORS)))
    for row in selected:
        for column, name in enumerate(OPERATORS):
            value = float(row[name]) if row[name] else np.nan
            values[int(row["target"]), layers.index(int(row["layer"])), column] = value
    for column, name in enumerate(OPERATORS):
        if name.endswith("energy"):
            values[..., column] = np.log1p(values[..., column])
    return values.reshape(count, -1), [f"operator/L{layer}/{name}" for layer in layers for name in OPERATORS]


def answer_features(reader, response, index, operators, mode, layer_scope, window):
    directory = f"responses/{index:04d}"
    saved = reader.arrays(directory + "/scores.npz")
    token_ids = np.asarray(response["token_ids"][response["prompt_length"]:])
    if not np.array_equal(saved["token_id"], token_ids) or not np.array_equal(saved["target"], np.arange(len(token_ids))):
        raise ValueError(f"{response['id']}: scored tokens differ from settings")
    source_count = len(reader.json(directory + "/sources.json")["blocks"])
    position, position_names = context_features(response, source_count)
    operator, operator_names = operator_features(operators, response["id"], len(token_ids))
    scalars = np.column_stack([saved[name] for name in SCALARS])
    values = np.column_stack((scalars, operator))
    names = list(SCALARS) + operator_names
    views = {"position": position, "routing": scalars[:, :3], "instant": values}
    schema = {"position": position_names, "routing": list(SCALARS[:3]), "instant": names}
    if mode == "heads":
        rows = read_heads(reader, directory, response)
        first = next(rows)
        layers = len(first["read_mass"])
        indices = list(range(layers // 4, layers - layers // 4)) if layer_scope == "middle" else list(range(layers))
        from itertools import chain
        heads, head_names = head_features(chain([first], rows), source_count, indices)
        values, names = np.column_stack((values, heads)), names + head_names
        views["heads"], schema["heads"] = values, names
        state, state_names = native_state_features(reader, directory, indices)
        views["native_state"] = np.column_stack((scalars, operator, state))
        schema["native_state"] = list(SCALARS) + operator_names + state_names
    views["temporal"], schema["temporal"] = temporal_features(values, names, window)
    return views, schema, {name: saved[name] for name in BASELINES}


def native_state_features(reader, directory, layer_indices):
    """Expose the saved kernel's full factor, reading vector and scale to a probe."""
    saved = reader.arrays(directory + "/state.npz", ("matrix", "reading", "response_scale"))
    matrix = saved["matrix"][:, layer_indices]
    reading = saved["reading"][:, layer_indices]
    scale = np.log1p(saved["response_scale"][:, layer_indices])
    count, _, rows, probes = matrix.shape
    heads = rows // (2 * len(ROLES))
    names = [f"factor/L{layer}/{channel}/H{head}/{role}/P{probe}" for layer in layer_indices
             for channel in ("residual", "ffn") for head in range(heads)
             for role in ROLES for probe in range(probes)]
    names += [f"sqrt_read/L{layer}/H{head}/{role}" for layer in layer_indices
              for head in range(heads) for role in ROLES]
    names += [f"log_response_scale/L{layer}" for layer in layer_indices]
    return np.column_stack((matrix.reshape(count, -1), reading.reshape(count, -1), scale)), names


def read_heads(reader, directory, response):
    fields = ("response_residual", "response_ffn", "response_total", "read_mass", "target", "query", "token_id")
    prompt = response["prompt_length"]
    for target, token in enumerate(response["token_ids"][prompt:]):
        path = f"{directory}/token_{target:06d}.npz"
        if not reader.exists(path):
            raise FileNotFoundError(f"{path}: --features heads requires the full capture directory, not a light ZIP")
        row = reader.arrays(path, fields)
        if (int(row["target"]), int(row["query"]), int(row["token_id"])) != (target, prompt + target - 1, token):
            raise ValueError(f"{response['id']}/{target}: native capture alignment differs")
        yield row


def load_dataset(path, mode="summary", layer_scope="middle", window=16):
    records, schema = [], None
    with closing(CaptureReader(path)) as reader:
        settings, protocol = reader.json("settings.json"), reader.json("protocol.json")
        operators = list(csv.DictReader(io.StringIO(reader.bytes("operators.csv").decode("utf-8"))))
        for index, response in enumerate(tqdm(settings["responses"], desc="readout features")):
            views, columns, baselines = answer_features(reader, response, index, operators, mode, layer_scope, window)
            if schema is not None and schema != columns:
                raise ValueError("Feature identities differ across answers")
            schema = columns
            records.append(dict(id=response["id"], source_id=response["source_id"], views=views,
                                baselines=baselines, response=response))
        # Feature construction above cannot inspect annotation labels or boundaries.
        annotations = reader.json("annotations.json")
    for record in records:
        response = record["response"]
        annotation = annotations[record["id"]]
        tokens = response["token_ids"][response["prompt_length"]:]
        if annotation["source_id"] != record["source_id"] or not np.array_equal(annotation["token_ids"], tokens):
            raise ValueError(f"{record['id']}: annotation identity differs")
        labels, onsets, firsts, valid = annotation_targets(annotation, len(tokens), record["id"])
        record.update(labels=labels, onsets=onsets, firsts=firsts, valid=valid,
                      target=np.arange(len(tokens)), response_length=len(tokens))
    return dict(settings=settings, protocol=protocol, records=records, schema=schema)


def attach_choice_features(dataset, path, window):
    """Optional historical choice observations, aligned by response/source and token IDs."""
    with closing(CaptureReader(path)) as reader:
        settings = reader.json("settings.json")
        lookup = {r["id"]: (index, r) for index, r in enumerate(settings["responses"])}
        for record in dataset["records"]:
            index, response = lookup[record["id"]]
            saved = reader.arrays(f"responses/{index:04d}/scores.npz")
            tokens = record["response"]["token_ids"][record["response"]["prompt_length"]:]
            if (settings["model"] != dataset["settings"]["model"]
                    or response["source_id"] != record["source_id"]
                    or response["token_ids"] != record["response"]["token_ids"]
                    or not np.array_equal(saved["token_id"], tokens)
                    or not np.array_equal(saved["target"], record["target"])):
                raise ValueError(f"{record['id']}: optional choice capture identity differs")
            choice = np.column_stack([saved[name] for name in CHOICE_FIELDS])
            names = dataset["schema"]["instant"] + [f"choice/{name}" for name in CHOICE_FIELDS]
            record["views"]["choice"] = np.column_stack((record["views"]["instant"], choice))
            record["views"]["choice_temporal"], temporal_names = temporal_features(record["views"]["choice"], names, window)
    dataset["schema"].update(choice=names, choice_temporal=temporal_names)
