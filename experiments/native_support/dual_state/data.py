"""Read observations only; labels and supervised predictions never enter this module."""

from contextlib import closing
from itertools import chain

import numpy as np
from tqdm import tqdm

from ..choice_cache import CaptureReader
from ..readout.data import BASELINES, read_heads
from ..readout.features import role_responses


def head_routes(rows, source_count, layer_scope):
    first = next(rows)
    layers = len(first["read_mass"])
    indices = list(range(layers // 4, layers - layers // 4)) if layer_scope == "middle" else list(range(layers))
    routes, scales = [], []
    for row in chain([first], rows):
        total = role_responses(row["response_total"][indices].astype(float), source_count)
        energy = np.linalg.norm(total, axis=-1)
        scale = energy.sum(-1)
        contrast = energy[..., 1] + energy[..., 2] - energy[..., 0]
        route = np.divide(contrast, scale, out=np.zeros_like(scale), where=scale > 0)
        routes.append(route)
        scales.append(scale)
    columns = [f"L{layer}/H{head}" for layer in indices for head in range(first["read_mass"].shape[1])]
    shape = (len(routes), len(columns))
    return np.asarray(routes).reshape(shape), np.asarray(scales).reshape(shape), columns


def load_observations(path, mode, layer_scope):
    records, schema = [], None
    with closing(CaptureReader(path)) as reader:
        settings, protocol = reader.json("settings.json"), reader.json("protocol.json")
        for index, response in enumerate(tqdm(settings["responses"], desc="dual observations")):
            directory = f"responses/{index:04d}"
            saved = reader.arrays(directory + "/scores.npz", (*BASELINES, "target", "token_id"))
            tokens = np.asarray(response["token_ids"][response["prompt_length"]:])
            if not np.array_equal(saved["token_id"], tokens) or not np.array_equal(saved["target"], np.arange(len(tokens))):
                raise ValueError(f"{response['id']}: token alignment differs")
            if not all(np.isfinite(saved[name]).all() for name in BASELINES):
                raise ValueError(f"{response['id']}: incomplete baseline coverage")
            record = dict(id=response["id"], source_id=response["source_id"], response=response,
                          target=saved["target"], baselines={name: saved[name] for name in BASELINES},
                          response_length=len(tokens))
            if mode == "heads":
                source_count = len(reader.json(directory + "/sources.json")["blocks"])
                route, scale, names = head_routes(read_heads(reader, directory, response), source_count, layer_scope)
                if schema is not None and schema != names:
                    raise ValueError("Physical head identities differ across answers")
                if not np.isfinite(route).all() or not np.isfinite(scale).all():
                    raise ValueError(f"{response['id']}: nonfinite head observations")
                record.update(head_route=route, response_energy=scale)
                schema = names
            records.append(record)
    return dict(settings=settings, protocol=protocol, records=records, schema=schema or [])


def validate_reference(dataset, reference):
    sources = {r["source_id"] for r in dataset["records"]}
    other_sources = {r["source_id"] for r in reference["records"]}
    if sources & other_sources:
        raise ValueError("External reference and target sources must be disjoint")
    if dataset["settings"]["model"] != reference["settings"]["model"] or dataset["schema"] != reference["schema"]:
        raise ValueError("Reference model or physical head identities differ")
    for field in ("rank", "seed", "dtype", "roles", "channels"):
        if dataset["protocol"][field] != reference["protocol"][field]:
            raise ValueError(f"Reference measurement protocol differs: {field}")
