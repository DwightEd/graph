"""Read existing attention; remove special keys before measuring head features."""

import json

import numpy as np
from tqdm import tqdm

from ..fixed_graph.inputs import (
    identity_record, list_samples, sample_channels, validate_roster,
)
from ..fixed_graph.pipeline import check_input_roster
from ..offline_span.data import write_json


SIGNALS = ("self", "prompt")


def special_ids(args):
    if args.special_token_ids is not None:
        return sorted(set(args.special_token_ids))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    return sorted(tokenizer.all_special_ids)


def channel_values(channel, sample, excluded):
    values = np.full((sample.response_length, 2), np.nan, np.float32)
    masses = np.full((sample.response_length, 2), np.nan, np.float32)
    for row, query in enumerate(channel.queries):
        target = int(query + 1 - sample.prompt_length)
        if not 0 <= target < sample.response_length:
            continue
        keys, weights = channel.row(row)
        causal = keys <= query
        keys, weights = keys[causal], weights[causal]
        ordinary = ~np.isin(sample.token_ids[keys], excluded)
        retained = float(weights[ordinary].sum())
        masses[target] = (weights.sum(), retained)
        if retained <= 0:
            continue
        keys, weights = keys[ordinary], weights[ordinary] / retained
        values[target] = (weights[keys == query].sum(),
                          weights[keys < sample.prompt_length].sum())
    return values, masses


def extract(sample, channels, excluded):
    values, masses, layout = [], [], []
    for channel in channels:
        current, retained = channel_values(channel, sample, excluded)
        values.append(current)
        masses.append(retained)
        layout.append((channel.layer, channel.head))
    layout = np.asarray(layout, int).reshape(-1, 2)
    order = np.lexsort((layout[:, 1], layout[:, 0]))
    layout = layout[order]
    layers, heads = np.unique(layout[:, 0]), np.unique(layout[:, 1])
    expected = np.array([(layer, head) for layer in layers for head in heads])
    if not len(layout) or not np.array_equal(layout, expected):
        raise ValueError("Head geometry requires a nonempty layer x head grid")
    shape = (sample.response_length, len(layers), len(heads), 2)
    observations = np.stack(values, axis=1)[:, order].reshape(shape)
    retained = np.stack(masses, axis=1)[:, order].reshape(shape)
    covered = np.isfinite(observations).all(axis=(1, 2, 3))
    covered &= ~np.isin(sample.token_ids[sample.prompt_length:], excluded)
    return dict(observations=observations, retained_mass=retained,
                coverage=covered, channels=layout, token_ids=sample.token_ids,
                offsets=sample.offsets, prompt_length=sample.prompt_length)


def prepare(args, excluded):
    samples, indexes = list_samples(args)
    validate_roster(samples)
    root = args.output / "observations"
    root.mkdir(parents=True, exist_ok=True)
    check_input_roster(root, samples)
    records, layout = [], None
    for number, sample in enumerate(tqdm(samples, desc="head observations", unit="answer")):
        path = root / f"{number:06d}.npz"
        if not (args.resume and path.exists()):
            channels = sample_channels(sample, indexes[sample.split], args.layers, args.heads)
            arrays = extract(sample, channels, excluded)
            partial = path.with_suffix(".partial.npz")
            np.savez_compressed(partial, **arrays)
            partial.replace(path)
        with np.load(path, allow_pickle=False) as saved:
            current = saved["channels"].tolist()
        if layout is not None and current != layout:
            raise ValueError("Physical head layout changed between answers")
        layout = current
        records.append(identity_record(sample, path.name))
    write_json(root / "manifest.json", dict(records=records, channels=layout,
               signals=SIGNALS, special_token_ids=excluded, labels_used=False, complete=True))


def inspect(args, excluded):
    samples, indexes = list_samples(args)
    validate_roster(samples)
    for split in ("train", "test"):
        sample = next(item for item in samples if item.split == split)
        channels = sample_channels(sample, indexes[split], args.layers, args.heads)
        arrays = extract(sample, channels, excluded)
        print(json.dumps(dict(split=split, id=sample.response_id,
              shape=list(arrays["observations"].shape),
              coverage=float(arrays["coverage"].mean()),
              special_token_ids=excluded, labels_used=False)), flush=True)
