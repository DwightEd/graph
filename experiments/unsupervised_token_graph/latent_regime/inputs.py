"""Natural head observations: query-self and total prompt routing per physical head."""

import json

import numpy as np
from tqdm import tqdm

from ..fixed_graph.inputs import list_samples, sample_channels, validate_roster
from ..offline_span.data import write_json


FEATURES = ("self", "prompt")


def channel_observations(channel, sample):
    values = np.full((sample.response_length, len(FEATURES)), np.nan, np.float32)
    present = np.zeros(sample.response_length, dtype=bool)

    for row, query in enumerate(channel.queries):
        target = int(query + 1 - sample.prompt_length)
        if not 0 <= target < sample.response_length:
            continue

        keys, weights = channel.row(row)
        values[target, 0] = weights[keys == query].sum()
        values[target, 1] = weights[keys < sample.prompt_length].sum()
        present[target] = weights.sum() > 0

    return values, present


def observation_tensor(sample, index, args):
    blocks = []
    layout = []
    coverage = np.ones(sample.response_length, dtype=bool)

    for channel in sample_channels(sample, index, args.layers, args.heads):
        values, present = channel_observations(channel, sample)
        blocks.append(values)
        layout.append((channel.layer, channel.head))
        coverage &= present

    layout = np.asarray(layout, dtype=int)
    order = np.lexsort((layout[:, 1], layout[:, 0]))
    layout = layout[order]

    layers = np.unique(layout[:, 0])
    heads = np.unique(layout[:, 1])
    expected = np.array([(layer, head) for layer in layers for head in heads])
    if not np.array_equal(layout, expected):
        raise ValueError("Selected channels must form one layer×head grid")

    values = np.stack(blocks, axis=1)[:, order]
    values = values.reshape(
        sample.response_length,
        len(layers),
        len(heads),
        len(FEATURES),
    )
    return values, coverage, layout


def save_observation(path, sample, values, coverage, layout):
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        observations=values,
        coverage=coverage,
        channels=layout,
        token_ids=sample.token_ids,
        offsets=sample.offsets,
        prompt_length=np.asarray(sample.prompt_length),
    )
    partial.replace(path)


def prepare(args):
    samples, indexes = list_samples(args)
    validate_roster(samples)

    root = args.output / "observations"
    root.mkdir(parents=True, exist_ok=True)
    records = []

    for number, sample in enumerate(
        tqdm(samples, desc="latent-regime observations", unit="answer")
    ):
        path = root / f"{number:06d}.npz"
        if not (args.resume and path.exists()):
            values, coverage, layout = observation_tensor(
                sample,
                indexes[sample.split],
                args,
            )
            save_observation(path, sample, values, coverage, layout)

        records.append(dict(
            id=sample.response_id,
            source_id=sample.source_id,
            task=sample.task,
            generator=sample.generator,
            split=sample.split,
            file=path.name,
            prompt_length=sample.prompt_length,
            response_sha256=sample.response_sha256,
        ))

    write_json(
        root / "manifest.json",
        dict(
            records=records,
            features=FEATURES,
            labels_used=False,
        ),
    )


def inspect(args):
    samples, indexes = list_samples(args)
    validate_roster(samples)

    for split in ("train", "test"):
        sample = next(item for item in samples if item.split == split)
        values, coverage, layout = observation_tensor(
            sample,
            indexes[split],
            args,
        )
        print(json.dumps(dict(
            split=split,
            id=sample.response_id,
            shape=list(values.shape),
            coverage=float(coverage.mean()),
            channels=layout.tolist(),
            labels_used=False,
        )), flush=True)
