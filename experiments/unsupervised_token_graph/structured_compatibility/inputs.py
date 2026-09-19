"""Extract head-resolved source/history/self routes from existing attention caches."""

from collections import defaultdict

import numpy as np

from ..fixed_graph.inputs import list_samples, sample_channels, validate_roster
from ..evaluation_data import read_sources
from .source_tokens import source_token_positions


VIEWS = ("source", "other_prompt", "recent_history", "remote_history", "self")


def load_inputs(args):
    samples, indexes = list_samples(args)
    validate_roster(samples)
    sources, path = read_sources(args.dataset / "response.jsonl", args.source_info)
    return samples, indexes, sources, path


def row_routes(keys, weights, prompt_length, query, source_lookup, recent_window):
    prompt = keys < prompt_length
    source = np.zeros(len(keys), dtype=bool)
    source[prompt] = source_lookup[keys[prompt]]

    history = (keys >= prompt_length) & (keys < query)
    recent = history & (keys >= max(prompt_length, query - recent_window))
    remote = history & ~recent

    return np.array([
        weights[source].sum(),
        weights[prompt & ~source].sum(),
        weights[recent].sum(),
        weights[remote].sum(),
        weights[keys == query].sum(),
    ], dtype=np.float32)


def channel_routes(channel, sample, source_lookup, recent_window):
    values = np.full((sample.response_length, len(VIEWS)), np.nan, np.float32)
    present = np.zeros(sample.response_length, dtype=bool)

    for row, query in enumerate(channel.queries):
        target = int(query + 1 - sample.prompt_length)
        if not 0 <= target < sample.response_length:
            continue

        keys, weights = channel.row(row)
        values[target] = row_routes(
            keys,
            weights,
            sample.prompt_length,
            int(query),
            source_lookup,
            recent_window,
        )
        present[target] = weights.sum() > 0

    return values, present


def route_tensor(sample, index, tokenizer, source_record, args):
    source_positions = source_token_positions(
        tokenizer,
        sample.token_ids[:sample.prompt_length].tolist(),
        source_record,
    )
    source_lookup = np.zeros(sample.prompt_length, dtype=bool)
    source_lookup[source_positions] = True

    blocks = []
    layout = []
    coverage = np.ones(sample.response_length, dtype=bool)
    for channel in sample_channels(sample, index, args.layers, args.heads):
        values, present = channel_routes(
            channel,
            sample,
            source_lookup,
            args.recent_window,
        )
        blocks.append(values)
        layout.append((channel.layer, channel.head))
        coverage &= present

    if not layout:
        raise ValueError(f"{sample.response_id}: no selected attention channels")

    layout = np.asarray(layout, dtype=int)
    order = np.lexsort((layout[:, 1], layout[:, 0]))
    layout = layout[order]
    layers = np.unique(layout[:, 0])
    heads = np.unique(layout[:, 1])
    expected = np.array([(layer, head) for layer in layers for head in heads])
    if not np.array_equal(layout, expected):
        raise ValueError("Selected attention channels must form one layer×head grid")

    routes = np.stack(blocks, axis=1)[:, order]
    routes = routes.reshape(
        sample.response_length,
        len(layers),
        len(heads),
        len(VIEWS),
    )
    return routes, coverage, layout, len(source_positions)


def group_samples(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample.task, sample.generator)].append(sample)
    return groups


def roster_records(samples):
    return [
        dict(
            id=sample.response_id,
            source_id=sample.source_id,
            task=sample.task,
            generator=sample.generator,
            split=sample.split,
        )
        for sample in samples
    ]
