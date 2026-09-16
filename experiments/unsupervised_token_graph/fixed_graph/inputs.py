"""复用现有样本、CSR与身份接口；不载入幻觉标签，不重新跑LLM。"""

from pathlib import Path

import numpy as np

from ..data import ResponseCache
from ..channels import iter_channels
from ..offline_span.data import load_samples, write_json
from .operators import FEATURES, VARIANTS, channel_arrays, embed_channel


def list_samples(args):
    rows = []
    indexes = {}
    tasks = () if args.tasks == ['all'] else args.tasks
    generators = () if args.generators == ['all'] else args.generators
    for split, path in [('train', args.train_cache), ('test', args.test_cache)]:
        samples, index = load_samples(path, args.index, split, tasks, generators,
                                      args.dataset, args.source_info)
        samples.sort(key=lambda sample: sample.response_id)
        if args.limit:
            samples = samples[:args.limit]
        rows.extend(samples)
        indexes[split] = index
    return rows, indexes


def sample_channels(sample, index, layers, heads):
    reader = ResponseCache(index=index)
    seen = set()
    for path in sample.cache_files:
        record = reader.load(path)
        if not np.array_equal(record.token_ids, sample.token_ids):
            raise ValueError(f'{sample.response_id}: channel token IDs differ')
        for channel in iter_channels(record, layers, heads):
            identity = (channel.layer, channel.head)
            if identity in seen:
                raise ValueError(f'duplicate physical channel {identity}')
            seen.add(identity)
            yield channel


def extract_sample(sample, channels, settings):
    shape = (sample.response_length, settings['dimensions'])
    embeddings = {name: np.zeros(shape, np.float32) for name in VARIANTS}
    coverage = np.ones(sample.response_length, bool)
    diagnostics = []
    channels_used = []
    node_summary = np.zeros((sample.response_length, len(FEATURES)), np.float64)
    for channel in channels:
        identity = (channel.layer, channel.head)
        features, present, masses, history = channel_arrays(channel, sample, settings['local_window'])
        projected, counts = embed_channel(features, present, history, sample.token_ids[sample.prompt_length:],
                                           identity, settings)
        for name in VARIANTS:
            embeddings[name] += projected[name]
        coverage &= present
        node_summary += features
        channels_used.append(identity)
        diagnostics.append([*identity, *counts, np.nanmean(masses)])
    if not channels_used:
        raise ValueError(f'{sample.response_id}: no selected channels')
    scale = np.sqrt(len(channels_used))
    for values in embeddings.values():
        values /= scale
        values[~coverage] = np.nan
    return embeddings, coverage, np.asarray(channels_used), np.asarray(diagnostics), node_summary / len(channels_used)


def save_sample(path, sample, extracted):
    embeddings, coverage, channels, diagnostics, attributes = extracted
    partial = path.with_suffix('.partial.npz')
    np.savez_compressed(partial, **embeddings, coverage=coverage, channels=channels,
                        diagnostics=diagnostics, mean_attributes=attributes,
                        token_ids=sample.token_ids, offsets=sample.offsets)
    partial.replace(path)


def identity_record(sample, relative):
    return dict(id=sample.response_id, source_id=sample.source_id, task=sample.task,
                generator=sample.generator, split=sample.split, file=relative,
                cache=sample.cache_files[0], response_sha256=sample.response_sha256,
                prompt_length=sample.prompt_length, tokens=sample.response_length)


def validate_roster(samples):
    """只在输入边界检查必要身份；禁止用回答编号冒充source。"""
    seen = {}
    for sample in samples:
        if not sample.source_id or sample.task in ('', 'unknown') or sample.generator in ('', 'unknown'):
            raise ValueError('Missing source/task/generator: use the original --dataset or existing --index')
        previous = seen.setdefault(sample.source_id, sample.split)
        if previous != sample.split:
            raise ValueError(f'source occurs in train and test: {sample.source_id}')
    identities = [(sample.split, sample.response_id) for sample in samples]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate answer identities')
