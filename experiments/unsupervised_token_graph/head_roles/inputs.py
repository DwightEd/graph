"""Per-head route measurements from existing caches, excluding special keys."""

import numpy as np
from tqdm import tqdm

from ..fixed_graph.inputs import identity_record, list_samples, sample_channels, validate_roster
from ..fixed_graph.pipeline import check_input_roster
from ..head_geometry.inputs import observation_summary
from ..offline_span.data import write_json


FEATURES = ("self", "prompt", "recent_history", "distant_history",
            "weighted_entropy", "largest_key", "ordinary_mass")


def route_features(keys, weights, query, prompt_length, recent_window):
    """No renormalization of route masses; m H(a/m) extends to zero at m=0."""
    mass = weights.sum()
    if mass == 0:
        return np.zeros(len(FEATURES))
    history = (keys >= prompt_length) & (keys < query)
    recent = history & (query - keys <= recent_window)
    positive = weights > 0
    weighted_entropy = -(weights[positive] * np.log(weights[positive] / mass)).sum()
    return (weights[keys == query].sum(), weights[keys < prompt_length].sum(),
            weights[recent].sum(), weights[history & ~recent].sum(),
            weighted_entropy, weights.max(), mass)


def channel_features(channel, sample, excluded, recent_window):
    result = np.full((sample.response_length, len(FEATURES)), np.nan, np.float32)
    masses = np.full(sample.response_length, np.nan, np.float32)
    for row, query in enumerate(channel.queries):
        target = int(query + 1 - sample.prompt_length)
        if not 0 <= target < sample.response_length:
            continue
        keys, weights = channel.row(row)
        ordinary = (keys <= query) & ~np.isin(sample.token_ids[keys], excluded)
        keys, weights = keys[ordinary], weights[ordinary].astype(np.float64)
        masses[target] = weights.sum()
        result[target] = route_features(keys, weights, query, sample.prompt_length, recent_window)
    return result, masses


def extract(sample, channels, excluded, recent_window):
    values, masses, layout = [], [], []
    for channel in channels:
        current, retained = channel_features(channel, sample, excluded, recent_window)
        values.append(current)
        masses.append(retained)
        layout.append((channel.layer, channel.head))
    layout = np.asarray(layout, int).reshape(-1, 2)
    order = np.lexsort((layout[:, 1], layout[:, 0]))
    layout = layout[order]
    layers, heads = np.unique(layout[:, 0]), np.unique(layout[:, 1])
    expected = np.array([(layer, head) for layer in layers for head in heads])
    if not len(layout) or not np.array_equal(layout, expected):
        raise ValueError("Expected a complete selected layer x head grid")
    shape = (sample.response_length, len(layers), len(heads), len(FEATURES))
    observations = np.stack(values, axis=1)[:, order].reshape(shape)
    coverage = np.isfinite(observations).all(axis=(1, 2, 3))
    coverage &= ~np.isin(sample.token_ids[sample.prompt_length:], excluded)
    retained = np.stack(masses, axis=1)[:, order]
    per_head_mass = retained.reshape(shape[:3])
    return dict(observations=observations, coverage=coverage, channels=layout,
                head_observed=np.isfinite(per_head_mass),
                conditional_defined=per_head_mass > 0,
                retained_mass=retained, token_ids=sample.token_ids,
                prompt_length=sample.prompt_length, offsets=sample.offsets)


def prepare(args, excluded):
    samples, indexes = list_samples(args)
    validate_roster(samples)
    root = args.output / "observations"
    root.mkdir(parents=True, exist_ok=True)
    check_input_roster(root, samples)
    records, summaries, layout = [], [], None
    for number, sample in enumerate(tqdm(samples, desc="ordinary head routes")):
        path = root / f"{number:06d}.npz"
        if not (args.resume and path.exists()):
            channels = sample_channels(sample, indexes[sample.split], args.layers, args.heads)
            arrays = extract(sample, channels, excluded, args.recent_window)
            arrays["observations"] = arrays["observations"][..., [FEATURES.index(name) for name in args.features]]
            partial = path.with_suffix(".partial.npz")
            np.savez_compressed(partial, **arrays)
            partial.replace(path)
        with np.load(path, allow_pickle=False) as saved:
            current = saved["channels"].tolist()
            summaries.append(dict(id=sample.response_id, split=sample.split,
                                  **observation_summary(saved)))
        if layout is not None and current != layout:
            raise ValueError("Physical head layout differs between answers")
        layout = current
        records.append(identity_record(sample, path.name))
    write_json(root / "manifest.json", dict(records=records, channels=layout,
               features=args.features, special_token_ids=excluded, labels_used=False, complete=True,
               representation="ordinary_key_submass", coverage_summary=summaries))
