"""CPU screening on all cached RAGTruth attention: no model weights."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.unsupervised_token_graph.span_audit.inputs import AuditInputs
from experiments.unsupervised_token_graph.span_audit.matching import match_controls
from experiments.unsupervised_token_graph.span_audit.readings import PredictionRows


PHASES = ("onset", "continuation")
METRICS = ("prompt_mass", "self_mass", "recent_history", "remote_history", "row_mass")


def row_features(answer, position, keys, weights, recent):
    prompt = answer.prompt_length
    query = prompt + position - 1
    prompt_mask = keys < prompt
    history = (keys >= prompt) & (keys < query)
    recent_mask = history & (keys >= query - recent)
    remote_mask = history & ~recent_mask
    return np.array([
        weights[prompt_mask].sum(),
        weights[keys == query].sum(),
        weights[recent_mask].sum(),
        weights[remote_mask].sum(),
        weights.sum(),
    ], dtype=float)


def phase_positions(pair):
    yield "onset", pair.error.start, pair.control.start
    for offset in range(1, pair.error.length):
        yield "continuation", pair.error.start + offset, pair.control.start + offset


def measure_pair(answer, pair, rows, recent):
    values = {phase: [] for phase in PHASES}
    for phase, error_position, control_position in phase_positions(pair):
        error_row = rows.read(error_position)
        control_row = rows.read(control_position)
        if error_row is None or control_row is None:
            continue
        error = row_features(answer, error_position, *error_row, recent)
        control = row_features(answer, control_position, *control_row, recent)
        values[phase].append(np.stack((error, control)))

    result = np.full((len(PHASES), 2, len(METRICS)), np.nan)
    for index, phase in enumerate(PHASES):
        if values[phase]:
            result[index] = np.mean(values[phase], axis=0)
    return result


def save_answer(path, answer, pairs, channels, readings):
    metadata = dict(
        response_id=answer.response_id,
        source_id=answer.source_id,
        task=answer.task,
        generator=answer.generator,
        pairs=[
            dict(error_start=pair.error.start, error_end=pair.error.end,
                 control_start=pair.control.start, control_end=pair.control.end)
            for pair in pairs
        ],
    )
    np.savez_compressed(
        path,
        channels=np.asarray(channels, dtype=int),
        readings=np.asarray(readings, dtype=np.float32),
        phases=np.asarray(PHASES),
        metrics=np.asarray(METRICS),
        metadata=np.asarray(json.dumps(metadata)),
    )


def screen_dataset(args):
    output = Path(args.output)
    samples = output / "screen_samples"
    samples.mkdir(parents=True, exist_ok=True)
    inputs = AuditInputs(
        args.train_cache, args.dataset, args.index, args.tokenizer, args.feature_root
    )
    ids = list(inputs.selected_ids("train", args.tasks))
    if args.limit:
        ids = ids[:args.limit]

    for response_id in tqdm(ids, desc="RAGTruth flow screen", unit="answer"):
        path = samples / (response_id + ".npz")
        if args.resume and path.exists():
            continue
        answer = inputs.load_answer(response_id)
        pairs = match_controls(
            answer, args.position_gap, args.repetition_gap, args.entropy_gap, args.window
        )
        channels = []
        readings = []
        if pairs:
            for channel in inputs.channels(answer, args.layers, args.heads):
                channels.append((channel.layer, channel.head))
                rows = PredictionRows(channel, answer.prompt_length)
                readings.append([
                    measure_pair(answer, pair, rows, args.recent_window)
                    for pair in pairs
                ])
        shape = (len(channels), len(pairs), len(PHASES), 2, len(METRICS))
        values = np.asarray(readings, dtype=np.float32).reshape(shape)
        save_answer(path, answer, pairs, channels, values)

    summarize_screen(output, args.screen_heads)


def source_mean(paths):
    total = None
    count = None
    channels = None
    metadata = None

    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            current_channels = saved["channels"]
            values = saved["readings"].astype(float)
        if not len(current_channels) or not values.size:
            continue
        if channels is None:
            channels = current_channels
            shape = (len(channels), len(PHASES), len(METRICS), 3)
            total = np.zeros(shape, dtype=float)
            count = np.zeros(shape[:-1], dtype=int)
            metadata = meta
        if not np.array_equal(channels, current_channels):
            raise ValueError("screen channels differ within one source/generator group")

        error = values[:, :, :, 0, :]
        control = values[:, :, :, 1, :]
        finite = np.isfinite(error) & np.isfinite(control)
        triplet = np.stack((error, control, error - control), axis=-1)
        total += np.where(finite[..., None], triplet, 0).sum(axis=1)
        count += finite.sum(axis=1)

    if channels is None:
        return None

    mean = np.divide(
        total,
        count[..., None],
        out=np.full_like(total, np.nan),
        where=count[..., None] > 0,
    )
    return metadata, channels, mean


def summarize_screen(output, head_count):
    groups = {}
    sources = {}
    for path in sorted((Path(output) / "screen_samples").glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
        key = (meta["source_id"], meta["task"], meta["generator"])
        groups.setdefault(key, []).append(path)
        sources.setdefault(meta["source_id"], []).append(path)

    accumulators = {}
    for (_, task, generator), paths in groups.items():
        result = source_mean(paths)
        if result is None:
            continue
        _, channels, mean = result

        for group_key in ((task, generator),):
            if group_key not in accumulators:
                accumulators[group_key] = dict(
                    channels=channels,
                    total=np.zeros_like(mean),
                    count=np.zeros(mean.shape[:-1], dtype=int),
                )
            bucket = accumulators[group_key]
            if not np.array_equal(bucket["channels"], channels):
                raise ValueError("screen channel layout differs across sources")
            finite = np.isfinite(mean[..., 2])
            bucket["total"] += np.where(np.isfinite(mean), mean, 0)
            bucket["count"] += finite

    for paths in sources.values():
        result = source_mean(paths)
        if result is None:
            continue
        _, channels, mean = result
        group_key = ("ALL", "ALL")
        if group_key not in accumulators:
            accumulators[group_key] = dict(
                channels=channels,
                total=np.zeros_like(mean),
                count=np.zeros(mean.shape[:-1], dtype=int),
            )
        bucket = accumulators[group_key]
        finite = np.isfinite(mean[..., 2])
        bucket["total"] += np.where(np.isfinite(mean), mean, 0)
        bucket["count"] += finite

    rows = []
    for (task, generator), bucket in accumulators.items():
        average = np.divide(
            bucket["total"],
            bucket["count"][..., None],
            out=np.full_like(bucket["total"], np.nan),
            where=bucket["count"][..., None] > 0,
        )
        for channel_index, (layer, head) in enumerate(bucket["channels"]):
            for phase_index, phase in enumerate(PHASES):
                for metric_index, metric in enumerate(METRICS):
                    error, control, difference = average[
                        channel_index, phase_index, metric_index
                    ]
                    if not np.isfinite(difference):
                        continue
                    rows.append(dict(
                        task=task,
                        generator=generator,
                        layer=int(layer),
                        head=int(head),
                        phase=phase,
                        metric=metric,
                        error_mean=float(error),
                        control_mean=float(control),
                        error_minus_control=float(difference),
                        sources=int(bucket["count"][channel_index, phase_index, metric_index]),
                    ))

    summary = pd.DataFrame(rows)
    summary.to_csv(Path(output) / "screen_head_effects.csv", index=False)

    score = summary[
        (summary.task == "ALL")
        & summary.metric.isin(
            ["prompt_mass", "self_mass", "recent_history", "remote_history"]
        )
    ].copy()
    score["size"] = score.error_minus_control.abs()
    selected = score.sort_values("size", ascending=False).drop_duplicates(
        ["layer", "head"]
    ).head(head_count)
    selected[[
        "layer", "head", "phase", "metric", "error_minus_control", "sources"
    ]].to_csv(Path(output) / "selected_heads.csv", index=False)
