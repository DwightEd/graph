"""Source-balanced self-supervised fitting and unlabeled calibration."""

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..fixed_graph.reference import calibrate, source_quantiles
from .inputs import route_tensor
from .model import (
    CLASS_NAMES,
    StructuredCompatibility,
    compatibility_scores,
    corruption_batch,
)


SCORES = (
    "compatibility",
    "head_identity",
    "source_head",
    "source_time",
    "previous_time",
)


def valid_positions(coverage):
    previous = np.r_[False, coverage[:-1]]
    return np.flatnonzero(coverage & previous)


def donor(position, candidates, random, block=8):
    local = candidates[
        (candidates // block == position // block)
        & (candidates != position)
    ]
    pool = local if len(local) else candidates[candidates != position]
    return int(random.choice(pool))


def gather_examples(samples, index, tokenizer, sources, roles, args):
    by_source = defaultdict(list)
    for sample in samples:
        if roles.get(sample.source_id) == "reference":
            by_source[sample.source_id].append(sample)

    random = np.random.default_rng(args.seed)
    ordered_sources = sorted(by_source)
    random.shuffle(ordered_sources)

    current = []
    previous = []
    source_donor = []
    previous_donor = []
    example_sources = []
    layout = None

    for source_id in ordered_sources:
        if len(current) >= args.train_budget:
            break

        remaining = min(
            args.tokens_per_source,
            args.train_budget - len(current),
        )
        source_examples = []

        for sample in by_source[source_id]:
            routes, coverage, channels, source_tokens = route_tensor(
                sample,
                index,
                tokenizer,
                sources[source_id],
                args,
            )
            if source_tokens == 0:
                continue

            positions = valid_positions(coverage)
            if len(positions) < 3:
                continue

            count = min(
                remaining - len(source_examples),
                len(positions),
            )
            chosen = random.choice(positions, count, replace=False)
            previous_candidates = positions - 1

            for position in chosen:
                source_position = donor(
                    int(position),
                    positions,
                    random,
                )
                previous_position = donor(
                    int(position - 1),
                    previous_candidates,
                    random,
                )
                source_examples.append((
                    routes[position],
                    routes[position - 1],
                    routes[source_position],
                    routes[previous_position],
                ))

            if layout is None:
                layout = channels
            elif not np.array_equal(layout, channels):
                raise ValueError("Attention layout changed within one fit")

            if len(source_examples) >= remaining:
                break

        for values in source_examples[:remaining]:
            current.append(values[0])
            previous.append(values[1])
            source_donor.append(values[2])
            previous_donor.append(values[3])
            example_sources.append(source_id)

    if not current:
        raise ValueError("No source-covered transitions available for training")

    return dict(
        current=np.asarray(current, np.float32),
        previous=np.asarray(previous, np.float32),
        source_donor=np.asarray(source_donor, np.float32),
        previous_donor=np.asarray(previous_donor, np.float32),
        source_id=np.asarray(example_sources),
        channels=layout,
    )


def normalization(examples):
    values = examples["current"]
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-4] = 1.
    return center.astype(np.float32), scale.astype(np.float32)


def normalize(values, center, scale):
    return (values - center) / scale


def fit_model(examples, center, scale, args):
    torch.manual_seed(args.seed)
    current = examples["current"]
    previous = examples["previous"]
    source_donor = examples["source_donor"]
    previous_donor = examples["previous_donor"]

    layers, heads, views = current.shape[1:]
    model = StructuredCompatibility(
        layers,
        heads,
        views,
        args.hidden,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    center_tensor = torch.from_numpy(center).to(args.device)
    scale_tensor = torch.from_numpy(scale).to(args.device)

    losses = []
    random = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = random.permutation(len(current))
        epoch_loss = 0.
        examples_seen = 0

        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            batch = [
                torch.from_numpy(values[rows]).to(args.device)
                for values in (
                    current,
                    previous,
                    source_donor,
                    previous_donor,
                )
            ]
            corrupted_current, corrupted_previous, labels = corruption_batch(
                *batch,
                center_tensor,
                scale_tensor,
            )

            logits = model(corrupted_current, corrupted_previous)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss) * len(labels)
            examples_seen += len(labels)

        losses.append(epoch_loss / examples_seen)
        print(
            f"compatibility epoch={epoch + 1}/{args.epochs} "
            f"loss={losses[-1]:.5f}",
            flush=True,
        )

    return model, losses


def score_routes(model, routes, coverage, center, scale, device, batch_size):
    positions = valid_positions(coverage)
    result = {
        name: np.full(len(routes), np.nan, np.float32)
        for name in SCORES
    }
    if not len(positions):
        return result

    current = normalize(routes[positions], center, scale)
    previous = normalize(routes[positions - 1], center, scale)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            stop = start + batch_size
            current_batch = torch.from_numpy(
                current[start:stop]
            ).to(device)
            previous_batch = torch.from_numpy(
                previous[start:stop]
            ).to(device)

            logits = model(current_batch, previous_batch)
            overall, components = compatibility_scores(logits)
            target = positions[start:stop]
            result["compatibility"][target] = overall.cpu().numpy()
            for index, name in enumerate(SCORES[1:]):
                result[name][target] = components[:, index].cpu().numpy()

    return result


def causal_sticky(values, rho):
    output = np.full(len(values), np.nan, np.float32)
    state = np.nan
    for index, value in enumerate(values):
        if not np.isfinite(value):
            state = np.nan
            continue
        if not np.isfinite(state):
            state = float(value)
        else:
            state = rho * state + (1 - rho) * float(value)
        output[index] = state
    return output


def estimate_rho(sequences):
    correlations = []
    for values in sequences:
        current = np.asarray(values[1:])
        previous = np.asarray(values[:-1])
        valid = np.isfinite(current) & np.isfinite(previous)
        if valid.sum() < 4:
            continue

        left = current[valid]
        right = previous[valid]
        if left.std() < 1e-6 or right.std() < 1e-6:
            continue
        correlations.append(np.corrcoef(left, right)[0, 1])

    if not correlations:
        return 0.
    return float(np.clip(np.median(correlations), 0., .95))


def fit_score_calibration(answer_scores, quantile):
    rows = {name: [] for name in SCORES}
    source_ids = {name: [] for name in SCORES}

    for source_id, scores in answer_scores:
        for name in SCORES:
            valid = np.isfinite(scores[name])
            rows[name].extend(scores[name][valid])
            source_ids[name].extend([source_id] * int(valid.sum()))

    calibration = {}
    for name in SCORES:
        values = np.asarray(rows[name], float)
        sources = np.asarray(source_ids[name])
        calibration[name] = calibrate(values, sources, quantile)
    return calibration


def standardize_scores(scores, calibration):
    result = {}
    for name in SCORES:
        values = scores[name]
        setting = calibration[name]
        result[name] = (
            (values - setting["center"]) / setting["scale"]
        ).astype(np.float32)
    return result


def fit_sticky_calibration(answer_scores, calibration, quantile):
    standardized = [
        (source_id, standardize_scores(scores, calibration))
        for source_id, scores in answer_scores
    ]
    rho = estimate_rho([
        scores["compatibility"]
        for _, scores in standardized
    ])

    values = []
    sources = []
    for source_id, scores in standardized:
        sticky = causal_sticky(scores["compatibility"], rho)
        valid = np.isfinite(sticky)
        values.extend(sticky[valid])
        sources.extend([source_id] * int(valid.sum()))

    sticky = calibrate(
        np.asarray(values),
        np.asarray(sources),
        quantile,
    )
    return rho, sticky


def self_jump(routes):
    delta = routes[1:, :, :, 4] - routes[:-1, :, :, 4]
    values = np.sqrt(np.mean(delta * delta, axis=(1, 2)))
    return np.r_[np.nan, values].astype(np.float32)


def high_transition_threshold(answer_routes, source_ids, quantile=.9):
    values = []
    sources = []
    for routes, source_id in zip(answer_routes, source_ids):
        jump = self_jump(routes)
        valid = np.isfinite(jump)
        values.extend(jump[valid])
        sources.extend([source_id] * int(valid.sum()))

    return float(source_quantiles(
        np.asarray(values),
        np.asarray(sources),
        [quantile],
    )[0])
