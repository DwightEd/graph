"""Learn unlabeled grounding-conditioned dynamics, then evaluate frozen surprise on RAGTruth."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from experiments.unsupervised_token_graph.span_audit.inputs import AuditInputs
from experiments.unsupervised_token_graph.span_audit.readings import PredictionRows
from .confirm import source_token_ids


GROUPS = ("source", "other_prompt", "history", "self")


def route_row(answer, position, keys, weights, source_lookup):
    prompt = answer.prompt_length
    query = prompt + position - 1
    prompt_key = keys < prompt
    source = np.zeros(len(keys), dtype=bool)
    source[prompt_key] = source_lookup[keys[prompt_key]]
    history = (keys >= prompt) & (keys < query)
    return np.array([
        weights[source].sum(),
        weights[prompt_key & ~source].sum(),
        weights[history].sum(),
        weights[keys == query].sum(),
    ], dtype=np.float64)


def answer_routes(inputs, tokenizer, answer):
    """Return [response, layer, head, source-group] attention masses."""
    source_record = inputs.sources[answer.source_id]
    source_positions = source_token_ids(
        tokenizer,
        answer.token_ids[:answer.prompt_length].tolist(),
        source_record,
    )
    source_lookup = np.zeros(answer.prompt_length, dtype=bool)
    source_lookup[source_positions] = True
    channels = []
    blocks = []
    for channel in inputs.channels(answer):
        rows = PredictionRows(channel, answer.prompt_length)
        values = np.full((len(answer.response_ids), len(GROUPS)), np.nan)
        for position in range(len(values)):
            row = rows.read(position)
            if row is not None:
                values[position] = route_row(
                    answer, position, *row, source_lookup
                )
        channels.append((channel.layer, channel.head))
        blocks.append(values)

    if not blocks:
        return None, 0
    channels = np.asarray(channels, dtype=int)
    order = np.lexsort((channels[:, 1], channels[:, 0]))
    channels = channels[order]
    layers = np.unique(channels[:, 0])
    heads = np.unique(channels[:, 1])
    if len(channels) != len(layers) * len(heads):
        raise ValueError("Grounding dynamics requires a rectangular layer×head cache")
    expected = np.array([(layer, head) for layer in layers for head in heads])
    if not np.array_equal(channels, expected):
        raise ValueError("Physical layer/head coordinates are not a complete grid")

    values = np.stack(blocks, axis=1)[:, order]
    values = values.reshape(len(values), len(layers), len(heads), len(GROUPS))
    return values, len(source_positions)


def transition_xy(routes):
    """Per layer: predict Δself from previous self and grounding-route changes."""
    previous_self = routes[:-1, :, :, 3]
    delta = routes[1:] - routes[:-1]
    target = delta[:, :, :, 3]
    predictors = np.concatenate(
        (
            previous_self,
            delta[:, :, :, 0],
            delta[:, :, :, 1],
            delta[:, :, :, 2],
        ),
        axis=2,
    )
    return predictors, target


class LinearAccumulator:
    def __init__(self, width, target):
        self.weight = 0.0
        self.x = np.zeros(width)
        self.y = np.zeros(target)
        self.xx = np.zeros((width, width))
        self.xy = np.zeros((width, target))

    def add(self, x, y, weight):
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[valid], y[valid]
        if not len(x):
            return
        w = np.full(len(x), weight, dtype=float)
        self.weight += w.sum()
        self.x += (x * w[:, None]).sum(0)
        self.y += (y * w[:, None]).sum(0)
        self.xx += (x.T * w) @ x
        self.xy += (x.T * w) @ y

    def fit(self, ridge):
        mean_x = self.x / self.weight
        mean_y = self.y / self.weight
        covariance = self.xx / self.weight - np.outer(mean_x, mean_x)
        cross = self.xy / self.weight - np.outer(mean_x, mean_y)
        diagonal = np.diag(covariance)
        scale = np.nanmean(diagonal[diagonal > 0]) if np.any(diagonal > 0) else 1.0
        covariance.flat[::len(covariance) + 1] += ridge * scale
        coefficient = np.linalg.solve(covariance, cross)
        intercept = mean_y - mean_x @ coefficient
        return coefficient, intercept


def fit_dynamics(inputs, tokenizer, ids, ridge):
    accumulators = None
    geometry = None
    coverage = []
    for response_id in tqdm(ids, desc="fit grounding dynamics", unit="answer"):
        answer = inputs.load_answer(response_id)
        routes, source_tokens = answer_routes(inputs, tokenizer, answer)
        coverage.append(source_tokens > 0)
        if routes is None or len(routes) < 2 or source_tokens == 0:
            continue
        x, y = transition_xy(routes)
        if accumulators is None:
            geometry = routes.shape[1:3]
            heads = geometry[1]
            accumulators = [
                LinearAccumulator(heads * 4, heads)
                for _ in range(geometry[0])
            ]
        if routes.shape[1:3] != geometry:
            raise ValueError("Mixed attention geometry")
        answer_weight = 1.0 / max(1, len(x))
        for layer in range(geometry[0]):
            accumulators[layer].add(x[:, layer], y[:, layer], answer_weight)

    models = [accumulator.fit(ridge) for accumulator in accumulators]
    return models, geometry, float(np.mean(coverage))


def residuals(routes, models):
    x, y = transition_xy(routes)
    blocks = []
    for layer, (coefficient, intercept) in enumerate(models):
        predicted = x[:, layer] @ coefficient + intercept
        blocks.append(y[:, layer] - predicted)
    return np.stack(blocks, axis=1)


class CovarianceAccumulator:
    def __init__(self, width):
        self.weight = 0.0
        self.sum = np.zeros(width)
        self.outer = np.zeros((width, width))

    def add(self, values, weight):
        valid = np.isfinite(values).all(axis=1)
        values = values[valid]
        if not len(values):
            return
        w = np.full(len(values), weight)
        self.weight += w.sum()
        self.sum += (values * w[:, None]).sum(0)
        self.outer += (values.T * w) @ values

    def fit(self, ridge):
        mean = self.sum / self.weight
        covariance = self.outer / self.weight - np.outer(mean, mean)
        scale = np.nanmean(np.diag(covariance))
        covariance.flat[::len(covariance) + 1] += ridge * max(scale, 1e-8)
        precision = np.linalg.inv(covariance)
        return mean, precision


def save_reference_checkpoint(path, raw, contrast, train_jumps, next_index, ids):
    payload = {
        "next_index": np.asarray(next_index, dtype=np.int64),
        "ids": np.asarray(ids),
        "train_jumps": np.asarray(train_jumps, dtype=np.float64),
    }
    for prefix, accumulators in (("raw", raw), ("contrast", contrast)):
        payload[prefix + "_weight"] = np.asarray(
            [item.weight for item in accumulators], dtype=np.float64
        )
        payload[prefix + "_sum"] = np.asarray(
            [item.sum for item in accumulators], dtype=np.float64
        )
        payload[prefix + "_outer"] = np.asarray(
            [item.outer for item in accumulators], dtype=np.float64
        )
    np.savez_compressed(path, **payload)


def load_reference_checkpoint(path, geometry, ids):
    with np.load(path, allow_pickle=False) as saved:
        saved_ids = saved["ids"].astype(str).tolist()
        if saved_ids != [str(value) for value in ids]:
            raise ValueError("Reference checkpoint TRAIN response IDs changed")
        next_index = int(saved["next_index"])
        train_jumps = saved["train_jumps"].astype(float).tolist()
        raw = [CovarianceAccumulator(geometry[1]) for _ in range(geometry[0])]
        contrast = [CovarianceAccumulator(geometry[1]) for _ in range(geometry[0])]
        for prefix, accumulators in (("raw", raw), ("contrast", contrast)):
            weights = saved[prefix + "_weight"]
            sums = saved[prefix + "_sum"]
            outers = saved[prefix + "_outer"]
            for layer, accumulator in enumerate(accumulators):
                accumulator.weight = float(weights[layer])
                accumulator.sum = sums[layer].astype(float)
                accumulator.outer = outers[layer].astype(float)
    return raw, contrast, train_jumps, next_index


def fit_residual_reference(
    inputs,
    tokenizer,
    ids,
    models,
    geometry,
    ridge,
    checkpoint_path=None,
    checkpoint_every=25,
    resume=False,
):
    raw = [CovarianceAccumulator(geometry[1]) for _ in range(geometry[0])]
    contrast = [CovarianceAccumulator(geometry[1]) for _ in range(geometry[0])]
    train_jumps = []
    start = 0

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        raw, contrast, train_jumps, start = load_reference_checkpoint(
            checkpoint_path, geometry, ids
        )
        print(
            f"grounding: resume reference at {start}/{len(ids)}",
            flush=True,
        )

    progress = tqdm(
        enumerate(ids[start:], start=start),
        total=len(ids),
        initial=start,
        desc="fit residual reference",
        unit="answer",
    )
    for index, response_id in progress:
        answer = inputs.load_answer(response_id)
        routes, source_tokens = answer_routes(inputs, tokenizer, answer)
        if routes is None or len(routes) < 2 or source_tokens == 0:
            if (
                checkpoint_path is not None
                and (index + 1) % checkpoint_every == 0
            ):
                save_reference_checkpoint(
                    checkpoint_path, raw, contrast, train_jumps, index + 1, ids
                )
            continue
        values = residuals(routes, models)
        delta_self = routes[1:, :, :, 3] - routes[:-1, :, :, 3]
        train_jumps.extend(
            np.sqrt(np.nanmean(delta_self * delta_self, axis=(1, 2))).tolist()
        )
        answer_weight = 1.0 / max(1, len(values))
        centered = values - values.mean(axis=2, keepdims=True)
        for layer in range(geometry[0]):
            raw[layer].add(values[:, layer], answer_weight)
            contrast[layer].add(centered[:, layer], answer_weight)

        if (
            checkpoint_path is not None
            and (index + 1) % checkpoint_every == 0
        ):
            save_reference_checkpoint(
                checkpoint_path, raw, contrast, train_jumps, index + 1, ids
            )

    if checkpoint_path is not None:
        save_reference_checkpoint(
            checkpoint_path, raw, contrast, train_jumps, len(ids), ids
        )

    high_jump = float(np.nanquantile(np.asarray(train_jumps), .90))
    return (
        [item.fit(ridge) for item in raw],
        [item.fit(ridge) for item in contrast],
        high_jump,
    )


def mahalanobis(values, reference):
    scores = np.zeros(len(values))
    used = np.zeros(len(values))
    for layer, (mean, precision) in enumerate(reference):
        residual = values[:, layer] - mean
        valid = np.isfinite(residual).all(axis=1)
        scores[valid] += np.einsum(
            "ni,ij,nj->n", residual[valid], precision, residual[valid]
        ) / residual.shape[1]
        used[valid] += 1
    return np.divide(scores, used, out=np.full(len(values), np.nan), where=used > 0)


def sentence_start(answer):
    flags = np.zeros(len(answer.response_ids), dtype=bool)
    if len(flags):
        flags[0] = True
    tokens = [answer.text[a:b] for a, b in answer.offsets]
    terminal = (".", "?", "!", "。", "？", "！")
    for position in range(1, len(tokens)):
        previous = tokens[position - 1].rstrip()
        flags[position] = (
            previous.endswith(terminal)
            or "\n" in previous
            or tokens[position].startswith("\n")
        )
    return flags


def score_answer(answer, routes, models, raw_reference, contrast_reference):
    values = residuals(routes, models)
    contrast = values - values.mean(axis=2, keepdims=True)
    raw_score = mahalanobis(values, raw_reference)
    contrast_score = mahalanobis(contrast, contrast_reference)

    delta = routes[1:] - routes[:-1]
    source_gain = np.nanmean(delta[:, :, :, 0], axis=(1, 2))
    history_gain = np.nanmean(delta[:, :, :, 2], axis=(1, 2))
    jump = np.sqrt(np.nanmean(delta[:, :, :, 3] ** 2, axis=(1, 2)))

    length = len(answer.response_ids)
    frame = pd.DataFrame(dict(
        id=answer.response_id,
        source_id=answer.source_id,
        task=answer.task,
        generator=answer.generator,
        token=np.arange(1, length),
        gold=answer.error_mask[1:].astype(int),
        previous_gold=answer.error_mask[:-1].astype(int),
        sentence_start=sentence_start(answer)[1:],
        raw_surprise=raw_score,
        head_contrast_surprise=contrast_score,
        source_gain=source_gain,
        history_gain=history_gain,
        self_jump=jump,
    ))
    frame["grounding_balance"] = frame.source_gain - frame.history_gain
    return frame


def safe_metric(labels, scores):
    valid = np.isfinite(scores)
    labels, scores = np.asarray(labels)[valid], np.asarray(scores)[valid]
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def evaluate(table, high_jump):
    scopes = {
        "all": np.ones(len(table), bool),
        "previous_gold_0": table.previous_gold.to_numpy() == 0,
        "previous_gold_0_sentence_start": (
            (table.previous_gold.to_numpy() == 0)
            & table.sentence_start.to_numpy(bool)
        ),
        "previous_gold_0_high_transition": (
            (table.previous_gold.to_numpy() == 0)
            & (table.self_jump.to_numpy() >= high_jump)
        ),
        "previous_gold_1": table.previous_gold.to_numpy() == 1,
    }
    score_names = (
        "raw_surprise",
        "head_contrast_surprise",
        "source_gain",
        "history_gain",
        "grounding_balance",
        "self_jump",
    )
    rows = []
    for scope, mask in scopes.items():
        for score in score_names:
            auroc, ap = safe_metric(
                table.gold.to_numpy()[mask],
                table[score].to_numpy()[mask],
            )
            rows.append(dict(
                scope=scope,
                score=score,
                tokens=int(mask.sum()),
                positives=int(table.gold.to_numpy()[mask].sum()),
                auroc=auroc,
                ap=ap,
            ))
    return pd.DataFrame(rows)


def write_status(output, stage, **extra):
    payload = dict(stage=stage, **extra)
    (output / "status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("grounding:", stage, extra, flush=True)


def save_dynamics(path, models, geometry, source_coverage):
    np.savez_compressed(
        path,
        geometry=np.asarray(geometry),
        source_coverage=float(source_coverage),
        regression=np.asarray([item[0] for item in models]),
        intercept=np.asarray([item[1] for item in models]),
    )


def load_dynamics(path):
    with np.load(path, allow_pickle=False) as saved:
        geometry = tuple(saved["geometry"].astype(int))
        models = list(zip(saved["regression"], saved["intercept"]))
        source_coverage = float(saved["source_coverage"])
    return models, geometry, source_coverage


def save_reference(path, raw_reference, contrast_reference, high_jump):
    np.savez_compressed(
        path,
        high_jump_train_q90=float(high_jump),
        raw_mean=np.asarray([item[0] for item in raw_reference]),
        raw_precision=np.asarray([item[1] for item in raw_reference]),
        contrast_mean=np.asarray([item[0] for item in contrast_reference]),
        contrast_precision=np.asarray([item[1] for item in contrast_reference]),
    )


def load_reference(path):
    with np.load(path, allow_pickle=False) as saved:
        raw_reference = list(zip(saved["raw_mean"], saved["raw_precision"]))
        contrast_reference = list(zip(
            saved["contrast_mean"], saved["contrast_precision"]
        ))
        high_jump = float(saved["high_jump_train_q90"])
    return raw_reference, contrast_reference, high_jump


def run_grounding_dynamics(args):
    from transformers import AutoTokenizer

    output = Path(args.output) / "grounding_dynamics"
    output.mkdir(parents=True, exist_ok=True)
    write_status(output, "initializing")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, use_fast=True
    )
    train = AuditInputs(
        args.train_cache, args.dataset, args.index, args.tokenizer, args.feature_root
    )
    test = AuditInputs(
        args.test_cache, args.dataset, args.index, args.tokenizer, args.feature_root
    )
    train_ids = list(train.selected_ids("train", args.tasks))
    test_ids = list(test.selected_ids("test", args.tasks))
    if args.limit:
        train_ids, test_ids = train_ids[:args.limit], test_ids[:args.limit]

    dynamics_path = output / "dynamics.npz"
    reference_path = output / "reference.npz"
    reference_checkpoint = output / "reference_checkpoint.npz"
    if args.resume and dynamics_path.exists():
        models, geometry, source_coverage = load_dynamics(dynamics_path)
        write_status(output, "dynamics_loaded", train_answers=len(train_ids))
    else:
        write_status(output, "fit_dynamics", train_answers=len(train_ids))
        models, geometry, source_coverage = fit_dynamics(
            train, tokenizer, train_ids, args.grounding_ridge
        )
        if geometry is None:
            raise ValueError(
                "No TRAIN answers had exact source_info coverage and usable attention rows"
            )
        save_dynamics(dynamics_path, models, geometry, source_coverage)
        write_status(
            output, "dynamics_saved",
            source_coverage=source_coverage,
            geometry=list(geometry),
        )

    if args.resume and reference_path.exists():
        raw_reference, contrast_reference, high_jump = load_reference(reference_path)
        write_status(output, "reference_loaded")
    else:
        write_status(output, "fit_reference", train_answers=len(train_ids))
        raw_reference, contrast_reference, high_jump = fit_residual_reference(
            train,
            tokenizer,
            train_ids,
            models,
            geometry,
            args.grounding_ridge,
            checkpoint_path=reference_checkpoint,
            checkpoint_every=args.grounding_checkpoint_every,
            resume=args.resume,
        )
        save_reference(reference_path, raw_reference, contrast_reference, high_jump)
        reference_checkpoint.unlink(missing_ok=True)
        write_status(output, "reference_saved", high_jump=high_jump)

    score_dir = output / "test_answers"
    score_dir.mkdir(exist_ok=True)
    write_status(output, "score_test", test_answers=len(test_ids))
    files = []
    for response_id in tqdm(test_ids, desc="score grounding dynamics", unit="answer"):
        path = score_dir / f"{response_id}.csv.gz"
        if args.resume and path.exists():
            files.append(path)
            continue
        answer = test.load_answer(response_id)
        routes, source_tokens = answer_routes(test, tokenizer, answer)
        if routes is None or len(routes) < 2 or source_tokens == 0:
            continue
        frame = score_answer(
            answer, routes, models, raw_reference, contrast_reference
        )
        frame.to_csv(path, index=False)
        files.append(path)

    if not files:
        raise ValueError(
            "No TEST answers had exact source_info coverage and usable attention rows"
        )
    table = pd.concat(
        [pd.read_csv(path, dtype={"id": str, "source_id": str}) for path in files],
        ignore_index=True,
    )
    metrics = evaluate(table, high_jump)

    table.to_csv(output / "token_scores.csv.gz", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    np.savez_compressed(
        output / "model.npz",
        geometry=np.asarray(geometry),
        source_coverage=source_coverage,
        high_jump_train_q90=high_jump,
        regression=np.asarray([item[0] for item in models]),
        intercept=np.asarray([item[1] for item in models]),
        raw_mean=np.asarray([item[0] for item in raw_reference]),
        raw_precision=np.asarray([item[1] for item in raw_reference]),
        contrast_mean=np.asarray([item[0] for item in contrast_reference]),
        contrast_precision=np.asarray([item[1] for item in contrast_reference]),
    )
    (output / "protocol.json").write_text(json.dumps(dict(
        training="all TRAIN tokens, source-balanced by answer; no hallucination labels",
        target="per-layer Δself-routing vector",
        predictors=["previous self", "Δsource", "Δother_prompt", "Δhistory"],
        scores=["raw residual Mahalanobis", "within-layer head-contrast residual Mahalanobis"],
        source_coverage=source_coverage,
        high_transition_threshold=dict(metric="self_jump", train_quantile=.90, value=high_jump),
        note="source means exact source_info text located in the saved prompt; missing coverage is skipped",
    ), indent=2), encoding="utf-8")

    write_status(
        output, "complete",
        test_answers=len(files),
        source_coverage=source_coverage,
    )
    print(metrics.to_string(index=False), flush=True)
    print("Results:", output, flush=True)
