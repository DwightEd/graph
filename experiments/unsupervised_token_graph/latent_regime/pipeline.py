"""Fit source-balanced natural regimes, then freeze causal TEST scores."""

import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..fixed_graph.reference import (
    apply_calibration,
    calibrate,
    source_quantiles,
    split_sources,
)
from ..offline_span.data import write_json
from .model import (
    RegimeModel,
    apply_standardization,
    fit_best,
    filtered_score,
    iid_score,
    standardize,
)


METHODS = ("iid", "filtered", "layer_mean_filtered")
CONTROLS = ("self_jump", "position")


def read_manifest(args):
    return json.loads(
        (args.output / "observations/manifest.json").read_text()
    )["records"]


def grouped(records):
    result = defaultdict(list)
    for row in records:
        result[(row["task"], row["generator"])].append(row)
    return result


def feature_view(observations, name):
    if name == "full":
        tokens, layers, heads, features = observations.shape
        return observations.reshape(tokens, layers, heads * features)
    if name == "layer_mean":
        return observations.mean(axis=2)
    raise ValueError("unknown latent-regime feature view: " + name)


def segment_bounds(coverage, minimum=2):
    starts = np.flatnonzero(coverage & ~np.r_[False, coverage[:-1]])
    ends = np.flatnonzero(coverage & ~np.r_[coverage[1:], False]) + 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= minimum
    ]


def load_sequences(root, records, view):
    sequences = []
    sequence_sources = []
    source_tokens = defaultdict(int)

    for row in records:
        with np.load(root / row["file"], allow_pickle=False) as saved:
            values = feature_view(saved["observations"], view)
            for start, end in segment_bounds(saved["coverage"]):
                sequence = values[start:end].astype(np.float32)
                sequences.append(sequence)
                sequence_sources.append(row["source_id"])
                source_tokens[row["source_id"]] += len(sequence)

    if not sequences:
        raise ValueError("No covered natural sequences for latent-regime fitting")

    weights = np.asarray([
        1. / source_tokens[source]
        for source in sequence_sources
    ], dtype=np.float64)
    return sequences, weights


def normalized_sequences(sequences, weights):
    center, scale = standardize(sequences, weights)
    normalized = [
        apply_standardization(values, center, scale).astype(np.float32)
        for values in sequences
    ]
    return normalized, center, scale


def save_regime(path, model, center, scale):
    np.savez_compressed(
        path,
        means=model.means,
        covariance=model.covariance,
        precision=model.precision,
        logdet=model.logdet,
        initial=model.initial,
        transition=model.transition,
        occupancy=model.occupancy,
        log_likelihood=np.asarray(model.log_likelihood),
        center=center,
        scale=scale,
    )


def load_regime(path):
    with np.load(path, allow_pickle=False) as saved:
        model = RegimeModel(
            means=saved["means"],
            covariance=saved["covariance"],
            precision=saved["precision"],
            logdet=saved["logdet"],
            initial=saved["initial"],
            transition=saved["transition"],
            occupancy=saved["occupancy"],
            log_likelihood=float(saved["log_likelihood"]),
        )
        center = saved["center"]
        scale = saved["scale"]
    return model, center, scale


def fit_view(root, records, view, path, args):
    raw, weights = load_sequences(root, records, view)
    sequences, center, scale = normalized_sequences(raw, weights)
    model = fit_best(sequences, weights, args)
    save_regime(path, model, center, scale)
    return model, center, scale


def score_view(values, coverage, model, center, scale, mode):
    scores = np.full(len(values), np.nan, np.float32)

    for start, end in segment_bounds(coverage):
        sequence = apply_standardization(
            values[start:end],
            center,
            scale,
        )
        if mode == "iid":
            current = iid_score(sequence, model)
        elif mode == "filtered":
            current = filtered_score(sequence, model)
        else:
            raise ValueError("unknown regime score mode: " + mode)

        scores[start:end] = current.astype(np.float32)

    return scores


def raw_scores(path, full, layer_mean):
    with np.load(path, allow_pickle=False) as saved:
        observations = saved["observations"]
        coverage = saved["coverage"]

        full_values = feature_view(observations, "full")
        mean_values = feature_view(observations, "layer_mean")
        scores = {
            "iid": score_view(
                full_values,
                coverage,
                full[0],
                full[1],
                full[2],
                "iid",
            ),
            "filtered": score_view(
                full_values,
                coverage,
                full[0],
                full[1],
                full[2],
                "filtered",
            ),
            "layer_mean_filtered": score_view(
                mean_values,
                coverage,
                layer_mean[0],
                layer_mean[1],
                layer_mean[2],
                "filtered",
            ),
        }

        self_values = observations[..., 0]
        jump = np.full(len(observations), np.nan, np.float32)
        delta = self_values[1:] - self_values[:-1]
        jump[1:] = np.sqrt(np.mean(delta * delta, axis=(1, 2)))
        scores["self_jump"] = jump

        length = len(observations)
        scores["position"] = (
            np.arange(length, dtype=np.float32)
            / max(1, length - 1)
        )

        return (
            scores,
            coverage,
            saved["token_ids"],
            saved["offsets"],
            int(saved["prompt_length"]),
        )


def calibration_scores(root, records, full, layer_mean):
    result = []
    for row in tqdm(
        records,
        desc="latent-regime calibration",
        unit="answer",
    ):
        scores, _, _, _, _ = raw_scores(
            root / row["file"],
            full,
            layer_mean,
        )
        result.append((row["source_id"], scores))
    return result


def fit_calibration(answer_scores, quantile):
    calibration = {}

    for method in METHODS:
        values = []
        sources = []
        for source_id, scores in answer_scores:
            valid = np.isfinite(scores[method])
            values.extend(scores[method][valid])
            sources.extend([source_id] * int(valid.sum()))

        calibration[method] = calibrate(
            np.asarray(values),
            np.asarray(sources),
            quantile,
        )

    return calibration


def high_transition_threshold(answer_scores):
    values = []
    sources = []

    for source_id, scores in answer_scores:
        valid = np.isfinite(scores["self_jump"])
        values.extend(scores["self_jump"][valid])
        sources.extend([source_id] * int(valid.sum()))

    return float(source_quantiles(
        np.asarray(values),
        np.asarray(sources),
        [.90],
    )[0])


def fit(args):
    records = read_manifest(args)
    roles = split_sources(records, args.seed)
    train = [row for row in records if row["split"] == "train"]

    root = args.output / "observations"
    output = args.output / "fit"
    output.mkdir(exist_ok=True)

    settings = dict(
        labels_used=False,
        state_orientation="state 1 = lower source-balanced TRAIN occupancy",
        sticky_prior=args.sticky_prior,
        groups={},
        roles=roles,
    )

    for number, (group, rows) in enumerate(sorted(grouped(train).items())):
        directory = output / str(number)
        complete = directory / "complete.json"
        key = "|".join(group)

        if args.resume and complete.exists():
            settings["groups"][key] = json.loads(
                (directory / "settings.json").read_text()
            )
            continue

        directory.mkdir(exist_ok=True)
        reference = [
            row for row in rows
            if roles[row["source_id"]] == "reference"
        ]
        calibration = [
            row for row in rows
            if roles[row["source_id"]] == "calibration"
        ]

        full_path = directory / "full.npz"
        if args.resume and full_path.exists():
            full = load_regime(full_path)
        else:
            full = fit_view(
                root,
                reference,
                "full",
                full_path,
                args,
            )

        layer_mean_path = directory / "layer_mean.npz"
        if args.resume and layer_mean_path.exists():
            layer_mean = load_regime(layer_mean_path)
        else:
            layer_mean = fit_view(
                root,
                reference,
                "layer_mean",
                layer_mean_path,
                args,
            )

        answer_scores = calibration_scores(
            root,
            calibration,
            full,
            layer_mean,
        )
        score_calibration = fit_calibration(
            answer_scores,
            args.quantile,
        )

        group_settings = dict(
            directory=str(number),
            reference_sources=sorted({
                row["source_id"] for row in reference
            }),
            calibration_sources=sorted({
                row["source_id"] for row in calibration
            }),
            score_calibration=score_calibration,
            high_transition=high_transition_threshold(answer_scores),
            full_occupancy=full[0].occupancy.tolist(),
            full_transition=full[0].transition.tolist(),
            full_log_likelihood=full[0].log_likelihood,
            layer_mean_occupancy=layer_mean[0].occupancy.tolist(),
            layer_mean_transition=layer_mean[0].transition.tolist(),
        )

        write_json(directory / "settings.json", group_settings)
        write_json(complete, dict(complete=True, labels_used=False))
        settings["groups"][key] = group_settings

    write_json(output / "settings.json", settings)
    write_json(output / "complete.json", dict(
        complete=True,
        labels_used=False,
    ))


def score_answer(path, row, full, layer_mean, calibration, destination):
    scores, coverage, token_ids, offsets, prompt_length = raw_scores(
        path,
        full,
        layer_mean,
    )

    arrays = {}
    for method in METHODS:
        arrays[method] = apply_calibration(
            scores[method],
            calibration[method],
        ).astype(np.float32)
        arrays[method + "_alarm"] = (
            arrays[method] > calibration[method]["threshold"]
        )

    for name in CONTROLS:
        arrays[name] = scores[name]

    arrays["coverage"] = coverage
    arrays["token_ids"] = token_ids
    arrays["offsets"] = offsets
    arrays["prompt_length"] = np.asarray(prompt_length)
    arrays["record_json"] = np.asarray(json.dumps(row))

    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)


def score(args):
    records = read_manifest(args)
    test = [row for row in records if row["split"] == "test"]
    settings = json.loads(
        (args.output / "fit/settings.json").read_text()
    )

    source = args.output / "observations"
    output = args.output / "predictions"
    output.mkdir(exist_ok=True)

    loaded = {}
    frozen = []

    for number, row in enumerate(
        tqdm(test, desc="latent-regime score", unit="answer")
    ):
        key = row["task"] + "|" + row["generator"]
        configured = settings["groups"][key]
        destination = output / f"{number:06d}.npz"

        if not (args.resume and destination.exists()):
            if key not in loaded:
                directory = args.output / "fit" / configured["directory"]
                loaded[key] = (
                    load_regime(directory / "full.npz"),
                    load_regime(directory / "layer_mean.npz"),
                )

            full, layer_mean = loaded[key]
            score_answer(
                source / row["file"],
                row,
                full,
                layer_mean,
                configured["score_calibration"],
                destination,
            )

        frozen.append(dict(row, file=destination.name))

    write_json(
        output / "freeze.json",
        dict(
            complete=True,
            labels_used=False,
            primary="filtered",
            methods=METHODS,
            controls=CONTROLS,
            records=frozen,
        ),
    )
