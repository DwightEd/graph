"""Fit corruption compatibility on TRAIN, freeze scores, then evaluate separately."""

import json

import numpy as np
import torch
from tqdm import tqdm

from ..fixed_graph.reference import split_sources
from ..offline_span.data import write_json
from .inputs import group_samples, load_inputs, roster_records, route_tensor
from .model import CLASS_NAMES, StructuredCompatibility
from .training import (
    SCORES,
    causal_sticky,
    fit_model,
    fit_score_calibration,
    fit_sticky_calibration,
    gather_examples,
    high_transition_threshold,
    normalization,
    score_routes,
    self_jump,
    standardize_scores,
)


METHODS = (*SCORES, "sticky")
CONTROLS = ("self_jump", "position")


def group_name(group):
    return "|".join(group)


def save_model(
    path,
    model,
    center,
    scale,
    channels,
    losses,
    train_examples,
    args,
):
    torch.save(
        dict(
            state_dict=model.state_dict(),
            center=center,
            scale=scale,
            channels=channels,
            classes=CLASS_NAMES,
            hidden=args.hidden,
            losses=losses,
            train_examples=int(train_examples),
        ),
        path,
    )


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    center = saved["center"]
    layers, heads, views = center.shape
    model = StructuredCompatibility(
        layers,
        heads,
        views,
        saved["hidden"],
    )
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    return model, saved


def calibration_samples(samples, roles):
    return [
        sample
        for sample in samples
        if roles.get(sample.source_id) == "calibration"
    ]


def score_calibration_samples(samples, index, tokenizer, sources, model, saved, args):
    answers = []
    route_blocks = []
    source_ids = []

    for sample in tqdm(samples, desc="compatibility calibration", unit="answer"):
        routes, coverage, channels, source_tokens = route_tensor(
            sample,
            index,
            tokenizer,
            sources[sample.source_id],
            args,
        )
        if source_tokens == 0:
            continue
        if not np.array_equal(saved["channels"], channels):
            raise ValueError("Calibration attention layout differs from fit")

        scores = score_routes(
            model,
            routes,
            coverage,
            saved["center"],
            saved["scale"],
            args.device,
            args.batch_size,
        )
        answers.append((sample.source_id, scores))
        route_blocks.append(routes)
        source_ids.append(sample.source_id)

    if not answers:
        raise ValueError("No calibration answers have source-covered transitions")
    return answers, route_blocks, source_ids


def fit(args):
    from transformers import AutoTokenizer

    samples, indexes, sources, source_path = load_inputs(args)
    train = [sample for sample in samples if sample.split == "train"]
    roles = split_sources(roster_records(samples), args.seed)
    groups = group_samples(train)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        use_fast=True,
    )
    root = args.output / "fit"
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(
        labels_used=False,
        source_info=source_path,
        roles=roles,
        groups={},
        corruption_classes=CLASS_NAMES,
    )

    for number, (group, group_samples_) in enumerate(sorted(groups.items())):
        directory = root / str(number)
        complete = directory / "complete.json"
        if args.resume and complete.exists():
            settings["groups"][group_name(group)] = json.loads(
                (directory / "settings.json").read_text()
            )
            continue

        directory.mkdir(exist_ok=True)
        model_path = directory / "model.pt"
        if args.resume and model_path.exists():
            model, saved = load_model(model_path, args.device)
            losses = saved["losses"]
            train_examples = int(saved["train_examples"])
        else:
            examples = gather_examples(
                group_samples_,
                indexes["train"],
                tokenizer,
                sources,
                roles,
                args,
            )
            center, scale = normalization(examples)
            model, losses = fit_model(examples, center, scale, args)
            train_examples = len(examples["current"])
            save_model(
                model_path,
                model,
                center,
                scale,
                examples["channels"],
                losses,
                train_examples,
                args,
            )
            _, saved = load_model(model_path, args.device)
        held_out = calibration_samples(group_samples_, roles)
        answer_scores, route_blocks, source_ids = score_calibration_samples(
            held_out,
            indexes["train"],
            tokenizer,
            sources,
            model,
            saved,
            args,
        )
        score_calibration = fit_score_calibration(
            answer_scores,
            args.quantile,
        )
        rho, sticky_calibration = fit_sticky_calibration(
            answer_scores,
            score_calibration,
            args.quantile,
        )
        high_transition = high_transition_threshold(
            route_blocks,
            source_ids,
        )

        group_settings = dict(
            directory=str(number),
            reference_sources=sorted({
                sample.source_id
                for sample in group_samples_
                if roles.get(sample.source_id) == "reference"
            }),
            calibration_sources=sorted({
                sample.source_id
                for sample in held_out
            }),
            train_examples=train_examples,
            score_calibration=score_calibration,
            sticky=dict(rho=rho, **sticky_calibration),
            high_transition=high_transition,
            final_self_supervised_loss=losses[-1],
        )
        write_json(directory / "settings.json", group_settings)
        write_json(complete, dict(complete=True, labels_used=False))
        settings["groups"][group_name(group)] = group_settings

    write_json(root / "settings.json", settings)
    write_json(root / "complete.json", dict(complete=True, labels_used=False))


def surface_sentence_start(tokenizer, response_ids):
    pieces = [
        tokenizer.decode(
            [int(token)],
            clean_up_tokenization_spaces=False,
        )
        for token in response_ids
    ]
    flags = np.zeros(len(pieces), dtype=bool)
    terminal = (".", "?", "!", "。", "？", "！")
    if len(flags):
        flags[0] = True

    for position in range(1, len(flags)):
        previous = pieces[position - 1].rstrip()
        flags[position] = (
            previous.endswith(terminal)
            or "\n" in previous
            or pieces[position].startswith("\n")
        )
    return flags


def score_sample(sample, index, tokenizer, source_record, model, saved, settings, args):
    routes, coverage, channels, source_tokens = route_tensor(
        sample,
        index,
        tokenizer,
        source_record,
        args,
    )
    if source_tokens == 0:
        raise ValueError(f"{sample.response_id}: source_info did not map to prompt")
    if not np.array_equal(saved["channels"], channels):
        raise ValueError("Test attention layout differs from fit")

    raw = score_routes(
        model,
        routes,
        coverage,
        saved["center"],
        saved["scale"],
        args.device,
        args.batch_size,
    )
    calibrated = standardize_scores(
        raw,
        settings["score_calibration"],
    )

    rho = settings["sticky"]["rho"]
    sticky_raw = causal_sticky(calibrated["compatibility"], rho)
    sticky = (
        (sticky_raw - settings["sticky"]["center"])
        / settings["sticky"]["scale"]
    ).astype(np.float32)

    arrays = {**calibrated, "sticky": sticky}
    for name in SCORES:
        arrays[name + "_alarm"] = (
            calibrated[name]
            > settings["score_calibration"][name]["threshold"]
        )
    arrays["sticky_alarm"] = sticky > settings["sticky"]["threshold"]
    arrays["self_jump"] = self_jump(routes)
    arrays["sentence_start"] = surface_sentence_start(
        tokenizer,
        sample.token_ids[sample.prompt_length:],
    )
    length = sample.response_length
    arrays["position"] = (
        np.arange(length, dtype=np.float32)
        / max(1, length - 1)
    )
    arrays["coverage"] = coverage
    arrays["token_ids"] = sample.token_ids
    arrays["prompt_length"] = np.asarray(sample.prompt_length)
    if len(sample.offsets):
        arrays["offsets"] = sample.offsets
    return arrays


def score(args):
    from transformers import AutoTokenizer

    samples, indexes, sources, _ = load_inputs(args)
    test = [sample for sample in samples if sample.split == "test"]
    fit_settings = json.loads(
        (args.output / "fit/settings.json").read_text()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        use_fast=True,
    )

    output = args.output / "predictions"
    output.mkdir(exist_ok=True)
    loaded = {}
    records = []

    for number, sample in enumerate(
        tqdm(test, desc="structured compatibility", unit="answer")
    ):
        group = group_name((sample.task, sample.generator))
        setting = fit_settings["groups"][group]
        path = output / f"{number:06d}.npz"

        if not (args.resume and path.exists()):
            if group not in loaded:
                directory = args.output / "fit" / setting["directory"]
                loaded[group] = load_model(
                    directory / "model.pt",
                    args.device,
                )
            model, saved = loaded[group]
            arrays = score_sample(
                sample,
                indexes["test"],
                tokenizer,
                sources[sample.source_id],
                model,
                saved,
                setting,
                args,
            )
            record = dict(
                id=sample.response_id,
                source_id=sample.source_id,
                task=sample.task,
                generator=sample.generator,
                split=sample.split,
                file=path.name,
                prompt_length=sample.prompt_length,
                response_sha256=sample.response_sha256,
            )
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(
                temporary,
                **arrays,
                record_json=np.asarray(json.dumps(record)),
            )
            temporary.replace(path)

        with np.load(path, allow_pickle=False) as archive:
            record = json.loads(str(archive["record_json"]))
        records.append(record)

    write_json(
        output / "freeze.json",
        dict(
            complete=True,
            labels_used=False,
            methods=METHODS,
            controls=CONTROLS,
            records=records,
        ),
    )


def inspect(args):
    from transformers import AutoTokenizer

    samples, indexes, sources, _ = load_inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        use_fast=True,
    )

    for split in ("train", "test"):
        sample = next(item for item in samples if item.split == split)
        routes, coverage, layout, source_tokens = route_tensor(
            sample,
            indexes[split],
            tokenizer,
            sources[sample.source_id],
            args,
        )
        print(json.dumps(dict(
            split=split,
            id=sample.response_id,
            shape=list(routes.shape),
            coverage=float(coverage.mean()),
            source_tokens=source_tokens,
            channels=layout.tolist(),
            labels_used=False,
        )), flush=True)
