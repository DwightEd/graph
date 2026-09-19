"""Evaluate frozen latent-regime scores; labels are read only after scoring."""

from collections import defaultdict
import json

import numpy as np

from ..evaluate import Ranking, label_views, scoped_metrics
from ..evaluation_data import EvaluationBinding, read_sources
from ..offline_span.data import write_json
from ..span_audit.inputs import default_observer_tokenizer
from .pipeline import CONTROLS, METHODS


EVALUATION_METHODS = (*METHODS, *CONTROLS)


def sentence_start(response, offsets):
    tokens = [response[start:end] for start, end in offsets]
    flags = np.zeros(len(tokens), dtype=bool)
    terminal = (".", "?", "!", "。", "？", "！")

    if len(flags):
        flags[0] = True

    for position in range(1, len(tokens)):
        previous = tokens[position - 1].rstrip()
        flags[position] = (
            previous.endswith(terminal)
            or "\n" in previous
            or tokens[position].startswith("\n")
        )

    return flags


def transition_views(
    error,
    sentence_flags,
    self_jump,
    high_transition,
):
    previous = np.r_[False, error[:-1]]
    full = np.ones(len(error), dtype=bool)

    return {
        "all": (error, full),
        "previous_gold_0": (error, ~previous),
        "previous_gold_1": (error, previous),
        "previous_gold_0_sentence_start": (
            error,
            ~previous & sentence_flags,
        ),
        "previous_gold_0_high_transition": (
            error,
            ~previous & (self_jump >= high_transition),
        ),
    }


def read_blocks(args):
    root = args.output / "predictions"
    freeze = json.loads((root / "freeze.json").read_text())
    fit = json.loads((args.output / "fit/settings.json").read_text())

    if not freeze["complete"] or freeze["labels_used"]:
        raise ValueError("Freeze label-free regime scores before evaluation")

    annotations = args.dataset / "response.jsonl"
    with annotations.open(encoding="utf-8") as stream:
        gold = {str(row["id"]): row for row in map(json.loads, stream)}

    sources, _ = read_sources(annotations, args.source_info)
    fallback = default_observer_tokenizer(args.test_cache.resolve())
    binder = EvaluationBinding(
        dict(cache="/", tokenizer_fallback=fallback),
        args.tokenizer,
    )

    blocks = []
    for row in freeze["records"]:
        annotation = gold[row["id"]]
        group = row["task"] + "|" + row["generator"]
        high_transition = fit["groups"][group]["high_transition"]

        with np.load(root / row["file"], allow_pickle=False) as saved:
            binding = {
                "token_ids": saved["token_ids"],
                "prompt_length": saved["prompt_length"],
            }
            if saved["offsets"].size:
                binding["offsets"] = saved["offsets"]

            identity, offsets = binder.bind(
                row,
                annotation,
                binding,
                sources,
            )
            standard = label_views(offsets, annotation["labels"])
            error = standard["all_error"][0]
            sentence_flags = sentence_start(
                annotation["response"],
                offsets,
            )
            transition = transition_views(
                error,
                sentence_flags,
                saved["self_jump"],
                high_transition,
            )
            scores = {
                name: saved[name].copy()
                for name in EVALUATION_METHODS
            }
            alarms = {
                name: saved[name + "_alarm"].astype(bool)
                for name in METHODS
            }

        blocks.append(dict(
            record=identity,
            standard=standard,
            transition=transition,
            scores=scores,
            alarms=alarms,
            tokens=len(offsets),
        ))

    return blocks


def metric_arrays(blocks, family, view, method):
    source_sizes = defaultdict(int)
    for block in blocks:
        source_sizes[block["record"]["source_id"]] += block["tokens"]

    labels = []
    scores = []
    sources = []
    weights = []

    for block in blocks:
        target, mask = block[family][view]
        source = block["record"]["source_id"]

        labels.extend(target[mask])
        scores.extend(block["scores"][method][mask])
        sources.extend([source] * int(mask.sum()))
        weights.extend([1. / source_sizes[source]] * int(mask.sum()))

    return (
        np.asarray(labels),
        np.asarray(scores),
        np.asarray(sources),
        np.asarray(weights),
    )


def threshold_metrics(blocks, family, view, method):
    positives = negatives = true_positive = false_positive = 0

    for block in blocks:
        labels, mask = block[family][view]
        scores = block["scores"][method]
        alarm = block["alarms"][method]
        valid = mask & np.isfinite(scores)

        positives += int(labels[valid].sum())
        negatives += int((~labels[valid]).sum())
        true_positive += int((alarm[valid] & labels[valid]).sum())
        false_positive += int((alarm[valid] & ~labels[valid]).sum())

    return dict(
        recall=true_positive / positives if positives else None,
        fpr=false_positive / negatives if negatives else None,
        positives=positives,
        negatives=negatives,
    )


def macro_within_answer(blocks, family, view, method):
    aucs = []
    aps = []

    for block in blocks:
        labels, mask = block[family][view]
        scores = block["scores"][method]
        valid = mask & np.isfinite(scores)
        current = labels[valid]

        if len(np.unique(current)) < 2:
            continue

        point = Ranking(current, scores[valid]).measure()
        aucs.append(point["auroc"])
        aps.append(point["ap"])

    return dict(
        mixed_answers=len(aucs),
        auroc=float(np.mean(aucs)) if aucs else None,
        ap=float(np.mean(aps)) if aps else None,
    )


def paired_delta(blocks, family, view, left, right, bootstrap):
    labels, left_score, sources, _ = metric_arrays(
        blocks,
        family,
        view,
        left,
    )
    _, right_score, _, _ = metric_arrays(
        blocks,
        family,
        view,
        right,
    )
    valid = np.isfinite(left_score) & np.isfinite(right_score)

    left_rank = Ranking(labels[valid], left_score[valid])
    right_rank = Ranking(labels[valid], right_score[valid])
    left_point = left_rank.measure()
    right_point = right_rank.measure()

    delta = {
        key: left_point[key] - right_point[key]
        if left_point[key] is not None and right_point[key] is not None
        else None
        for key in ("auroc", "ap")
    }

    unique, inverse = np.unique(sources[valid], return_inverse=True)
    random = np.random.default_rng(20260920)
    draws = []

    for _ in range(bootstrap if len(unique) > 1 else 0):
        multiplicity = np.bincount(
            random.integers(len(unique), size=len(unique)),
            minlength=len(unique),
        )[inverse]
        current_left = left_rank.measure(multiplicity)
        current_right = right_rank.measure(multiplicity)

        if current_left["auroc"] is not None:
            draws.append([
                current_left["auroc"] - current_right["auroc"],
                current_left["ap"] - current_right["ap"],
            ])

    return dict(
        tokens=int(valid.sum()),
        delta=delta,
        order=["auroc", "ap"],
        ci95=np.quantile(draws, [.025, .975], axis=0).tolist()
        if draws else None,
    )


def evaluate_group(blocks, bootstrap):
    result = {
        "standard": {},
        "transition": {},
        "temporal_gain": {},
        "head_structure_gain": {},
    }

    for family in ("standard", "transition"):
        for view in blocks[0][family]:
            methods = {}

            for method in EVALUATION_METHODS:
                arrays = metric_arrays(
                    blocks,
                    family,
                    view,
                    method,
                )
                methods[method] = scoped_metrics(
                    *arrays,
                    bootstrap=bootstrap,
                )
                methods[method]["macro_within_answer"] = macro_within_answer(
                    blocks,
                    family,
                    view,
                    method,
                )

                if method in METHODS:
                    methods[method]["threshold"] = threshold_metrics(
                        blocks,
                        family,
                        view,
                        method,
                    )

            result[family][view] = methods
            key = family + ":" + view
            result["temporal_gain"][key] = paired_delta(
                blocks,
                family,
                view,
                "filtered",
                "iid",
                bootstrap,
            )
            result["head_structure_gain"][key] = paired_delta(
                blocks,
                family,
                view,
                "filtered",
                "layer_mean_filtered",
                bootstrap,
            )

    return result


def evaluate(args):
    blocks = read_blocks(args)
    groups = defaultdict(list)
    groups["ALL"] = blocks

    for block in blocks:
        row = block["record"]
        groups[row["task"] + "|" + row["generator"]].append(block)

    report = dict(
        mode="unlabelled two-state sticky head-regime HMM",
        primary="filtered",
        labels_used_for_fit=False,
        suspect_orientation="lower source-balanced TRAIN occupancy state",
        groups={},
    )

    for name, selected in groups.items():
        result = evaluate_group(selected, args.bootstrap)
        report["groups"][name] = result

        for method in EVALUATION_METHODS:
            all_error = result["standard"]["all_error"][method]
            onset = result["transition"]["previous_gold_0"][method]
            continuation = result["transition"]["previous_gold_1"][method]

            print(json.dumps(dict(
                group=name,
                method=method,
                all_auroc=all_error["pooled"]["auroc"],
                all_ap=all_error["pooled"]["ap"],
                onset_auroc=onset["pooled"]["auroc"],
                continuation_auroc=continuation["pooled"]["auroc"],
                tokens=all_error["evaluated_tokens"],
                positives=all_error["evaluated_positives"],
            )), flush=True)

    write_json(
        args.output / "predictions/evaluation.json",
        report,
    )
    return report
