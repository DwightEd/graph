"""Read natural labels only after compatibility scores are frozen."""

from collections import defaultdict
import json

import numpy as np

from ..evaluate import Ranking, label_views, scoped_metrics
from ..evaluation_data import EvaluationBinding, read_sources
from ..offline_span.data import write_json
from ..span_audit.inputs import default_observer_tokenizer
from .pipeline import CONTROLS, METHODS

EVALUATION_METHODS = (*METHODS, *CONTROLS)


def transition_views(error, sentence_start, self_jump, high_transition):
    previous = np.r_[False, error[:-1]]
    full = np.ones(len(error), dtype=bool)
    return {
        "previous_gold_0": (error, ~previous),
        "previous_gold_1": (error, previous),
        "previous_gold_0_sentence_start": (
            error,
            ~previous & sentence_start,
        ),
        "previous_gold_0_high_transition": (
            error,
            ~previous & (self_jump >= high_transition),
        ),
        "all": (error, full),
    }


def binding_arrays(saved, row):
    """Supply identity fields needed for exact token alignment.

    Predictions written before prompt_length was added remain valid because
    prompt_length is already frozen in record_json/freeze.json.
    """
    prompt_length = int(row["prompt_length"])
    if "prompt_length" in saved:
        current = int(saved["prompt_length"])
        if current != prompt_length:
            raise ValueError("prediction prompt_length disagrees with frozen record")
        prompt_length = current

    arrays = {
        "token_ids": saved["token_ids"],
        "prompt_length": np.asarray(prompt_length),
    }
    if "offsets" in saved and saved["offsets"].size:
        arrays["offsets"] = saved["offsets"]
    return arrays


def read_blocks(args):
    root = args.output / "predictions"
    freeze = json.loads((root / "freeze.json").read_text())
    fit = json.loads((args.output / "fit/settings.json").read_text())
    if not freeze["complete"] or freeze["labels_used"]:
        raise ValueError("Freeze label-free compatibility scores before evaluation")

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
        with np.load(root / row["file"], allow_pickle=False) as saved:
            identity, offsets = binder.bind(
                row,
                annotation,
                binding_arrays(saved, row),
                sources,
            )
            standard = label_views(offsets, annotation["labels"])
            error = standard["all_error"][0]
            group = identity["task"] + "|" + identity["generator"]
            high_transition = fit["groups"][group]["high_transition"]
            extra = transition_views(
                error,
                saved["sentence_start"].astype(bool),
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
                transition=extra,
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


def paired_delta(blocks, family, view, left, right, draws):
    labels, left_scores, sources, _ = metric_arrays(
        blocks, family, view, left
    )
    _, right_scores, _, _ = metric_arrays(
        blocks, family, view, right
    )
    common = np.isfinite(left_scores) & np.isfinite(right_scores)

    left_rank = Ranking(labels[common], left_scores[common])
    right_rank = Ranking(labels[common], right_scores[common])
    left_point = left_rank.measure()
    right_point = right_rank.measure()
    delta = {
        key: left_point[key] - right_point[key]
        if left_point[key] is not None and right_point[key] is not None
        else None
        for key in ("auroc", "ap")
    }

    unique, inverse = np.unique(sources[common], return_inverse=True)
    random = np.random.default_rng(20260919)
    samples = []
    for _ in range(draws if len(unique) > 1 else 0):
        counts = np.bincount(
            random.integers(len(unique), size=len(unique)),
            minlength=len(unique),
        )[inverse]
        current_left = left_rank.measure(counts)
        current_right = right_rank.measure(counts)
        if current_left["auroc"] is not None:
            samples.append([
                current_left["auroc"] - current_right["auroc"],
                current_left["ap"] - current_right["ap"],
            ])

    return dict(
        tokens=int(common.sum()),
        delta=delta,
        ci95=np.quantile(samples, [.025, .975], axis=0).tolist()
        if samples else None,
        order=["auroc", "ap"],
    )



def threshold_metrics(blocks, family, view, method):
    positives = 0
    negatives = 0
    true_positive = 0
    false_positive = 0

    for block in blocks:
        labels, mask = block[family][view]
        alarm = block["alarms"][method]
        valid = mask & np.isfinite(block["scores"][method])
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


def evaluate_group(blocks, bootstrap):
    report = {
        "standard": {},
        "transition": {},
        "sticky_minus_base": {},
    }
    for family in ("standard", "transition"):
        views = blocks[0][family]
        for view in views:
            methods = {}
            for method in EVALUATION_METHODS:
                arrays = metric_arrays(blocks, family, view, method)
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
            report[family][view] = methods
            report["sticky_minus_base"][family + ":" + view] = paired_delta(
                blocks,
                family,
                view,
                "sticky",
                "compatibility",
                bootstrap,
            )
    return report


def evaluate(args):
    blocks = read_blocks(args)
    groups = defaultdict(list)
    groups["ALL"] = blocks
    for block in blocks:
        record = block["record"]
        groups[record["task"] + "|" + record["generator"]].append(block)

    report = dict(
        mode="self-supervised structured compatibility",
        primary="compatibility",
        persistence_secondary="sticky",
        labels_used_for_fit=False,
        score_direction=(
            "higher = less compatible with native head/source/time relations"
        ),
        groups={},
    )

    for name, selected in groups.items():
        result = evaluate_group(selected, args.bootstrap)
        report["groups"][name] = result
        for method in EVALUATION_METHODS:
            metric = result["standard"]["all_error"][method]
            print(json.dumps(dict(
                group=name,
                method=method,
                tokens=metric["evaluated_tokens"],
                positives=metric["evaluated_positives"],
                auroc=metric["pooled"]["auroc"],
                ap=metric["pooled"]["ap"],
            )), flush=True)

    write_json(args.output / "predictions/evaluation.json", report)
    return report
