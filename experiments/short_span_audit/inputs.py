"""Read frozen scores and build a reusable, score-independent short-span cohort."""

from types import SimpleNamespace

import numpy as np

from experiments.unsupervised_token_graph.head_geometry.continuity import (
    load_frozen,
    prepare_controls,
)
from experiments.unsupervised_token_graph.head_geometry.continuity_metrics import (
    merge_token_spans,
)
from experiments.unsupervised_token_graph.head_geometry.pipeline import read_json

DEFAULT_METHODS = ("all__raw", "all__moment", "all__pair_state_smooth")


def selected_methods(available, requested):
    methods = list(requested or [name for name in DEFAULT_METHODS if name in available])
    unknown = set(methods) - set(available)
    if unknown or not methods:
        raise ValueError(f"Choose saved score methods; available: {list(available)}; missing: {unknown}")
    return methods


def snapshot_blocks(directory, methods):
    answers = read_json(directory / "answers.json")
    matching = read_json(directory / "matching.json")
    pairs = {str(row["record"]["id"]): row["pairs"] for row in matching}
    blocks = []
    with np.load(directory / "tokens.npz", allow_pickle=False) as saved:
        for index, answer in enumerate(answers):
            rows = saved["answer_index"] == index
            if int(rows.sum()) != answer["tokens"]:
                raise ValueError(f"Snapshot token count differs for {answer['record']['id']}")
            offsets = np.asarray(answer["offsets"], dtype=int)
            valid = offsets[:, 1] > offsets[:, 0]
            finite = saved["common_finite"][rows] & valid
            finite &= np.logical_and.reduce([np.isfinite(saved[name][rows]) for name in methods])
            block = dict(answer, offsets=offsets, gold=merge_token_spans(answer["gold"]))
            block["record"] = dict(answer["record"])
            block["record"].setdefault("dataset", "RAGTruth")
            block["scores"] = {name: np.where(finite, saved[name][rows], np.nan) for name in methods}
            block["alarms"] = {name: saved[name + "__alarm"][rows].copy() for name in methods}
            block["common_finite"] = finite
            block["views"] = {"all_error": (saved["label"][rows].copy(), valid)}
            block["pairs"] = pairs[str(answer["record"]["id"])]
            blocks.append(block)
    return blocks, matching


def load_input(args):
    directory = args.input
    if (directory / "continuity" / "tokens.npz").exists():
        directory = directory / "continuity"
    if (directory / "tokens.npz").exists():
        audit = read_json(directory / "audit.json")
        methods = selected_methods(audit["methods"], args.methods)
        blocks, matching = snapshot_blocks(directory, methods)
        settings = read_json(directory / "input_settings.json")
        provenance = dict(input=str(directory.resolve()), coverage="saved_common_coverage",
                          original_scored_tokens={key: value["original_scored_tokens"]
                                                 for key, value in audit["groups"].items()})
    else:
        options = SimpleNamespace(output=directory, dataset=args.dataset,
                                  source_info=args.source_info, tokenizer=args.tokenizer,
                                  methods=args.methods)
        original, freeze, _, settings = load_frozen(options)
        methods = selected_methods(freeze["methods"], args.methods)
        blocks, matching = prepare_controls(original, methods, neighborhood=0)
        for block in blocks:
            block["record"] = dict(block["record"])
            block["record"].setdefault("dataset", "RAGTruth")
        provenance = dict(input=str(directory.resolve()), coverage="selected_methods_common_coverage")
    if args.tasks:
        blocks = [block for block in blocks if block["record"]["task"] in args.tasks]
        matching = [row for row in matching if row["record"]["task"] in args.tasks]
    return blocks, methods, matching, settings, provenance


def interval_targets(start, end, side, pair_id):
    return [dict(target=target, side=side, pair_id=pair_id, span_start=int(start),
                 span_end=int(end), offset=target - int(start)) for target in range(start, end)]


def short_cohort(blocks):
    """Gold selects audit positions only; it never changes saved scores or fits a bank."""
    cohort = []
    for block in blocks:
        pairs = [pair for pair in block["pairs"] if pair["length"] <= 8]
        pair_ids = {(pair["error_start"], pair["error_end"]): pair["pair_id"] for pair in pairs}
        targets = []
        for start, end in block["gold"]:
            if end - start <= 8:
                targets.extend(interval_targets(start, end, "error", pair_ids.get((start, end))))
        for pair in pairs:
            targets.extend(interval_targets(pair["normal_start"], pair["normal_end"],
                                            "normal", pair["pair_id"]))
        if targets:
            cohort.append(dict(record=block["record"], tokens=block["tokens"],
                               gold=block["gold"].tolist(), offsets=block["offsets"].tolist(),
                               text=block["text"], pairs=pairs,
                               targets=sorted(targets, key=lambda row: row["target"])))
    return cohort
