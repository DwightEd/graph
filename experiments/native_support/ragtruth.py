"""Small official-answer pilots, with separate verified evaluation annotations."""

import numpy as np
from state_audit.analysis.annotations import token_spans
from state_audit.dataset import Example
from state_audit.dataset.jsonl import read_jsonl, validate
from state_audit.tokenization import encode_prompt, special_token_ids


def select_responses(rows, sources, *, task, split, generator, limit, balanced):
    selected = [
        row for row in rows
        if sources[str(row["source_id"])]["task_type"] == task
        and row["split"] == split and row["model"] == generator
    ]
    selected.sort(key=lambda row: str(row["id"]))
    if balanced:
        positive = [row for row in selected if row["labels"]]
        negative = [row for row in selected if row["labels"] == []]
        positive_count = limit // 2
        negative_count = limit - positive_count
        if len(positive) < positive_count or len(negative) < negative_count:
            raise ValueError("Not enough official positive/negative answers for the requested pilot")
        selected = positive[:positive_count] + negative[:negative_count]
        selected.sort(key=lambda row: str(row["id"]))
    else:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No official responses match task={task}, split={split}, generator={generator}")
    return selected


def encode_response(row, source, tokenizer):
    if row["labels"] is None:
        raise ValueError(f"{row['id']}: official labels are missing, not a reviewed negative")
    example = Example(
        str(row["id"]), str(row["source_id"]), source["prompt"],
        response=row["response"], labels=row["labels"],
    )
    validate(example)
    prompt = encode_prompt(tokenizer, example, "chat")
    encoded = tokenizer(example.response, add_special_tokens=False, return_offsets_mapping=True)
    ids = prompt["prompt_ids"] + encoded["input_ids"]
    response = {
        "id": example.id, "source_id": example.source_id, "token_ids": ids,
        "prompt_length": len(prompt["prompt_ids"]),
        "token_text": [tokenizer.decode([token]) for token in ids],
    }
    annotation = aligned_labels(example, encoded, special_token_ids(tokenizer))
    return response, annotation


def aligned_labels(example, encoded, special_ids):
    ids = encoded["input_ids"]
    offsets = np.asarray(encoded["offset_mapping"])
    valid = (offsets[:, 1] > offsets[:, 0]) & ~np.isin(ids, special_ids)
    answer = {
        "labels": example.labels, "response_offsets": offsets,
        "response_ids": ids, "special_token_ids": special_ids,
    }
    for span in example.labels:
        overlap = (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])
        if not (overlap & valid).any():
            raise ValueError(f"{example.id}: a gold span has no ordinary token coverage")
    labels = np.zeros(len(ids), dtype=np.int64)
    onsets = np.zeros(len(ids), dtype=bool)
    for start, end in token_spans(answer):
        labels[start:end] = 1
        onsets[start] = True
    return {
        "source_id": example.source_id, "token_ids": ids, "labels": labels.tolist(),
        "span_onsets": onsets.tolist(), "valid_tokens": valid.tolist(),
        "response_text": example.response, "character_spans": example.labels,
        "annotation_origin": "RAGTruth/response.jsonl",
    }


def prepare_official(args, model_name):
    from transformers import AutoTokenizer

    sources = {str(row["source_id"]): row for row in read_jsonl(args.dataset / "source_info.jsonl")}
    rows = select_responses(
        read_jsonl(args.dataset / "response.jsonl"), sources, task=args.task,
        split=args.split, generator=args.generator, limit=args.limit, balanced=args.balanced,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    responses, annotations = [], {}
    for row in rows:
        response, annotation = encode_response(row, sources[str(row["source_id"])], tokenizer)
        responses.append(response)
        annotations[response["id"]] = annotation
    cohort = {
        "dataset": str(args.dataset.resolve()), "task": args.task, "split": args.split,
        "generator": args.generator, "observer_model": model_name,
        "selection": "balanced_answer_pilot" if args.balanced else "first_matching_ids",
        "labels_used_for_selection": args.balanced,
        "labels_used_for_scoring": False, "response_ids": [row["id"] for row in responses],
        "scope": "small_diagnostic_pilot_not_full_dataset",
    }
    return {"model": model_name, "responses": responses}, annotations, cohort
