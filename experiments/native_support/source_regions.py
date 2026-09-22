"""Prompt-only source regions; no response text, annotation, or correct-answer roles."""

from pathlib import Path

import numpy as np
from state_audit.dataset import Example
from state_audit.dataset.jsonl import read_jsonl
from state_audit.dataset.ragtruth import evidence_texts, locate_evidence
from state_audit.storage import read_json, write_json
from state_audit.tokenization import encode_prompt


def source_spans(source):
    """QA matches the historical passage block; other tasks use teaching spans."""
    if source["task_type"] != "QA":
        return locate_evidence(source["prompt"], evidence_texts(source))
    prompt = source["prompt"]
    start = prompt.index("passages:\n") + len("passages:\n")
    stops = [prompt.find(marker, start) for marker in ("\nIn case the passages", "\noutput:")]
    stop = min(position for position in stops if position >= 0)
    return [{"id": "passages", "start": start, "end": stop}]


def compile_regions(settings, output):
    """Only load a tokenizer when official source masks are not already cached."""
    path = output / "source_regions.json"
    if path.exists():
        return read_json(path)
    if "cohort" not in settings:
        return {"status": "unavailable", "reason": "no_official_prompt_regions", "responses": {}}
    from transformers import AutoTokenizer

    directory = Path(settings["cohort"]["dataset"])
    sources = {str(row["source_id"]): row for row in read_jsonl(directory / "source_info.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], use_fast=True)
    regions = {}
    for response in settings["responses"]:
        source = sources[response["source_id"]]
        example = Example(response["id"], response["source_id"], source["prompt"], source_spans(source))
        encoded = encode_prompt(tokenizer, example, "chat")
        prefix = response["token_ids"][:response["prompt_length"]]
        if encoded["prompt_ids"] != prefix:
            raise ValueError(f"{response['id']}: source-region tokenizer differs from saved prefix")
        regions[response["id"]] = {
            "prompt_ids": prefix, "source_mask": (np.asarray(encoded["key_sources"]) >= 0).tolist(),
        }
    result = {"status": "available", "labels_used": False, "scope": "prompt_source_blocks_not_applicability", "responses": regions}
    write_json(path, result)
    return result


def region_mask(regions, response):
    if regions["status"] == "unavailable":
        return None
    region = regions["responses"][response["id"]]
    if region["prompt_ids"] != response["token_ids"][:response["prompt_length"]]:
        raise ValueError(f"{response['id']}: source-region cache differs from saved prefix")
    return np.asarray(region["source_mask"], dtype=bool)
