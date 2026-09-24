"""Official text selection and tokenization; annotations enter only after scoring."""

from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from state_audit.dataset import Example
from state_audit.dataset.jsonl import read_jsonl, validate
from state_audit.storage import read_json, start_stage, write_arrays, write_json
from state_audit.tokenization import encode_prompt, special_token_ids

from ..choice_cache import CaptureReader
from ..evidence_contrast.views import unit_intervals
from ..ragtruth import aligned_labels
from ..source_regions import source_spans

TASKS = ("QA", "Summary", "Data2txt")


def selected_rows(args, sources):
    rows = read_jsonl(args.dataset / "response.jsonl")
    chosen = [row for row in rows if sources[str(row["source_id"])]["task_type"] in args.tasks
              and row["split"] in args.splits and (not args.generators or row["model"] in args.generators)]
    chosen.sort(key=lambda row: (sources[str(row["source_id"])]["task_type"], row["split"],
                                 row["model"], str(row["id"])))
    if args.limit is not None:
        chosen = chosen[:args.limit]
    if not chosen:
        raise ValueError("No official responses match the requested tasks/splits/generators")
    return chosen, len(rows)


def encode_source(source, tokenizer):
    example = Example(str(source["source_id"]), str(source["source_id"]),
                      source["prompt"], source_spans(source))
    encoded = encode_prompt(tokenizer, example, "chat")
    mask = np.asarray(encoded["key_sources"]) >= 0
    kept = np.flatnonzero(~mask)
    if not len(kept) or not mask.any():
        raise ValueError(f"{example.id}: source region is empty or removes the whole prompt")
    return dict(prompt_with_source=encoded["prompt_ids"],
                prompt_without_source=[encoded["prompt_ids"][index] for index in kept],
                source_mask=mask.tolist())


def prepare_official(args):
    from transformers import AutoTokenizer
    sources = {str(row["source_id"]): row for row in read_jsonl(args.dataset / "source_info.jsonl")}
    rows, population = selected_rows(args, sources)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    source_files, records, excluded = {}, [], []
    for index, row in enumerate(tqdm(rows, desc="prepare official RAGTruth")):
        identity, source_id = str(row["id"]), str(row["source_id"])
        encoded = tokenizer(row["response"], add_special_tokens=False, return_offsets_mapping=True)
        if not encoded["input_ids"]:
            excluded.append(dict(id=identity, reason="empty_tokenized_response"))
            continue
        if source_id not in source_files:
            source_files[source_id] = f"sources/{len(source_files):06d}.json"
            start_stage(args.output / source_files[source_id], encode_source(sources[source_id], tokenizer), args.resume)
        response = dict(answer_ids=encoded["input_ids"], offsets=encoded["offset_mapping"],
            text=row["response"], token_text=[tokenizer.decode([token]) for token in encoded["input_ids"]],
            special_ids=special_token_ids(tokenizer))
        response["units"] = unit_intervals(dict(prompt_length=0, token_text=response["token_text"]), args.max_unit_tokens)
        directory = f"responses/{index:06d}"
        start_stage(args.output / directory / "response.json", response, args.resume)
        records.append(dict(id=identity, source_id=source_id, task=sources[source_id]["task_type"],
            split=row["split"], generator=row["model"], directory=directory,
            source_file=source_files[source_id], tokens=len(encoded["input_ids"])))
    return dict(model=args.model, dataset=str(args.dataset.resolve()), records=records, excluded=excluded,
                official_population=population, selected_answers=len(rows), labels_used_for_selection=False)


def prepare_cache(args):
    """Import only existing scalar observations; never repeat a native experiment."""
    records = []
    with closing(CaptureReader(args.cache_input)) as reader:
        settings = reader.json("settings.json")
        cohort = settings["cohort"]
        for index, response in enumerate(settings["responses"]):
            original, directory = f"responses/{index:04d}", f"responses/{index:06d}"
            views, values = reader.json(original + "/views.json"), reader.arrays(original + "/scores.npz")
            ids = response["token_ids"][response["prompt_length"]:]
            if not np.array_equal(values["token_id"], ids):
                raise ValueError(f"{response['id']}: cached token identities differ")
            observed = {name: values[name] for name in ("source_local", "source_full", "raw_route", "raw_attention", "entropy")}
            write_arrays(args.output / directory / "observations.npz", token_id=np.asarray(ids), **observed)
            start_stage(args.output / directory / "response.json", dict(answer_ids=ids, units=views["units"],
                token_text=response["token_text"][response["prompt_length"]:]), args.resume)
            records.append(dict(id=response["id"], source_id=response["source_id"],
                task=cohort["task"], split=cohort["split"], generator=cohort["generator"],
                directory=directory, tokens=len(ids)))
    return dict(model=settings["model"], cache_input=str(args.cache_input.resolve()), records=records,
        selected_answers=len(records), excluded=[], original_cohort=cohort,
        labels_used_for_selection=cohort["labels_used_for_selection"])


def prepare(args):
    manifest = prepare_cache(args) if args.cache_input else prepare_official(args)
    identities = [row["id"] for row in manifest["records"]]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Benchmark needs nonempty, unique response identities")
    counts = Counter((r["task"], r["split"], r["generator"]) for r in manifest["records"])
    manifest["groups"] = [dict(task=t, split=s, generator=g, answers=n) for (t, s, g), n in sorted(counts.items())]
    start_stage(args.output / "manifest.json", manifest, args.resume)
    return manifest


def annotations(output, manifest, records):
    """A caller supplies the allowed IDs; parameter selection requests train IDs only."""
    wanted = {row["id"] for row in records}
    if "cache_input" in manifest:
        with closing(CaptureReader(Path(manifest["cache_input"]))) as reader:
            saved = reader.json("annotations.json")
            return {identity: saved[identity] for identity in wanted}
    result = {}
    for row in read_jsonl(Path(manifest["dataset"]) / "response.jsonl"):
        if str(row["id"]) not in wanted:
            continue
        result[str(row["id"])] = row
    lookup = {row["id"]: row for row in records}
    aligned = {}
    for identity, row in result.items():
        response = read_json(output / lookup[identity]["directory"] / "response.json")
        if row["labels"] is None:
            raise ValueError(f"{identity}: missing annotations are not reviewed negatives")
        if row["response"] != response["text"]:
            raise ValueError(f"{identity}: official response text changed after preparation")
        example = Example(identity, str(row["source_id"]), "", response=row["response"], labels=row["labels"])
        validate(example)
        encoded = dict(input_ids=response["answer_ids"], offset_mapping=response["offsets"])
        aligned[identity] = aligned_labels(example, encoded, response["special_ids"])
    if set(aligned) != wanted:
        raise ValueError("Official annotations do not cover all selected responses")
    return aligned
