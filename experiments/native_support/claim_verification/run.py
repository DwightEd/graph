"""Prepare, score and evaluate an explicitly exploratory small RAGTruth pilot."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .prompts import decomposition_messages, verification_messages
from .text import flatten_source, parse_claims, project_scores, sentences
from ..source_regions import source_spans


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    from transformers import AutoTokenizer

    sources = {str(row["source_id"]): row for row in read_jsonl(args.dataset / "source_info.jsonl")}
    rows = read_jsonl(args.dataset / "response.jsonl")
    history = read_jsonl(args.historical_roster)
    regression = {row["id"] for row in history if row["split"] == "development"}
    regression.update(("11907", "12015", "12045", "12219"))
    excluded = {row["source_id"] for row in history}
    excluded.update(str(row["source_id"]) for row in rows if str(row["id"]) in regression)
    selected = [(row, "regression") for row in rows if str(row["id"]) in regression]
    for task in ("QA", "Summary", "Data2txt"):
        candidates = {str(row["source_id"]) for row in rows if row["split"] == "test"
                      and str(row["source_id"]) not in excluded
                      and sources[str(row["source_id"])]["task_type"] == task}
        chosen = sorted(candidates, key=lambda key: hashlib.sha256(f"binding42:{key}".encode()).hexdigest())[:4]
        for source_id in chosen:
            answers = sorted((row for row in rows if str(row["source_id"]) == source_id
                              and row["split"] == "test"), key=lambda row: (row["model"], str(row["id"])))
            selected.extend((row, "source_holdout") for row in answers[:2])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    records = [prepare_record(row, group, sources, tokenizer) for row, group in selected]
    save_json(args.output / "manifest.json", dict(records=records, model=args.model,
        tokenizer=args.tokenizer, dataset=str(args.dataset.resolve()), seed=42,
        labels_used_for_scoring=False, group_scope="exploratory; prior aggregate test inspected",
        primary="atomic_raw", threshold=0.0, decomposition_limit=3072))
    print(json.dumps(dict(prepared=len(records), groups={group: sum(r["group"] == group for r in records)
                                                        for group in ("regression", "source_holdout")})))


def prepare_record(row, group, sources, tokenizer):
    source_id = str(row["source_id"])
    source = sources[source_id]
    spans = source_spans(source)
    evidence = "\n".join(source["prompt"][span["start"]:span["end"]] for span in spans)
    encoded = tokenizer(row["response"], add_special_tokens=False, return_offsets_mapping=True)
    flattened = evidence
    if source["task_type"] == "Data2txt":
        flattened = "\n".join(flatten_source(source["source_info"]))
    return dict(id=str(row["id"]), source_id=source_id, task=source["task_type"], group=group,
        generator=row["model"], split=row["split"], source=evidence, flat_source=flattened,
        text=row["response"], token_ids=encoded["input_ids"], offsets=encoded["offset_mapping"],
        sentences=sentences(row["response"]))


def score_claims(reader, source, context, claims):
    messages = [verification_messages(source, context, claim, reverse)
                for claim in claims for reverse in (False, True)]
    margins = np.asarray(reader.margins(messages)).reshape(-1, 2)
    return (0.5 * (margins[:, 0] - margins[:, 1])).tolist(), margins.tolist()


def score_record(reader, record, extraction):
    claims, invalid = parse_claims(extraction["text"], record["text"], record["source"])
    units = record["sentences"]
    direct, direct_orders = score_claims(reader, record["source"], record["text"], [unit["text"] for unit in units])
    atom_scores, atom_orders = score_claims(reader, record["source"], record["text"], [claim["claim"] for claim in claims])
    flat_scores = atom_scores
    flat_orders = atom_orders
    if record["task"] == "Data2txt":
        flat_scores, flat_orders = score_claims(reader, record["flat_source"], record["text"], [claim["claim"] for claim in claims])
    alternatives = [claim for claim in claims if claim["alternative"].strip()]
    alternative_scores, alternative_orders = score_claims(reader, record["source"], record["text"],
                                                         [claim["alternative"] for claim in alternatives])
    for claim, score in zip(alternatives, alternative_scores, strict=True):
        claim["alternative_risk"] = score
    baseline, atomic, covered = project_scores(record["offsets"], units, direct, claims, atom_scores)
    _, flattened, _ = project_scores(record["offsets"], units, direct, claims, flat_scores)
    arrays = dict(token_id=np.asarray(record["token_ids"]), direct_sentence=baseline,
                  atomic_raw=atomic, atomic_flat=flattened, claim_covered=covered)
    audit = dict(id=record["id"], claims=claims, invalid_claims=invalid, extraction=extraction,
                 direct_scores=direct, direct_label_orders=direct_orders,
                 atom_scores=atom_scores, atom_label_orders=atom_orders,
                 flat_scores=flat_scores, flat_label_orders=flat_orders,
                 alternative_label_orders=alternative_orders, covered_tokens=int(covered.sum()),
                 tokens=len(covered), evidence_exact=sum(claim["evidence_exact"] for claim in claims))
    return arrays, audit


def score(args):
    from .runtime import Reader

    manifest = json.loads((args.output / "manifest.json").read_text())
    records = [row for row in manifest["records"] if args.group == "all" or row["group"] == args.group]
    if args.ids:
        records = [row for row in records if row["id"] in args.ids]
    pending = [row for row in records if not (args.output / "responses" / row["id"] / "scores.npz").exists()]
    if not pending:
        print("All requested predictions already frozen")
        return
    reader = Reader(manifest["model"], args.batch_size)
    started = perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        extractions = []
        for record in batch:
            path = args.output / "responses" / record["id"] / "extraction.json"
            extractions.append(json.loads(path.read_text()) if path.exists() else None)
        missing = [index for index, value in enumerate(extractions) if value is None]
        generated = reader.generate([decomposition_messages(batch[index]) for index in missing])
        for index, extraction in zip(missing, generated, strict=True):
            extractions[index] = extraction
            save_json(args.output / "responses" / batch[index]["id"] / "extraction.json", extraction)
        for record, extraction in zip(batch, extractions, strict=True):
            arrays, audit = score_record(reader, record, extraction)
            directory = args.output / "responses" / record["id"]
            save_json(directory / "audit.json", audit)
            np.savez_compressed(directory / "scores.tmp.npz", **arrays)
            (directory / "scores.tmp.npz").replace(directory / "scores.npz")
            print(json.dumps(dict(id=record["id"], claims=len(audit["claims"]), invalid=len(audit["invalid_claims"]),
                covered=audit["covered_tokens"], tokens=audit["tokens"], elapsed_seconds=round(perf_counter() - started, 2))), flush=True)
    completion_name = "completed_subset.json" if args.ids else f"completed_{args.group}.json"
    save_json(args.output / completion_name, dict(ids=[row["id"] for row in records],
              seconds=perf_counter() - started, actual_completed=True, whole_group=not bool(args.ids)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "score", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--historical-roster", type=Path)
    parser.add_argument("--model", default="/share/home/tm902089733300000/a903202310/lys/models/Qwen3-8B")
    parser.add_argument("--tokenizer", default="/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group", choices=("all", "regression", "source_holdout"), default="all")
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    if args.stage == "prepare":
        if (args.output / "manifest.json").exists():
            raise ValueError("Roster already frozen; use a new output directory")
        prepare(args)
    elif args.stage == "score":
        score(args)
    else:
        from .evaluate import evaluate
        evaluate(args)


if __name__ == "__main__":
    main()
