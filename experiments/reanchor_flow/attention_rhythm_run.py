"""Raw head-by-head rhythm and bounded native relay inspection.

Capture is label-free. Analysis joins labels only after all observations and
example choices are saved. --phase analyze reuses NPZ without loading the LLM.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
from tqdm.auto import tqdm

from .attention_rhythm import RhythmConfig, capture_rhythm
from .attention_rhythm_report import (
    analyze_rhythm, event_examples, label_observations, plot_sample,
    prompt_history_trends, save_json, summarize_rhythm,
)

MODEL = "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
CACHE = ("/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/"
         "outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876")
SOURCE = "/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl"
TASKS = ("QA", "Summary", "Data2txt")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("all", "capture", "analyze"), default="all")
    p.add_argument("--model", type=Path, default=Path(MODEL))
    p.add_argument("--cache", type=Path, help="label/input cache root; defaults to scan manifest or original cache")
    p.add_argument("--scans", type=Path, help="existing scan root with train/test; token inputs only")
    p.add_argument("--source-info", type=Path, default=Path(SOURCE))
    p.add_argument("--output", type=Path, default=Path("experiments/reanchor_flow/outputs/attention_rhythm"))
    p.add_argument("--split", choices=("train", "test", "all"), default="test")
    p.add_argument("--task", choices=(*TASKS, "all"), default="QA")
    p.add_argument("--sample-id", action="append", default=[])
    p.add_argument("--samples-per-task", type=int, default=1, help="0 = all; no label-based selection")
    p.add_argument("--max-response-tokens", type=int, default=0, help="0 = full available response")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    p.add_argument("--query-chunk", type=int, default=8)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--future-lo", type=int, default=10)
    p.add_argument("--future-hi", type=int, default=100)
    p.add_argument("--local-window", type=int, default=10)
    p.add_argument("--head-fraction", type=float, default=.30)
    p.add_argument("--head", action="append", default=[], help="map head L:H, repeatable; all heads retain curves")
    p.add_argument("--map-tokens", type=int, default=128, help="raw plot window only, 0 disables maps")
    p.add_argument("--map-offset", type=int, default=0, help="relative to first response predictor")
    p.add_argument("--plots-per-task", type=int, default=1, help="per split/task; 0 disables detail forward")
    p.add_argument("--relay-examples", type=int, default=2, help="native two-hop examples per plotted sample; 0 disables")
    p.add_argument("--paper-groups", action="store_true", help="optional group-mean reference, excluded from primary statistics")
    p.add_argument("--evaluate", action="store_true", help="post-hoc normal/hallucinated comparisons, not detector fitting")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    return p


def capture(args):
    import torch
    from research_dataset import open_research_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .scan_dataset import ScanDataset
    from .subset_data import inspect_records, select_records, sample_tokens
    from .units import build_source_units

    config = RhythmConfig(args.window, args.future_lo, args.future_hi, args.local_window, args.head_fraction)
    if min(args.samples_per_task, args.max_response_tokens, args.plots_per_task,
           args.map_tokens, args.map_offset, args.relay_examples) < 0 or args.query_chunk < 1:
        raise ValueError("budgets must be nonnegative and query chunk positive")
    explicit_heads = tuple(tuple(map(int, value.split(":"))) for value in args.head)
    if any(len(pair) != 2 for pair in explicit_heads):
        raise ValueError("--head uses L:H")
    with args.source_info.open(encoding="utf-8") as stream:
        sources = {str(row["source_id"]): row for row in map(json.loads, stream)}
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model, entries = None, []
    splits = ("train", "test") if args.split == "all" else (args.split,)
    tasks = TASKS if args.task == "all" else (args.task,)
    for split in splits:
        dataset = (ScanDataset(args.scans / split) if args.scans else open_research_dataset(
            (args.cache or Path(CACHE)) / split, device="cpu", retain_embedded_labels=False))
        records = dataset.records if args.scans else inspect_records(dataset, source_info=sources)
        chosen = select_records(records, tasks=tasks, samples_per_task=args.samples_per_task,
                                seed=args.seed, sample_ids=tuple(args.sample_id))
        plot_counts = dict.fromkeys(TASKS, 0)
        progress = tqdm(chosen, desc=f"rhythm capture {split}", unit="sample")
        for record in progress:
            progress.set_postfix_str(f"{record.task_type}/{record.sample_id}")
            if args.scans:
                scan = dataset.load(record.sample_id, fields=("token_ids",))
                ids, start = scan["token_ids"], scan.response_start
                full_count = scan.metadata["full_response_tokens"]
            else:
                ids, start = sample_tokens(dataset, record.sample_id)
                ids = ids.numpy()
                full_count = len(ids) - start
            if args.max_response_tokens:
                ids = ids[:start + args.max_response_tokens]
            units = build_source_units(sources[record.source_id], tokenizer, ids, start)
            evidence = np.zeros(len(ids), bool)
            for unit, kind in enumerate(units.kind):
                if kind not in {"response", "other_prompt"}:
                    evidence[:-1] |= units.token_unit_id.numpy() == unit
            plot = plot_counts[record.task_type] < args.plots_per_task
            plot_counts[record.task_type] += 1
            folder = args.output / split / record.task_type
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (quote(record.sample_id, safe="") + ".npz")
            settings = {**asdict(config), "rhythm_schema": 2, "model": str(args.model), "dtype": args.dtype,
                        "query_chunk": args.query_chunk, "map_tokens": args.map_tokens if plot else 0,
                        "map_offset": args.map_offset, "heads": [list(pair) for pair in explicit_heads],
                        "relay_examples": args.relay_examples if plot else 0, "paper_groups": args.paper_groups}
            if path.exists():
                with np.load(path, allow_pickle=False) as stored:
                    if (json.loads(str(stored["settings"])) != settings
                            or not np.array_equal(stored["token_ids"], ids)
                            or not np.array_equal(stored["evidence_mask"], evidence)):
                        raise ValueError("capture differs; use --phase analyze for old NPZ, or a new --output")
                resumed = True
            else:
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(args.model), local_files_only=True, torch_dtype=getattr(torch, args.dtype),
                        attn_implementation="eager").to(args.device).eval()
                trace = capture_rhythm(model, ids, start, evidence, config=config,
                                       query_chunk=args.query_chunk, map_tokens=args.map_tokens if plot else 0,
                                       map_offset=args.map_offset, explicit_heads=explicit_heads,
                                       paper_groups=args.paper_groups, relay_examples=args.relay_examples if plot else 0)
                trace.update(settings=np.array(json.dumps(settings)), evidence_mask=evidence,
                             sample_id=np.array(record.sample_id), source_id=np.array(record.source_id),
                             task_type=np.array(record.task_type), full_response_tokens=np.array(full_count),
                             generator_model=np.array(getattr(record, "generator_model", "unknown")),
                             token_text=np.array([tokenizer.decode([int(t)]) for t in ids]))
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **trace)
                temporary.replace(path)
                del trace
                resumed = False
            entries.append({"split": split, "sample_id": record.sample_id, "source_id": record.source_id,
                            "task_type": record.task_type, "path": str(path.relative_to(args.output)),
                            "response_tokens": len(ids) - start, "full_response_tokens": full_count,
                            "plot": plot, "resumed": resumed})
    if not entries:
        raise ValueError("no samples selected")
    manifest = {"rhythm_run_schema": 2, "labels_used_for_capture": False,
                "config": {"scans": str(args.scans) if args.scans else None,
                           "cache": str(args.cache) if args.cache else None}, "samples": entries}
    save_json(args.output / "index.json", manifest)
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest


def _join_labels(args, manifest, entries):
    from research_dataset import open_research_dataset
    from .scan_dataset import ScanDataset, ScanLabelStore

    config = manifest.get("config", {})
    scans_root = args.scans or (Path(config["scans"]) if config.get("scans") else None)
    cache_root = args.cache or (Path(config["cache"]) if config.get("cache") else None)
    stores = {}
    for entry in tqdm(entries, desc="rhythm label comparisons", unit="sample"):
        path, split = args.output / entry["path"], entry["split"]
        label_path = path.with_suffix(".labels.npz")
        if label_path.exists():
            with np.load(label_path, allow_pickle=False) as stored:
                labels = stored["labels"]
        else:
            if split not in stores:
                if scans_root:
                    dataset = ScanDataset(scans_root / split)
                    label_store = ScanLabelStore(dataset, dataset_root=cache_root / split if cache_root else None)
                    stores[split] = (dataset, label_store)
                else:
                    stores[split] = (open_research_dataset((cache_root or Path(CACHE)) / split,
                                     device="cpu", retain_embedded_labels=True), None)
            dataset, store = stores[split]
            if store is not None:
                labels = store.load(dataset.load(entry["sample_id"], fields=()))
            else:
                lookup = dataset.prepare_evaluation_labels([entry["sample_id"]])
                sample = dataset[entry["sample_id"]]
                try:
                    labels = lookup.response_labels(sample).detach().cpu().numpy()
                finally:
                    sample.release_attention()
            labels = labels[:entry["response_tokens"]]
            np.savez_compressed(label_path, labels=labels)
        with np.load(path, allow_pickle=False) as stored:
            trace = {key: stored[key] for key in ("row_position", "response_start", "distance",
                     "waad", "message_waad", "fai", "attention_buckets")}
        entry.update(label_observations(trace, labels))


def analyze(args, manifest):
    from .attention_relay import relay_examples
    from .attention_rhythm_plot import plot_relays
    from .attention_rhythm_report import plot_timeline

    entries = [dict(entry) for entry in manifest["samples"]]
    # Every audit/example choice is saved before reading the first label.
    for entry in tqdm(entries, desc="rhythm analysis", unit="sample"):
        path = args.output / entry["path"]
        with np.load(path, allow_pickle=False) as stored:
            trace = {key: stored[key] for key in stored.files if not key.startswith("relay_")}
            trace["relay_paths"] = stored["relay_paths"] if "relay_paths" in stored else np.empty((0, 7), int)
        audit = analyze_rhythm(trace)
        np.savez_compressed(path.with_suffix(".audit.npz"), **audit)
        save_json(path.with_suffix(".events.json"), event_examples(trace, audit))
        save_json(path.with_suffix(".relays.json"), relay_examples(trace))
        entry.update(waad_peak_rate=audit["waad_peak_count"] / entry["response_tokens"],
                     bucket_missed_fraction=audit["bucket_missed_fraction"],
                     attention_message_waad_mae=audit["attention_message_waad_mae"],
                     relay_examples=len(trace["relay_paths"]), **prompt_history_trends(trace))
        entry.setdefault("plot", "map_heads" in trace)
    if args.evaluate:
        _join_labels(args, manifest, entries)
    plotted = [entry for entry in entries if entry["plot"]]
    for entry in tqdm(plotted, desc="rhythm figures", unit="sample"):
        path = args.output / entry["path"]
        with np.load(path, allow_pickle=False) as stored:
            trace = dict(stored)
        with np.load(path.with_suffix(".audit.npz"), allow_pickle=False) as stored:
            audit = dict(stored)
        labels = None
        if args.evaluate:
            with np.load(path.with_suffix(".labels.npz"), allow_pickle=False) as stored:
                labels = stored["labels"]
        plot_sample(trace, audit, path.with_suffix(".png"), f"{entry['split']}/{entry['task_type']}/{entry['sample_id']}")
        plot_timeline(trace, labels, path.with_suffix(".timeline.png"))
        plot_relays(trace, path.with_suffix(".relay.png"), labels)
    return summarize_rhythm(entries, args.output, bootstrap=args.bootstrap, seed=args.seed,
                            evaluated=args.evaluate)


def run(args):
    if args.phase == "analyze":
        manifest = json.loads((args.output / "index.json").read_text(encoding="utf-8"))
        if manifest.get("labels_used_for_capture") is not False:
            raise ValueError("analysis requires label-free capture")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = capture(args)
    if args.phase == "capture":
        return manifest
    return analyze(args, manifest)


if __name__ == "__main__":
    run(parser().parse_args())
