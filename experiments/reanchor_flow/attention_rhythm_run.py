"""python -m experiments.reanchor_flow.attention_rhythm_run --help

Additive phenomenology audit. Existing detector/discover/corridor code is untouched.
The repository's research_dataset input and native Llama forward are reused.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .attention_rhythm import RhythmConfig, capture_rhythm
from .attention_rhythm_report import (
    analyze_rhythm, event_examples, label_observations, plot_sample, save_json, source_bootstrap,
)

MODEL = "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
CACHE = ("/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/"
         "outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876")
SOURCE = "/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl"
TASKS = ("QA", "Summary", "Data2txt")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path(MODEL))
    p.add_argument("--cache", type=Path, default=Path(CACHE))
    p.add_argument("--scans", type=Path, help="optional existing scan root with train/test; tokens only")
    p.add_argument("--source-info", type=Path, default=Path(SOURCE))
    p.add_argument("--output", type=Path, default=Path("experiments/reanchor_flow/outputs/attention_rhythm"))
    p.add_argument("--split", choices=("train", "test", "all"), default="test")
    p.add_argument("--task", choices=(*TASKS, "all"), default="QA")
    p.add_argument("--sample-id", action="append", default=[])
    p.add_argument("--samples-per-task", type=int, default=1, help="0 = all; no label-based selection")
    p.add_argument("--max-response-tokens", type=int, default=0, help="0 = full response; otherwise explicit truncation")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    p.add_argument("--query-chunk", type=int, default=8)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--future-lo", type=int, default=10)
    p.add_argument("--future-hi", type=int, default=100)
    p.add_argument("--local-window", type=int, default=10)
    p.add_argument("--head-fraction", type=float, default=.30)
    p.add_argument("--head", action="append", default=[], help="exact map head L:H, repeatable; default span-rank representatives")
    p.add_argument("--map-tokens", type=int, default=128, help="exact plot window only, NOT metric truncation; 0 disables maps")
    p.add_argument("--map-offset", type=int, default=0, help="relative to the first response predictor")
    p.add_argument("--plots-per-task", type=int, default=1, help="0 disables extra map pass; first samples selected without labels")
    p.add_argument("--paper-groups", action="store_true", help="optional group-MEAN paper reference; individual heads remain primary")
    p.add_argument("--evaluate", action="store_true", help="post-hoc H/N position-bin comparisons, NOT detector fitting")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    return p


def _finite_average(values):
    values = np.asarray(values)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else float("nan")


def _summary(entries, output, bootstrap, seed):
    keys = ("waad_peak_rate", "bucket_missed_fraction", "attention_message_waad_mae",
            "coupling_pair_mean_lift", "coupling_full_horizon_mean_lift")
    report = {"scope": "teacher-forced observational geometry; no confirmed factual re-anchor",
              "head_roles": "per-sample Eq.7 span ranks; descriptive, not train-frozen semantic roles",
              "coupling_null": "count-preserving uniform peak permutation expectation",
              "limitations": ["FAI is offline; partial horizons reported separately",
                              "peak rule is this audit's explicit convention, not author source code",
                              "position-bin comparisons do not remove lexical/generator confounding",
                              "confidence intervals condition on the fixed head/peak convention"],
              "groups": {}}
    groups = [(split, task) for split in sorted({e["split"] for e in entries})
              for task in ("ALL", *TASKS)]
    for split, task in groups:
        selected = [e for e in entries if e["split"] == split
                    and (task == "ALL" or e["task_type"] == task)]
        if not selected:
            continue
        task_keys = list(keys)
        if "waad_matched_h_minus_n" in selected[0]:
            task_keys += ["waad_matched_h_minus_n", "message_waad_matched_h_minus_n",
                          "distance_matched_h_minus_n"]
        group = {"samples": len(selected), "response_tokens": sum(e["response_tokens"] for e in selected),
                 "full_response_tokens": sum(e["full_response_tokens"] for e in selected)}
        for key in task_keys:
            group[key] = source_bootstrap(selected, key, repetitions=bootstrap, seed=seed)
        report["groups"][f"{split}/{task}"] = group
        coupling = group["coupling_pair_mean_lift"]
        print(f"{split}/{task:9s} samples={len(selected)} tokens={group['response_tokens']}/{group['full_response_tokens']} "
              f"sources={coupling['sources']} coupling_lift={float(coupling['mean']):.4f}", flush=True)
    save_json(output / "summary.json", report)
    # These heatmaps summarize effects over sources, never average raw heads.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task, group in report["groups"].items():
        columns = ["waad_peak_rate", "bucket_missed_fraction", "attention_message_waad_mae"]
        fig, axes = plt.subplots(1, len(columns), figsize=(14, 4))
        for axis, key in zip(axes, columns):
            image = axis.imshow(np.asarray(group[key]["mean"]), aspect="auto")
            axis.set_title(key.replace("_", " "), fontsize=9)
            axis.set_xlabel("head")
            axis.set_ylabel("layer")
            fig.colorbar(image, ax=axis, fraction=.045)
        fig.suptitle(f"{task}: source-balanced observations (not causal mechanisms)")
        fig.tight_layout()
        fig.savefig(output / f"population_{task.replace('/', '_')}.png", dpi=140)
        plt.close(fig)
    return report


def run(args):
    # Heavy/model imports are delayed so --help and numerical tests work offline.
    import torch
    from research_dataset import open_research_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .scan_dataset import ScanDataset, ScanLabelStore
    from .subset_data import inspect_records, select_records, sample_tokens
    from .units import build_source_units

    config = RhythmConfig(args.window, args.future_lo, args.future_hi, args.local_window, args.head_fraction)
    if min(args.samples_per_task, args.max_response_tokens, args.plots_per_task, args.map_tokens, args.map_offset) < 0:
        raise ValueError("sample, horizon and map budgets must be nonnegative")
    explicit_heads = tuple(tuple(map(int, value.split(":"))) for value in args.head)
    if any(len(pair) != 2 for pair in explicit_heads):
        raise ValueError("--head uses L:H")
    sources = {}
    with args.source_info.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sources[str(row["source_id"])] = row
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = None
    args.output.mkdir(parents=True, exist_ok=True)
    datasets, entries = {}, []
    splits = ("train", "test") if args.split == "all" else (args.split,)
    tasks = TASKS if args.task == "all" else (args.task,)
    for split in splits:
        dataset = (ScanDataset(args.scans / split) if args.scans else open_research_dataset(
            args.cache / split, device="cpu", retain_embedded_labels=False))
        datasets[split] = dataset
        records = dataset.records if args.scans else inspect_records(dataset, source_info=sources)
        chosen = select_records(records, tasks=tasks, samples_per_task=args.samples_per_task,
                                seed=args.seed, sample_ids=tuple(args.sample_id))
        plot_counts = dict.fromkeys(TASKS, 0)
        for record in chosen:
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
            settings = {**asdict(config), "model": str(args.model), "dtype": args.dtype,
                        "query_chunk": args.query_chunk, "map_tokens": args.map_tokens if plot else 0,
                        "map_offset": args.map_offset, "heads": [list(pair) for pair in explicit_heads],
                        "paper_groups": args.paper_groups}
            if path.exists():
                with np.load(path, allow_pickle=False) as stored:
                    trace = dict(stored)
                if (json.loads(str(trace["settings"])) != settings
                        or not np.array_equal(trace["token_ids"], ids)
                        or not np.array_equal(trace["evidence_mask"], evidence)):
                    raise ValueError("existing audit configuration differs; choose another --output")
            else:
                print(f"capture {split}/{record.task_type}/{record.sample_id}: all heads, {len(ids)-start} response tokens", flush=True)
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(args.model), local_files_only=True, torch_dtype=getattr(torch, args.dtype),
                        attn_implementation="eager").to(args.device).eval()
                trace = capture_rhythm(model, ids, start, evidence, config=config,
                                       query_chunk=args.query_chunk, map_tokens=args.map_tokens if plot else 0,
                                       map_offset=args.map_offset, explicit_heads=explicit_heads,
                                       paper_groups=args.paper_groups)
                trace.update(settings=np.array(json.dumps(settings)), evidence_mask=evidence,
                             sample_id=np.array(record.sample_id), source_id=np.array(record.source_id),
                             task_type=np.array(record.task_type), full_response_tokens=np.array(full_count),
                             generator_model=np.array(getattr(record, "generator_model", "unknown")),
                             token_text=np.array([tokenizer.decode([int(t)]) for t in ids]))
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **trace)
                temporary.replace(path)
            audit = analyze_rhythm(trace)
            np.savez_compressed(path.with_suffix(".audit.npz"), **audit)
            save_json(path.with_suffix(".events.json"), event_examples(trace, audit))
            if plot:
                plot_sample(trace, audit, path.with_suffix(".png"), f"{split}/{record.task_type}/{record.sample_id}")
            entries.append({"split": split, "sample_id": record.sample_id, "source_id": record.source_id,
                            "task_type": record.task_type, "path": str(path.relative_to(args.output)),
                            "response_tokens": len(ids) - start, "full_response_tokens": full_count,
                            "waad_peak_rate": audit["waad_peak_count"] / (len(ids) - start),
                            "bucket_missed_fraction": audit["bucket_missed_fraction"],
                            "attention_message_waad_mae": audit["attention_message_waad_mae"],
                            "coupling_pair_mean_lift": _finite_average(audit["coupling_lift"]),
                            "coupling_full_horizon_mean_lift": _finite_average(audit["coupling_full_horizon_lift"])} )
            print(f"{split}/{record.task_type}/{record.sample_id} tokens={len(ids)-start}/{full_count} "
                  f"head_WAAD_peaks={int(audit['waad_peak_count'].sum())} "
                  f"bucket_miss={_finite_average(audit['bucket_missed_fraction']):.3f}", flush=True)
    # Capture, head selection, peaks, maps and index are frozen before opening labels.
    save_json(args.output / "index.json", {"labels_used_for_capture": False, "samples": entries})
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.evaluate:
        for split, dataset in datasets.items():
            store = ScanLabelStore(dataset) if args.scans else None
            label_dataset = None if args.scans else open_research_dataset(
                args.cache / split, device="cpu", retain_embedded_labels=True)
            for entry in (e for e in entries if e["split"] == split):
                if args.scans:
                    labels = store.load(dataset.load(entry["sample_id"], fields=()))
                else:
                    lookup = label_dataset.prepare_evaluation_labels([entry["sample_id"]])
                    sample = label_dataset[entry["sample_id"]]
                    try:
                        labels = lookup.response_labels(sample).detach().cpu().numpy()
                    finally:
                        sample.release_attention()
                labels = labels[:entry["response_tokens"]]
                with np.load(args.output / entry["path"], allow_pickle=False) as stored:
                    entry.update(label_observations(dict(stored), labels))
    return _summary(entries, args.output, args.bootstrap, args.seed)


if __name__ == "__main__":
    run(parser().parse_args())
