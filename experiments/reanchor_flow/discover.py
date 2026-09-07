"""Capture native computation, discover unlabeled modes, then audit all test rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .artifacts import save_json, save_result
from .detection_metrics import detection_report, render_detection_report
from .native_patterns import NativePatternModel
from .native_report import (
    NativeCohort,
    native_transition_examples,
    render_native_sample,
)
from .scan_dataset import ScanDataset, ScanLabelStore


def _load(path, fields=None, *, omit_vectors=False, omit_head_codes=False):
    with np.load(path, allow_pickle=False) as archive:
        names = archive.files if fields is None else fields
        if omit_head_codes:
            names = [key for key in names if key not in {"head_code", "sketch_projection"}]
        return {key: archive[key] for key in names if not omit_vectors
                or (not key.endswith("_sketch") and key not in {"head_code", "sketch_projection"})}


def _selected(records, limit):
    counts = defaultdict(int)
    selected = []
    for record in sorted(records, key=lambda r: (r.task_type, r.sample_id)):
        if not limit or counts[record.task_type] < limit:
            selected.append(record)
            counts[record.task_type] += 1
    return selected


def _load_model(path, dtype, device):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=getattr(torch, dtype),
        attn_implementation="eager",
    ).to(device).eval()


def _load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _capture_trace(model, ids, start, args):
    from .native_trace import capture_native_trace

    return capture_native_trace(
        model, ids, start, sketch_dim=args.sketch_dim, seed=args.seed,
        query_chunk=args.query_chunk, display_edges=args.display_edges,
        save_head_codes=args.save_head_codes,
    )


def capture(args, datasets):
    """Reuse completed atomic traces; one lazily loaded model serves both splits."""
    config = {
        "model": args.model, "dtype": args.dtype, "sketch_dim": args.sketch_dim,
        "seed": args.seed, "query_chunk": args.query_chunk,
        "display_edges": args.display_edges, "save_head_codes": args.save_head_codes,
        "max_response_tokens": args.max_response_tokens,
    }
    model, tokenizer = None, None
    for split, dataset in datasets.items():
        root = args.output / split
        manifest_path = root / "native_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["config"] != config:
                raise ValueError("native capture configuration changed; choose a new --output")
        manifest = {"native_manifest_schema": 1, "split": split, "config": config,
                    "complete": False, "labels_used_for_capture": False, "entries": []}
        save_json(manifest_path, manifest)
        for record in tqdm(_selected(dataset.records, args.samples_per_task),
                           desc=f"native capture {split}", unit="sample"):
            scan = dataset.load(record.sample_id, fields=("token_ids",))
            stop = scan.response_start + args.max_response_tokens if args.max_response_tokens else None
            ids = scan["token_ids"][:stop]
            relative = Path("traces") / record.task_type / f"{record.sample_id}.npz"
            path = root / relative
            if path.exists():
                stored = _load(path, ("token_ids", "response_start", "native_trace_schema"))
                if (int(stored["native_trace_schema"]) != 1
                        or int(stored["response_start"]) != scan.response_start
                        or not np.array_equal(stored["token_ids"], ids)):
                    raise ValueError(f"existing native trace disagrees with source tokens: {path}")
            else:
                if model is None:
                    model = _load_model(args.model, args.dtype, args.device)
                    tokenizer = _load_tokenizer(args.model)
                trace = _capture_trace(model, ids, scan.response_start, args)
                trace.update(dataset_sample_id=record.sample_id, source_id=record.source_id,
                             task_type=record.task_type,
                             token_text=np.asarray([tokenizer.decode([int(token)]) for token in ids]))
                save_result(path, trace)
                del trace
            manifest["entries"].append({
                "dataset_sample_id": record.sample_id, "source_id": record.source_id,
                "task_type": record.task_type, "path": str(relative),
                "rows": len(ids) - scan.response_start,
            })
            save_json(manifest_path, manifest)
        manifest["complete"] = True
        save_json(manifest_path, manifest)
    if model is not None:
        import gc

        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fit_row_bank(output, manifests, budget, seed):
    """Equal source quotas, approximately equal samples within each source.

    Every eligible source contributes at least one row. Short sources can leave
    budget unused; they are not replicated to manufacture independent rows.
    """
    test_sources = {entry["source_id"] for entry in manifests["test"]["entries"]}
    grouped = defaultdict(list)
    excluded = set()
    for entry in manifests["train"]["entries"]:
        if entry["source_id"] in test_sources:
            excluded.add(entry["source_id"])
        else:
            grouped[entry["source_id"]].append(entry)
    if not grouped or budget < len(grouped):
        raise ValueError(f"--fit-rows must cover all {len(grouped)} disjoint train sources")
    rng = np.random.default_rng(seed)
    sources = sorted(grouped)
    rng.shuffle(sources)
    allocations = []
    for source_index, source in enumerate(sources):
        quota = budget // len(sources) + (source_index < budget % len(sources))
        entries = sorted(grouped[source], key=lambda e: e["dataset_sample_id"])
        rng.shuffle(entries)
        counts = np.zeros(len(entries), dtype=int)
        capacities = np.array([entry["rows"] for entry in entries])
        for _ in range(min(quota, int(capacities.sum()))):
            available = np.flatnonzero(counts < capacities)
            selected = available[np.argmin(counts[available])]
            counts[selected] += 1
        for entry, count in zip(entries, counts):
            if count:
                indices = np.sort(rng.choice(entry["rows"], size=int(count), replace=False))
                allocations.append((entry, indices))
    bank, selection, head_shape, mlp_shape = [], [], None, None
    for entry, indices in tqdm(allocations, desc="native fit row bank", unit="sample"):
        trace = _load(output / "train" / entry["path"], ("head_sketch", "mlp_sketch"))
        head = trace["head_sketch"]
        geometry = ((*head.shape[:2], head.shape[-1]), (head.shape[0], head.shape[-1]))
        if head_shape is not None and geometry != (head_shape, mlp_shape):
            raise ValueError("native train traces disagree on head/MLP geometry")
        head_shape, mlp_shape = geometry
        bank.append(NativePatternModel.vectorize(trace)[indices])
        selection.append({**entry, "fit_rows": len(indices), "row_indices": indices.tolist()})
    partition = {
        "train_sources": sorted(grouped), "test_sources": sorted(test_sources),
        "excluded_overlap_sources": sorted(excluded), "rows_budget": budget,
        "rows_used": sum(len(rows) for _, rows in allocations), "samples": selection,
        "sampling": "equal source quotas; approximately equal sample quotas; seeded uniform random rows without replacement",
        "labels_used": False,
    }
    return np.concatenate(bank), head_shape, mlp_shape, partition


def _manifests(output):
    manifests = {split: json.loads((output / split / "native_manifest.json").read_text())
                 for split in ("train", "test")}
    for split, manifest in manifests.items():
        if (manifest["native_manifest_schema"] != 1 or not manifest["complete"]
                or manifest["labels_used_for_capture"] or manifest["split"] != split
                or not manifest["entries"]):
            raise ValueError(f"{split} native capture must be complete and label-free")
    if manifests["train"]["config"] != manifests["test"]["config"]:
        raise ValueError("train/test native capture configurations differ")
    return manifests


def _prediction_path(output, split, entry):
    return output / split / "predictions" / entry["task_type"] / f"{entry['dataset_sample_id']}.npz"


def analyze(args, datasets):
    """Freeze fit and all predictions before opening any correctness labels."""
    manifests = _manifests(args.output)
    bank, head_shape, mlp_shape, partition = fit_row_bank(
        args.output, manifests, args.fit_rows, args.seed,
    )
    for _ in tqdm(range(1), desc="native joint-mode fit", unit="model"):
        model = NativePatternModel.fit(bank, head_shape, mlp_shape,
                                       n_components=args.components,
                                       n_patterns=args.patterns, seed=args.seed)
    del bank
    model.save(args.output / "native_patterns.npz")
    save_json(args.output / "source_partition.json", partition)
    save_result(args.output / "pattern_geometry.npz", {
        **{f"center_{k}": v for k, v in model.inverse_pattern_centers().items()},
        **{f"loading_{k}": v for k, v in model.component_loadings().items()},
    })
    for split, manifest in manifests.items():
        for entry in tqdm(manifest["entries"], desc=f"native predict {split}", unit="sample"):
            trace = _load(args.output / split / entry["path"],
                          ("head_sketch", "mlp_sketch", "row_position", "response_start"))
            prediction = model.transform(trace)
            prediction.update(row_position=trace["row_position"],
                              response_index=trace["row_position"] + 1 - trace["response_start"])
            save_result(_prediction_path(args.output, split, entry), prediction)
            example_path = args.output / split / "examples" / entry["task_type"] / f"{entry['dataset_sample_id']}.json"
            save_json(example_path, {"dataset_sample_id": entry["dataset_sample_id"],
                                     "labels_used": False,
                                     "transition_examples": native_transition_examples(trace, prediction)})
    # Label-free display selection is frozen before constructing the label store.
    plot_ids, counts = set(), defaultdict(int)
    for entry in sorted(manifests["test"]["entries"], key=lambda e: (e["task_type"], e["dataset_sample_id"])):
        if not args.plots_per_task or counts[entry["task_type"]] < args.plots_per_task:
            plot_ids.add(entry["dataset_sample_id"])
            counts[entry["task_type"]] += 1
    save_json(args.output / "plot_selection.json", {"sample_ids": sorted(plot_ids),
                                                   "labels_used": False})
    labels_store = ScanLabelStore(datasets["test"], dataset_root=args.cache / "test" if args.cache else None)
    cohort = NativeCohort(len(model.centers), seed=args.seed, bootstrap=args.bootstrap)
    collected = defaultdict(list)
    plotted = []
    for entry in tqdm(manifests["test"]["entries"], desc="native test label audit", unit="sample"):
        sample_id = entry["dataset_sample_id"]
        trace = _load(args.output / "test" / entry["path"], omit_vectors=True)
        prediction = _load(_prediction_path(args.output, "test", entry))
        index = prediction["response_index"]
        scan = datasets["test"].load(sample_id, fields=())
        labels = labels_store.load(scan)[index]
        cohort.add(sample_id, entry["source_id"], entry["task_type"], index, labels, prediction, trace)
        for key, value in {
            "labels": labels, "pattern_distance": prediction["distance"],
            "pattern_transition": prediction["transition"],
            "low_native_margin": -trace["final_margin"], "position": index,
            "source": np.repeat(entry["source_id"], len(index)),
            "task": np.repeat(entry["task_type"], len(index)),
        }.items():
            collected[key].append(value)
        if sample_id in plot_ids:
            plotted.append((entry, labels))
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    scores = {key: arrays[key] for key in ("pattern_distance", "pattern_transition", "low_native_margin", "position")}
    detection = detection_report(arrays["labels"], scores, arrays["source"], arrays["task"],
                                 primary="pattern_distance", bootstrap=args.bootstrap, seed=args.seed)
    detection["status"] = "fixed-direction novelty diagnostics; not a validated hallucination detector"
    report = cohort.report()
    report["discovery"] = {
        "fit_rows": model.fit_rows, "head_shape": head_shape, "mlp_shape": mlp_shape,
        "explained_variance_ratio": model.explained_variance_ratio.tolist(),
        "labels_used_for_fit": False, "head_averaging": False,
        "representation": "all signed head and MLP vector sketches; fixed random residual projection; train layer RMS; joint PCA and KMeans",
        "limits": "Lossy observational modes. Direct-logit accounting is neither factual support nor causal attribution. No missed-reanchor failure class is assigned.",
    }
    save_json(args.output / "mechanism_report.json", report)
    save_json(args.output / "detection_report.json", detection)
    if not args.no_plot:
        from .native_report import plot_native_mode_writes

        cohort.plot(args.output / "mechanism_cohort.png", report)
        for task in tqdm(report["tasks"], desc="native mode plots", unit="task"):
            plot_native_mode_writes(args.output / f"mode_writes_{task}.png", report, task=task)
        render_detection_report(args.output / "detection_curves.png", arrays["labels"], scores,
                                arrays["task"], primary="pattern_distance")
        for entry, labels in tqdm(plotted, desc="native sample plots", unit="figure"):
            trace = _load(args.output / "test" / entry["path"], omit_head_codes=True)
            prediction = _load(_prediction_path(args.output, "test", entry))
            path = args.output / "figures" / entry["task_type"] / f"{entry['dataset_sample_id']}.png"
            metadata = render_native_sample(path, trace, prediction, labels=labels)
            save_json(path.with_suffix(".json"), metadata)
    lines = ["# Native computation discovery", "", report["discovery"]["limits"], "",
             f"Fit rows: {model.fit_rows}; retained sketch variance: {model.explained_variance_ratio.sum():.4f}.",
             "", "All modes are reported. H/N counts are descriptive. H−N prevalence gaps compare mode membership within matched position bins; uncertainty resamples complete sources. Global BH q includes every task/mode comparison and ALL.",
             "", "| Task | Mode | H tokens | N tokens | Matched H−N | 95% CI | Global q | Paired sources |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for task, group in report["tasks"].items():
        for mode in group["patterns"]:
            lines.append(f"| {task} | {mode['pattern_id']} | {mode['tokens_hallucinated']} | {mode['tokens_nonhallucinated']} | {mode['position_matched_h_minus_n']} | [{mode['ci_lower']}, {mode['ci_upper']}] | {mode['bh_q']} | {mode['paired_sources']} |")
    lines.extend(["", "## Diagnostic discrimination", "",
                  "These fixed-direction novelty scores are diagnostics, not validated mechanism-based detectors.",
                  "", "| Task | Diagnostic | AUROC | AUPRC |", "|---|---|---:|---:|"])
    for task, group in detection["groups"].items():
        for name, metrics in group["scores"].items():
            lines.append(f"| {task} | {name} | {metrics['auroc']} | {metrics['auprc']} |")
        print(f"{task}: tokens={group['tokens']} pattern_distance AUROC={group['scores']['pattern_distance']['auroc']} AUPRC={group['scores']['pattern_distance']['auprc']}")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, detection


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("experiments/reanchor_flow/outputs")
    parser.add_argument("--scans", type=Path, default=base / "mechanism_all_v3")
    parser.add_argument("--output", type=Path, default=base / "native_discovery_v1")
    parser.add_argument("--phase", choices=("all", "capture", "analyze"), default="all")
    parser.add_argument("--model")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda")
    for name, default in (("query-chunk", 8), ("sketch-dim", 16), ("seed", 2026),
                          ("display-edges", 2), ("samples-per-task", 0),
                          ("max-response-tokens", 0), ("components", 8),
                          ("patterns", 6), ("fit-rows", 4096), ("bootstrap", 200)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--plots-per-task", type=int, default=4, help="label-free ID order; 0 plots all captured samples")
    parser.add_argument("--save-head-codes", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cache", type=Path, help="research dataset root containing train/ and test/")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (min(args.query_chunk, args.sketch_dim, args.components, args.patterns, args.fit_rows) < 1
            or min(args.display_edges, args.samples_per_task, args.max_response_tokens, args.plots_per_task) < 0
            or args.bootstrap < 2):
        raise ValueError("sizes must be positive, limits nonnegative, bootstrap at least two")
    datasets = {split: ScanDataset(args.scans / split) for split in ("train", "test")}
    args.model = args.model or datasets["train"].config.get("model")
    args.dtype = args.dtype or datasets["train"].config.get("model_dtype", "bfloat16")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "capture"):
        if not args.model:
            raise ValueError("--model is required when the old scan manifest has no model")
        capture(args, datasets)
    if args.phase in ("all", "analyze"):
        return analyze(args, datasets)


if __name__ == "__main__":
    main()
