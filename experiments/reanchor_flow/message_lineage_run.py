"""Reuse completed v3 caches: propagate material vectors, then evaluate N/H."""
import argparse
from html import escape
import json
from pathlib import Path
import shutil

import numpy as np
from tqdm.auto import tqdm

from .attention_audit_run import analysis_manifest
from .attention_rhythm_report import save_json
from .message_lineage import SCHEMA, CheckpointWeights, propagate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=Path("experiments/reanchor_flow/outputs/attention_audit_v3"))
    p.add_argument("--output", type=Path, help="default: AUDIT/message_lineage")
    p.add_argument("--model", type=Path, help="capture checkpoint; defaults to the saved model path")
    p.add_argument("--phase", choices=("all", "propagate", "evaluate"), default="all")
    p.add_argument("--completed-only", action="store_true", help="use completed captures/lineages, preserving the original resume index")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query-chunk", type=int, default=16)
    p.add_argument("--top-k", type=int, default=4, help="display budget only; propagation is unpruned")
    p.add_argument("--mlp-rule", choices=("symmetric", "up"), default="symmetric")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--match-window", type=int, default=32)
    p.add_argument("--plots-per-class", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    return p


def run(args):
    import torch
    if min(args.query_chunk, args.top_k, args.match_window, args.cpu_threads) < 1 or min(args.bootstrap, args.plots_per_class) < 0:
        raise ValueError("invalid chunk/display/evaluation budget")
    torch.set_num_threads(args.cpu_threads)
    output = args.output or args.audit / "message_lineage"
    if output.resolve() == args.audit.resolve():
        raise ValueError("derived lineage output must differ from the native audit directory")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.audit / "index.json").read_text())
    if original.get("audit_schema") != 3:
        raise ValueError("message lineage needs v3 full-state captures; v2 curves cannot reconstruct it")
    if not original["settings"].get("save_states") and args.phase != "evaluate":
        raise ValueError("capture used --discard-states; material message propagation needs saved native values/residuals")
    manifest = analysis_manifest(args.audit, original, args.completed_only,
                                 required_suffixes=(".npz",) if args.phase == "evaluate" else None)
    model = args.model or Path(original["settings"]["model"])
    settings = dict(schema=SCHEMA, model=str(model), mlp_rule=args.mlp_rule, top_k=args.top_k,
                    attribution="fixed attention/RMS denominator; material prompt-V boundary; signed values")
    ready, missing, weights = [], [], None
    progress = tqdm(manifest["samples"], desc="material message propagation", unit="sample")
    for e in progress:
        native, target = args.audit / e["path"], output / e["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        identity = f"{e['split']}/{e['task_type']}/{e['sample_id']}"
        if target.exists():
            with np.load(target) as stored, np.load(native) as capture:
                saved_settings = json.loads(str(stored["settings"]))
                if args.phase == "evaluate" and not ready:
                    settings = saved_settings
                if (saved_settings != settings
                        or not np.array_equal(stored["token_ids"], capture["token_ids"])
                        or not np.array_equal(stored["material_roots"], capture["evidence_mask"] & ~capture["special_mask"])):
                    raise ValueError(f"{target}: cached decomposition differs; choose another --output")
            progress.set_postfix_str(identity + " cached")
        elif args.phase == "evaluate":
            missing.append(e["path"])
            continue
        else:
            if weights is None:
                print(f"Reading checkpoint weights on {args.device}; no model inference, no ablation", flush=True)
                weights = CheckpointWeights(model, args.device)
            result = propagate(native, weights, chunk=args.query_chunk, top_k=args.top_k, mlp_rule=args.mlp_rule,
                               progress=lambda done, total: progress.set_postfix_str(f"{identity} layer={done}/{total}", refresh=True))
            result["settings"] = np.array(json.dumps(settings))
            progress.set_postfix_str(identity + " saving", refresh=True)
            temp = target.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **result)
            temp.replace(target)
            del result
        ready.append(dict(e))
    del weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if missing and not args.completed_only:
        raise ValueError("some message lineages are unfinished; use --completed-only or resume --phase all")
    if not ready:
        raise ValueError("no completed message lineages to evaluate")
    coverage = {**manifest["analysis_coverage"], "native_completed_samples": len(manifest["samples"]),
                "completed_samples": len(ready), "lineage_missing": missing,
                "skipped_samples": len(original["samples"]) - len(ready),
                "partial": len(ready) != len(original["samples"])}
    for group, counts in coverage["groups"].items():
        selected = [e for e in ready if e["split"] + "/" + e["task_type"] == group]
        counts["lineage_completed_samples"] = len(selected)
        counts["lineage_completed_tokens"] = sum(e["response_tokens"] for e in selected)
    manifest = {**manifest, "samples": ready, "analysis_coverage": coverage, "lineage_settings": settings,
                "native_audit": str(args.audit.resolve())}
    save_json(output / "index.json", manifest)
    if args.phase == "propagate":
        return manifest
    # Labels are only opened after every selected graph and score has finished.
    absent = [e["path"] for e in ready if not (args.audit/e["path"]).with_suffix(".labels.npz").exists()]
    if absent:
        raise ValueError("lineages saved, but evaluation labels are missing; run attention_audit_run --phase analyze --completed-only first")
    from .message_lineage_report import evaluate, render_sample
    report = evaluate(args.audit, output, manifest, bootstrap=args.bootstrap, match_window=args.match_window, seed=args.seed)
    counts, links = {}, []
    for e in ready:
        with np.load((args.audit / e["path"]).with_suffix(".labels.npz")) as file, np.load(args.audit / e["path"]) as trace:
            ordinary = ~trace["special_mask"][int(trace["response_start"]):]
            y = file["labels"][ordinary]
            cls = "H" if (y == 1).any() else "N" if len(y) and (y == 0).all() else "unknown"
        group = e["split"] + "/" + e["task_type"] + "/" + cls
        count = counts.get(group, 0)
        if cls != "unknown" and count < args.plots_per_class:
            html = render_sample(args.audit / e["path"], output / e["path"])
            links.append(f'<li><a href="{escape(html.relative_to(output).as_posix(), quote=True)}">{escape(group)}/{escape(e["sample_id"])}</a></li>')
            counts[group] = count + 1
    (output / "gallery.html").write_text('<!doctype html><meta charset="utf-8"><h1>材料消息与正常／幻觉比较</h1>'
        + f'<p>完成 {len(ready)}/{len(original["samples"])} 个计划样本。<a href="summary.md">评估表</a>；<a href="summary.json">完整统计</a></p>'
        + '<p>示例按标签平衡展示；标签不参与传播和评分。任意样本/head 可用 view_message_lineage.ipynb 查看。</p><ul>'
        + ''.join(links) + '</ul>', encoding="utf-8")
    notebook = Path(__file__).with_name("view_message_lineage.ipynb")
    shutil.copyfile(notebook, output / notebook.name)
    return report


if __name__ == "__main__":
    run(parser().parse_args())
