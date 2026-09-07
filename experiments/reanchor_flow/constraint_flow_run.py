"""Small controlled constraint audit, or externally reviewed aligned JSONL pairs.

This is a mechanism experiment, not a RAGTruth detector or an AUROC benchmark.
Run --phase analyze on saved traces without loading model weights or a tokenizer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from itertools import permutations
from pathlib import Path
import re

import numpy as np
from tqdm.auto import tqdm

from .attention_rhythm_report import save_json, source_bootstrap


def synthetic_records(sources):
    """Known binding, with consistent renaming, irrelevant-fact and wording controls.

    The common answer prefix contains no color commitment. All four comparisons
    share condition 0 and the same registered color contrast. These are designed
    interventions, not natural hallucinated/grounded samples.
    """
    colors = ("red", "blue", "green", "black", "white", "brown")
    assignments = list(permutations(colors, 4))
    if sources > len(assignments):
        raise ValueError("synthetic source budget exceeds the distinct color assignments")
    order = np.random.default_rng(2026).permutation(len(assignments))
    records = []
    for i in range(sources):
        a, b, c, d = assignments[order[i]]
        first, second = (("A", "B") if i % 2 == 0 else ("B", "A"))
        facts = [f"Object {first} is {a}.", f"Object {second} is {b}.", f"Object C is {c}."]
        order_in_prompt = np.random.default_rng(3026+i).permutation(3)
        text = ("Records:\n" + "\n".join(facts[j] for j in order_in_prompt)
                + f"\nQuestion: What color is object {first}?\nAnswer from the records.")
        alternatives = {
            "fact_flip": text.replace(f"What color is object {first}?", f"What color is object {second}?"),
            # Same A/B query edit as fact_flip, but rename ALL mentions so binding
            # and the correct color remain unchanged. Tests lexical shortcuts.
            "rename": re.sub(r"\b[AB]\b", lambda m: {"A": "B", "B": "A"}[m[0]], text),
            "irrelevant": text.replace(f"Object C is {c}.", f"Object C is {d}."),
            "wording": text.replace("Records:", "Details:"),
        }
        for kind, alternative in alternatives.items():
            records.append({"source_id": f"binding_{i:03d}", "kind": kind,
                            "condition0": text, "condition1": alternative,
                            "response_prefix": "Based on the records, the color of the requested object is",
                            "negative": a, "positive": b, "prefix_validated": True,
                            "relation": "entity-color binding", "dataset": "controlled_synthetic"})
    return records


def encode_pair(record, tokenizer):
    from .constraint_flow import ConstraintPair

    if record.get("prefix_validated") is not True:
        raise ValueError("review both prefixes for semantic validity before capture")
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": record[name]}],
               tokenize=True, add_generation_prompt=True) for name in ("condition0", "condition1")]
    if len(prompts[0]) != len(prompts[1]):
        raise ValueError("paired prompts need exact token alignment; do not pad or silently truncate")
    continuation = tokenizer.encode(record["response_prefix"], add_special_tokens=False)
    choices = [tokenizer.encode(" " + record[name], add_special_tokens=False)
               for name in ("negative", "positive")]
    common = 0
    while common < min(map(len, choices)) and choices[0][common] == choices[1][common]:
        common += 1
    if common == min(map(len, choices)):
        raise ValueError("candidates need a distinct next token; prefix-of-other candidates require a separate branch protocol")
    continuation += choices[0][:common]
    ids = np.asarray([prompt + continuation for prompt in prompts], dtype=np.int64)
    return ConstraintPair(ids, len(prompts[0]), choices[0][common], choices[1][common]).check()


def _load_model(args):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager").to(args.device).eval()
    if model.config.model_type != "llama":
        raise ValueError("this native forward is validated for Llama; other architectures need their own validation")
    return model


def capture(args):
    import torch
    from transformers import AutoTokenizer
    from .constraint_flow import capture_constraint_pair, select_constraint_paths, confirm_paths, PATH_COLUMNS

    if args.pairs:
        records = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    else:
        records = synthetic_records(args.synthetic_sources)
    groups = defaultdict(list)
    for record in records:
        groups[record.get("group_id", record["source_id"])].append(record)
    if not groups:
        raise ValueError("no comparison groups")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model, entries = None, []
    settings = {"constraint_schema": 1, "model": str(args.model), "dtype": args.dtype,
                "query_chunk": args.query_chunk}
    for group_index, (group_id, group) in enumerate(tqdm(groups.items(), desc="constraint sources", unit="source")):
        facts = [r for r in group if r["kind"] == "fact_flip"]
        if len(facts) != 1:
            raise ValueError("each group needs exactly one fact_flip reference; controls use its frozen paths")
        ordered = facts + [r for r in group if r["kind"] != "fact_flip"]
        reference, paths, roles = None, None, None
        for record in tqdm(ordered, desc=str(group_id), unit="pair", leave=False):
            pair = encode_pair(record, tokenizer)
            if reference is None:
                reference = pair
            elif (not np.array_equal(pair.token_ids[0], reference.token_ids[0])
                  or pair.response_start != reference.response_start
                  or pair.positive_token_id != reference.positive_token_id
                  or pair.negative_token_id != reference.negative_token_id):
                raise ValueError("controls must share the exact baseline prefix and fixed candidates")
            name = f"pair_{len(entries):04d}"
            path = args.output / (name + ".npz")
            metadata = {**record, "group_id": str(group_id), "settings": settings}
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    trace = dict(saved)
                if json.loads(str(trace["metadata"])) != metadata or not np.array_equal(trace["token_ids"], pair.token_ids):
                    raise ValueError("saved pair differs; use a new output directory")
            else:
                if model is None:
                    model = _load_model(args)
                roots = paths[:, 0] if paths is not None else ()
                trace = capture_constraint_pair(model, pair, query_chunk=args.query_chunk, roots=roots)
                if paths is None:
                    paths, roles = select_constraint_paths(model, trace)
                trace.update(paths=paths.copy(), path_roles=roles.copy(), path_columns=np.asarray(PATH_COLUMNS),
                             metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
                             token_text=np.asarray([tokenizer.decode([int(t)]) for t in pair.token_ids[0]]),
                             token_text_pair=np.asarray([[tokenizer.decode([int(t)]) for t in ids] for ids in pair.token_ids]),
                             candidate_text=np.asarray([tokenizer.decode([pair.negative_token_id]), tokenizer.decode([pair.positive_token_id])]))
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **trace)
                temporary.replace(path)
            if paths is None:
                paths, roles = trace["paths"], trace["path_roles"]
            elif not np.array_equal(trace["paths"], paths):
                raise ValueError("a control cannot select a different path from its fact_flip reference")
            confirmation = path.with_suffix(".confirmation.npz")
            if group_index < args.confirm_sources and not confirmation.exists():
                if model is None and len(paths):
                    model = _load_model(args)
                # The native pair and its frozen paths already exist on disk.
                measured = confirm_paths(model, trace, paths, query_chunk=args.query_chunk)
                np.savez_compressed(confirmation, paths=paths, path_roles=roles, **measured)
            entries.append({"path": path.name, "source_id": record["source_id"],
                            "group_id": str(group_id), "kind": record["kind"],
                            "dataset": record.get("dataset", "externally_reviewed"),
                            "confirmation": confirmation.name if confirmation.exists() else None})
            del trace
    manifest = {"constraint_run_schema": 1, "labels_used": False, "entries": entries,
                "settings": settings, "confirmation_source_budget": args.confirm_sources}
    save_json(args.output / "index.json", manifest)
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest


def analyze(args, manifest):
    from .constraint_flow_plot import plot_pair, plot_cohort, write_gallery

    entries, pairs, gallery = [], defaultdict(dict), []
    for item in tqdm(manifest["entries"], desc="constraint reports", unit="pair"):
        path = args.output / item["path"]
        with np.load(path, allow_pickle=False) as stored:
            trace = dict(stored)
        confirmation = None
        if item["confirmation"]:
            with np.load(args.output / item["confirmation"], allow_pickle=False) as stored:
                confirmation = dict(stored)
            if not np.array_equal(confirmation["paths"], trace["paths"]):
                raise ValueError("confirmation paths differ from the frozen capture")
        m0, m1 = map(float, trace["margin_pair"])
        row = {**item, "margin0": m0, "margin1": m1, "delta": m1-m0, "absolute_delta": abs(m1-m0),
               "candidate_preference_switch": float(m0 < 0 < m1),
               "ledger_rounding": float(trace["ledger_rounding"]), "paths": trace["paths"].tolist(),
               "path_roles": trace["path_roles"].tolist(),
               "capture_seconds": float(trace["capture_seconds"]), "peak_cuda_bytes": int(trace["peak_cuda_bytes"]),
               "trace_bytes": path.stat().st_size,
               "confirmation_forward_calls": int(confirmation["confirmation_forward_calls"]) if confirmation is not None else 0,
               "confirmation_seconds": float(confirmation["confirmation_seconds"]) if confirmation is not None else 0.,
               "path_effect": confirmation["path_effect"].tolist() if confirmation is not None else None}
        entries.append(row)
        gallery.append({"path": path.with_suffix(".png").name, "title": f"{item['group_id']} / {item['kind']}",
                        "metadata": json.loads(str(trace["metadata"]))})
        pairs[item["group_id"]][item["kind"]] = row
        if not args.no_plot:
            plot_pair(trace, confirmation, path.with_suffix(".png"), f"{item['group_id']} / {item['kind']}")
    grouped = {}
    for kind in sorted({e["kind"] for e in entries}):
        chosen = [e for e in entries if e["kind"] == kind]
        grouped[kind] = {key: source_bootstrap(chosen, key, repetitions=args.bootstrap)
                         for key in ("delta", "absolute_delta", "candidate_preference_switch")}
    specificity = {}
    for kind in sorted({e["kind"] for e in entries} - {"fact_flip"}):
        comparison = [{"source_id": group["fact_flip"]["source_id"],
                       "gap": group["fact_flip"]["delta"] - group[kind]["absolute_delta"]}
                      for group in pairs.values() if "fact_flip" in group and kind in group]
        specificity[kind] = source_bootstrap(comparison, "gap", repetitions=args.bootstrap)
    report = {"scope": "aligned-prefix constraint response and bounded value-path intervention",
              "hallucination_labels_used": False, "detection_metrics_run": False,
              "groups": grouped, "fact_delta_minus_absolute_control_delta": specificity,
              "pairs": entries,
              "limits": ["synthetic templates do not establish RAGTruth hallucination mechanisms",
                         "the positive candidate is registered, not a native runner",
                         "symmetric AV terms are exact accounting, not independent causal effects",
                         "carrier projections are coordinates, not contributions to q",
                         "confirmation isolates a value-channel path and excludes Q/K-mediated paths",
                         "controls reuse fact_flip paths; no reranking after intervention outcomes",
                         "four failure classes and a single-forward detector remain unvalidated"]}
    save_json(args.output / "summary.json", report)
    lines = ["# 约束响应与路径审计", "", "这是有语义对照的机制实验；没有幻觉检测 AUROC。", "",
             "| group | condition | margin 0 | margin 1 | change | frozen paths | confirmed effects |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in entries:
        effects = ", ".join(f"{v:.5g}" for v in row["path_effect"]) if row["path_effect"] is not None else "未运行"
        lines.append(f"| {row['group_id']} | {row['kind']} | {row['margin0']:.5g} | {row['margin1']:.5g} | "
                     f"{row['delta']:.5g} | {len(row['paths'])} | {effects} |")
    lines += ["", "正方向始终是登记的 positive−negative；控制条件沿用相同候选和路径。",
              "先检查事实翻转是否改变候选偏好、是否超过一致重命名／措辞／无关事实对照，再检查冻结路径的效果。",
              "所有失败、无路径和未确认实例均保留。不要把低敏感性直接作为幻觉评分。",
              "源级区间和完整逐例数值见 summary.json；逐 head 内容／路由与原生 MLP、RMSNorm 见各 NPZ/PNG。"]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    if not args.no_plot:
        plot_cohort(entries, args.output / "cohort.png")
        write_gallery(gallery, args.output / "gallery.html")
    for kind, values in grouped.items():
        tqdm.write(f"{kind:12s} sources={values['delta']['sources']} "
                   f"mean_candidate_margin_change={float(values['delta']['mean']):+.6g}")
    tqdm.write(f"Mechanism report (not detection AUROC): {args.output / 'summary.md'}")
    return report


def parser():
    from .attention_rhythm_run import MODEL

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("all", "capture", "analyze"), default="all")
    p.add_argument("--model", type=Path, default=Path(MODEL))
    p.add_argument("--pairs", type=Path, help="externally reviewed JSONL; otherwise use the controlled synthetic suite")
    p.add_argument("--synthetic-sources", type=int, default=8)
    p.add_argument("--confirm-sources", type=int, default=0, help="first N groups, all conditions, at most 2 frozen paths each")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    p.add_argument("--query-chunk", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--output", type=Path, default=Path("experiments/reanchor_flow/outputs/constraint_flow_v1"))
    return p


def run(args):
    if args.phase == "analyze":
        manifest = json.loads((args.output / "index.json").read_text())
    else:
        if args.query_chunk < 1 or args.synthetic_sources < 1 or args.confirm_sources < 0:
            raise ValueError("query chunk/source count must be positive; confirmation budget nonnegative")
        manifest = capture(args)
    return manifest if args.phase == "capture" else analyze(args, manifest)


if __name__ == "__main__":
    run(parser().parse_args())
