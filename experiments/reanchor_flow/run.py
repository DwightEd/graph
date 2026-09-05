"""Foreground CLI for the complete re-anchor mechanism audit."""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common.ragtruth_alignment import TASK_TYPES

from .analyze import analyze_split
from .audit import audit_target, save_audit
from .detection import run_detection
from .evaluate import evaluate_results
from .flow import FlowSignal
from .subset import run_subset_split
from .subset_report import evaluate_subset_split
from .worlds import load_world

MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
)
CACHE = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)
SOURCE_INFO = Path(
    "/share/home/tm902089733300000/a903202310/lys/data/"
    "RAGTruth/dataset/source_info.jsonl"
)
DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def output_root(args) -> Path:
    if args.output:
        return args.output
    model = getattr(args, "model", MODEL)
    root = Path(__file__).resolve().parent / "outputs" / model.name / "mechanism_v8"
    return root / "smoke" if getattr(args, "smoke", False) else root


def selected_splits(args) -> tuple[str, ...]:
    return ("train", "test") if args.split == "all" else (args.split,)


def split_output(args, split: str) -> Path:
    root = output_root(args)
    return root / split if args.split == "all" else root


def subset_output_root(args) -> Path:
    if args.output:
        return args.output
    return (
        Path(__file__).resolve().parent
        / "outputs"
        / args.model.name
        / "native_subset_v1"
    )


def subset_split_output(args, split: str) -> Path:
    return subset_output_root(args) / split


def load_model(path: Path, device: str, dtype: str):
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            torch_dtype=DTYPE[dtype],
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    return model, tokenizer


def analyze(args) -> dict:
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    limit = 1 if args.smoke and args.limit is None else args.limit
    max_events = 96 if args.smoke and args.max_events is None else args.max_events
    mechanism_limit = (
        1 if args.smoke and args.mechanism_limit == 0 else args.mechanism_limit
    )
    captured = {}
    for split in selected_splits(args):
        counts = analyze_split(
            model,
            tokenizer,
            args.cache / split,
            args.source_info,
            split_output(args, split),
            model_path=str(args.model),
            model_id=args.model.name,
            dtype=args.dtype,
            limit=limit,
            max_events=max_events,
            query_chunk=args.query_chunk,
            route_window=args.route_window,
            future_horizon=args.future_horizon,
            distance_scale=args.distance_scale,
            peak_quantile=args.peak_quantile,
            max_lag=args.max_lag,
            plot_limit=args.plot_limit,
            plot_sample_id=args.plot_sample_id,
            mechanism_limit=mechanism_limit,
        )
        captured[split] = counts
        print(
            f"captured {split} "
            + " ".join(f"{task}={count}" for task, count in counts.items())
        )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return captured


def number(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def interval(summary: dict) -> str:
    low, high = summary["ci95"]
    return f"[{number(low)},{number(high)}]"


def effect(name: str, summary: dict) -> str:
    return f"{name}={number(summary['mean'])} CI={interval(summary)}"


def evaluate(args) -> dict:
    all_reports = {}
    seed_offset = {"test": 0, "train": 1000}
    for split in selected_splits(args):
        reports = evaluate_results(
            split_output(args, split),
            args.cache / split,
            bootstrap=args.bootstrap,
            seed=args.seed + seed_offset[split],
            curve_radius=args.curve_radius,
        )
        all_reports[split] = reports
        print(f"\n=== {split.upper()} ===")
        for task, report in reports.items():
            print_report(task, report)
    return all_reports


def detect(args) -> dict:
    report = run_detection(
        output_root(args),
        args.cache,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print("\n=== FROZEN TRAIN→TEST DETECTION ===")
    for task in ("QA", "Summary", "Data2txt", "ALL"):
        task_report = report["tasks"][task]
        token = task_report["token"]["scores"]["online_failure"]
        onset = task_report["onset"]["scores"]["onset_trigger"]
        print(
            f"{task:9s} samples={task_report['samples']} "
            f"tokens={token['tokens']} positives={token['positives']} "
            f"token_AUROC={number(token['auroc'])} "
            f"token_AUPRC={number(token['auprc'])} "
            f"onset_AUROC={number(onset['auroc'])} "
            f"onset_AUPRC={number(onset['auprc'])}"
        )
    return report


def corridor(args) -> dict:
    """Run the paired evidence-to-target causal corridor audit."""

    world = load_world(args.pair)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    if Path(world.tokenizer_id).name != Path(tokenizer.name_or_path).name:
        raise ValueError(
            "paired-world tokenizer does not match the loaded model tokenizer"
        )
    output = args.output or args.pair.parent / "etcc_outputs"
    output.mkdir(parents=True, exist_ok=True)
    signal = FlowSignal(args.flow_signal)
    reports = {}
    for target in world.targets:
        result = audit_target(
            model,
            world,
            target,
            signal,
            carrier_scope=args.carrier_scope,
            coverage=args.edge_coverage,
            gradient_steps=args.gradient_steps,
            query_chunk=args.query_chunk,
            root_screen_limit=args.root_screen_limit,
            carrier_limit=args.carrier_limit,
            materialize_messages=args.materialize_messages,
        )
        destination = output / (
            f"{world.sample_id}_q{target.query_position}"
            f"_a{target.positive_token_id}_b{target.negative_token_id}"
            f"_{signal.value}.npz"
        )
        save_audit(
            destination,
            world,
            result,
            model_id=str(args.model),
            model_dtype=args.dtype,
            coverage=args.edge_coverage,
            gradient_steps=args.gradient_steps,
            carrier_scope=args.carrier_scope,
            query_chunk=args.query_chunk,
            root_screen_limit=args.root_screen_limit,
            carrier_limit=args.carrier_limit,
            materialize_messages=args.materialize_messages,
        )
        effect = result.effect
        report_key = (
            f"q{target.query_position}"
            f"_a{target.positive_token_id}_b{target.negative_token_id}"
        )
        reports[report_key] = {
            "output": destination,
            "pair_effect": effect.pair_effect,
            "selected_root_unit_id": result.selected_root_unit_id,
            "selected_root_confirmed": result.selected_root_confirmed,
            "edges": result.flow.edges.count,
            "corridor_edges": effect.edge_count,
            "corridor_confirmed": result.corridor_confirmed,
            "necessity": effect.necessity,
            "sufficiency": effect.sufficiency,
            "mediated_sufficiency": effect.mediated_sufficiency,
            "restoration_error": effect.restoration_error,
            "restoration_valid": effect.restoration_valid,
        }
        print(
            f"q={target.query_position} signal={signal.value} "
            f"pair={number(effect.pair_effect)} "
            f"root_unit={result.selected_root_unit_id} "
            f"root_confirmed={result.selected_root_confirmed} "
            f"edges={result.flow.edges.count} "
            f"corridor={effect.edge_count} necessity={number(effect.necessity)} "
            f"corridor_confirmed={result.corridor_confirmed} "
            f"sufficiency={number(effect.sufficiency)} "
            f"mediated={number(effect.mediated_sufficiency)} "
            f"restore_error={number(effect.restoration_error)} "
            f"restore_valid={effect.restoration_valid}"
        )
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reports


def audit_subset(args) -> dict:
    """Run the label-free native mechanism pilot on selected real samples."""

    model, tokenizer = load_model(args.model, args.device, args.dtype)
    tasks = TASK_TYPES if args.task == "all" else (args.task,)
    reports = {}
    for split in selected_splits(args):
        counts = run_subset_split(
            model,
            tokenizer,
            args.cache / split,
            args.source_info,
            subset_split_output(args, split),
            split=split,
            model_path=args.model,
            model_dtype=args.dtype,
            tasks=tasks,
            samples_per_task=args.samples_per_task,
            sample_ids=tuple(args.sample_id or ()),
            selection_seed=args.selection_seed,
            targets_per_sample=args.targets_per_sample,
            target_policy=args.target_policy,
            max_response_tokens=(
                None if args.max_response_tokens == 0 else args.max_response_tokens
            ),
            signal=FlowSignal(args.flow_signal),
            carrier_scope=args.carrier_scope,
            coverage=args.edge_coverage,
            query_chunk=args.query_chunk,
            root_screen_limit=args.root_screen_limit,
            carrier_limit=args.carrier_limit,
            saved_edges=args.saved_edges,
        )
        reports[split] = counts
        print(
            f"subset {split}: samples={counts['samples']} "
            f"targets={counts['targets']} resumed={counts['resumed']} "
            f"corridors_confirmed={counts['confirmed']}"
        )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reports


def evaluate_subset(args) -> dict:
    """Open labels only after native subset artifacts have been frozen."""

    reports = {}
    for split in selected_splits(args):
        report = evaluate_subset_split(
            args.cache / split,
            subset_split_output(args, split),
        )
        reports[split] = report
        print(f"\n=== NATIVE SUBSET {split.upper()} ===")
        for task in ("QA", "Summary", "Data2txt", "ALL"):
            groups = report["groups"][task]
            clean = groups["clean"]
            hallucinated = groups["hallucinated"]
            print(
                f"{task:9s} clean={clean['targets']} "
                f"hallucinated={hallucinated['targets']} "
                f"root_ok={number(groups['all']['root_confirmed_rate'])} "
                f"corridor_ok={number(groups['all']['corridor_confirmed_rate'])} "
                f"carrier_ok={number(groups['all']['carrier_confirmed_rate'])}"
            )
    return reports


def print_report(task: str, report: dict) -> None:
    normal = report["normal"]
    shift = normal["direct_route_shift"]
    transition = normal["internal_transition"]
    onset = report["onset_minus_matched_clean"]
    functional = report["functional"]
    mechanism = report["mechanism"]
    print(
        f"{task:9s} samples={report['samples']} tokens={report['tokens']} "
        f"positives={report['positive_tokens']} "
        f"prevalence={number(report['prevalence'])} "
        f"functional_pairs={functional['onset_pairs']} "
        f"functional_pair_sources={functional['onset_pair_sources']} "
        f"grouped_samples={mechanism['samples']} "
        f"grouped_sources={mechanism['sources']} "
        f"grouped_pairs={mechanism['onset_pairs']} "
        f"grouped_pair_sources={mechanism['onset_pair_sources']}"
    )
    print(
        "  H0 direct drift: "
        + effect("prompt_slope", shift["prompt_lift_slope"])
        + "  "
        + effect("history_slope", shift["history_lift_slope"])
        + "  "
        + effect(
            "prompt_vs_history",
            shift["conditional_prompt_history_log_odds_slope"],
        )
    )
    print(
        "  H1 transition: "
        + effect("prompt", transition["prompt_delta"])
        + "  "
        + effect("evidence", transition["evidence_delta"])
        + "  "
        + effect("predictor_reuse", transition["predictor_reuse"])
        + "  "
        + effect("emitted_anchor", transition["emitted_token_anchor"])
    )
    print(
        "  H2 onset-clean: "
        + effect("route_change", onset["route_change"])
        + "  "
        + effect("prompt", onset["prompt_delta"])
        + "  "
        + effect("evidence", onset["evidence_delta"])
        + "  "
        + effect("predictor_reuse", onset["predictor_reuse"])
        + "  "
        + effect("emitted_token_anchor", onset["emitted_token_anchor"])
    )
    if functional["samples"]:
        deep = functional["onset_minus_clean"]
        print(
            "  H3 functional context: "
            + effect("entry", deep["evidence_entry"])
            + "  "
            + effect("target_effect", deep["evidence_effect"])
            + "  "
            + effect("distribution_js", deep["context_distribution_js"])
        )
        print(
            "  H3 adoption: "
            + effect("target_logprob_gain", deep["context_target_logprob_gain"])
            + "  "
            + effect("adoption_margin", deep["context_adoption_margin"])
            + "  "
            + effect("target_log_rank", deep["context_target_log_rank"])
        )
    if mechanism["samples"]:
        deep = mechanism["onset_minus_clean"]
        print(
            "  H4 grouped/state: "
            + effect("integration", deep["evidence_prompt_interaction"])
            + "  "
            + effect("history_effect", deep["history_effect"])
            + "  "
            + effect("late_control_loss", deep["evidence_late_control_loss"])
            + "  "
            + effect("readout_gain", deep["evidence_readout_gain"])
        )
    balance = report["matching_balance"]
    print(
        "  onset matching: "
        f"pairs={balance['pairs']} sources={balance['sources']} "
        "position_gap="
        f"{number(balance['mean_absolute_relative_position_gap']['mean'])} "
        f"boundary_match={number(balance['boundary_match_fraction']['mean'])} "
        f"token_match={number(balance['token_match_fraction']['mean'])}"
    )
    decisions = report["registered_decisions"]
    print(
        "  decisions: "
        + " ".join(f"{name}={status}" for name, status in decisions.items())
    )


def run_all(args) -> dict:
    captured = analyze(args)
    detection = None
    if (
        args.split == "all"
        and not args.smoke
        and args.max_events is None
        and args.plot_sample_id is None
    ):
        detection = detect(args)
    mechanism = evaluate(args)
    return {"captured": captured, "detection": detection, "mechanism": mechanism}


def add_common(command) -> None:
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--cache", type=Path, default=CACHE)
    command.add_argument("--source-info", type=Path, default=SOURCE_INFO)
    command.add_argument("--output", type=Path)
    command.add_argument("--split", choices=("train", "test", "all"), default="test")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPE), default="bfloat16")
    command.add_argument("--limit", type=int)
    command.add_argument("--max-events", type=int)
    command.add_argument("--query-chunk", type=int, default=64)
    command.add_argument("--route-window", type=int, default=4)
    command.add_argument("--future-horizon", type=int, default=16)
    command.add_argument("--distance-scale", type=int, default=16)
    command.add_argument("--peak-quantile", type=float, default=0.9)
    command.add_argument("--max-lag", type=int, default=3)
    command.add_argument("--plot-limit", type=int, default=1)
    command.add_argument("--plot-sample-id")
    command.add_argument(
        "--mechanism-limit",
        type=int,
        default=0,
        help="deep grouped-cut samples per task; -1 means every selected sample",
    )
    command.add_argument("--smoke", action="store_true")


def add_evaluation(command) -> None:
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument("--curve-radius", type=int, default=6)


def add_detection(command) -> None:
    command.add_argument("--cache", type=Path, default=CACHE)
    command.add_argument("--output", type=Path)
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--seed", type=int, default=2026)


def add_corridor(command) -> None:
    command.add_argument("--pair", type=Path, required=True)
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--output", type=Path)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPE), default="bfloat16")
    command.add_argument(
        "--flow-signal",
        choices=tuple(signal.value for signal in FlowSignal),
        default=FlowSignal.MESSAGE.value,
        help="edge ranking: signed target effect of true messages or raw attention",
    )
    command.add_argument(
        "--carrier-scope",
        choices=("response", "all"),
        default="all",
    )
    command.add_argument("--edge-coverage", type=float, default=0.95)
    command.add_argument("--gradient-steps", type=int, default=1)
    command.add_argument("--query-chunk", type=int, default=8)
    command.add_argument(
        "--root-screen-limit",
        type=int,
        default=8,
        help="candidate roots receiving exact bidirectional patches; 0 means all",
    )
    command.add_argument("--carrier-limit", type=int, default=3)
    command.add_argument("--materialize-messages", action="store_true")


def add_subset(command, *, evaluation: bool = False) -> None:
    command.add_argument("--model", type=Path, default=MODEL)
    command.add_argument("--cache", type=Path, default=CACHE)
    command.add_argument("--output", type=Path)
    command.add_argument("--split", choices=("train", "test", "all"), default="test")
    if evaluation:
        return
    command.add_argument("--source-info", type=Path, default=SOURCE_INFO)
    command.add_argument("--task", choices=(*TASK_TYPES, "all"), default="all")
    command.add_argument("--samples-per-task", type=int, default=1)
    command.add_argument(
        "--sample-id",
        action="append",
        help="exact sample ID; repeat for a fixed cohort",
    )
    command.add_argument("--selection-seed", type=int, default=2026)
    command.add_argument("--targets-per-sample", type=int, default=1)
    command.add_argument(
        "--target-policy",
        choices=("uncertain", "low-margin", "evenly-spaced", "all"),
        default="uncertain",
    )
    command.add_argument(
        "--max-response-tokens",
        type=int,
        default=128,
        help="causal pilot horizon; 0 keeps the full response",
    )
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--dtype", choices=tuple(DTYPE), default="bfloat16")
    command.add_argument(
        "--flow-signal",
        choices=tuple(signal.value for signal in FlowSignal),
        default=FlowSignal.MESSAGE.value,
        help="transport graph uses raw attention or exact native message norm",
    )
    command.add_argument(
        "--carrier-scope", choices=("response", "all"), default="response"
    )
    command.add_argument("--edge-coverage", type=float, default=0.9)
    command.add_argument("--query-chunk", type=int, default=8)
    command.add_argument("--root-screen-limit", type=int, default=4)
    command.add_argument("--carrier-limit", type=int, default=2)
    command.add_argument(
        "--saved-edges",
        type=int,
        default=2048,
        help="top scalar corridor edges persisted; exact reruns happen first",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Complete re-anchor mechanism audit")
    commands = root.add_subparsers(dest="command", required=True)
    analyze_command = commands.add_parser("analyze")
    add_common(analyze_command)
    analyze_command.set_defaults(handler=analyze)
    evaluate_command = commands.add_parser("evaluate")
    add_common(evaluate_command)
    add_evaluation(evaluate_command)
    evaluate_command.set_defaults(handler=evaluate)
    detect_command = commands.add_parser(
        "detect", help="fit on unlabeled train and evaluate frozen test scores"
    )
    add_detection(detect_command)
    detect_command.set_defaults(handler=detect)
    corridor_command = commands.add_parser(
        "corridor", help="audit a matched clean/corrupt evidence pair"
    )
    add_corridor(corridor_command)
    corridor_command.set_defaults(handler=corridor)
    subset_command = commands.add_parser(
        "subset",
        help="audit a label-free source-diverse RAGTruth mechanism subset",
    )
    add_subset(subset_command)
    subset_command.set_defaults(handler=audit_subset)
    subset_evaluate_command = commands.add_parser(
        "subset-evaluate",
        help="join hallucination labels after subset capture is complete",
    )
    add_subset(subset_evaluate_command, evaluation=True)
    subset_evaluate_command.set_defaults(handler=evaluate_subset)
    all_command = commands.add_parser("all")
    add_common(all_command)
    add_evaluation(all_command)
    all_command.set_defaults(handler=run_all)
    return root


def validate_args(args) -> None:
    positive_names = (
        "query_chunk",
        "route_window",
        "future_horizon",
        "distance_scale",
        "max_lag",
    )
    for name in positive_names:
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("limit", "max_events"):
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "mechanism_limit") and args.mechanism_limit < -1:
        raise ValueError("--mechanism-limit must be -1 or non-negative")
    if hasattr(args, "peak_quantile") and not 0 < args.peak_quantile < 1:
        raise ValueError("--peak-quantile must lie in (0,1)")
    if hasattr(args, "plot_limit") and args.plot_limit < 0:
        raise ValueError("--plot-limit cannot be negative")
    if hasattr(args, "bootstrap") and args.bootstrap < 0:
        raise ValueError("--bootstrap cannot be negative")
    if hasattr(args, "curve_radius") and args.curve_radius < 1:
        raise ValueError("--curve-radius must be positive")
    if hasattr(args, "edge_coverage") and not 0 < args.edge_coverage <= 1:
        raise ValueError("--edge-coverage must lie in (0,1]")
    if hasattr(args, "gradient_steps") and args.gradient_steps < 1:
        raise ValueError("--gradient-steps must be positive")
    if hasattr(args, "carrier_limit") and args.carrier_limit < 0:
        raise ValueError("--carrier-limit cannot be negative")
    if hasattr(args, "root_screen_limit") and args.root_screen_limit < 0:
        raise ValueError("--root-screen-limit cannot be negative")
    for name in ("samples_per_task", "targets_per_sample"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "max_response_tokens") and args.max_response_tokens < 0:
        raise ValueError("--max-response-tokens cannot be negative")
    if hasattr(args, "saved_edges") and args.saved_edges < 0:
        raise ValueError("--saved-edges cannot be negative")
    if (
        hasattr(args, "sample_id")
        and args.sample_id
        and getattr(args, "split", None) == "all"
    ):
        raise ValueError("--sample-id requires one concrete --split")


def main() -> None:
    command_parser = parser()
    args = command_parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        command_parser.error(str(error))
    args.handler(args)


if __name__ == "__main__":
    main()
