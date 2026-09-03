"""CLI for frozen-model mechanism-state detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .collect import capture_all
from .data import TASK_TYPES
from .evaluate import SCORE_ORDER, evaluate_all, plot_saved_sample
from .shortcut import SHORTCUT_SCORE_NAMES

PRINTED_AUDIT_ORDER = (
    "causal_evidence_support",
    "causal_history_support",
    "causal_interaction",
    "raw_evidence_bypass",
    "raw_history_after_cut",
    "raw_old_symmetric",
    "raw_takeover",
    "raw_interaction",
    "prompt_edge_log_volume_mean",
    "prompt_attention_log_volume_mean",
    "register_evidence_adoption_step_principal_energy",
    "register_autonomous_history_step_principal_energy",
    "register_evidence_adoption_terminal_norm",
    "register_autonomous_history_terminal_norm",
    "register_evidence_adoption_mlp_alignment_mean",
    "register_autonomous_history_mlp_alignment_mean",
    "register_evidence_adoption_response_history_mass_mean",
    "register_autonomous_history_response_history_mass_mean",
    "register_evidence_adoption_response_history_carrier_contribution_mean",
    "register_evidence_adoption_response_history_gate_contribution_mean",
    "register_autonomous_history_response_history_root_contribution_mean",
    "register_evidence_adoption_response_history_effective_routes_mean",
    "register_autonomous_history_response_history_effective_routes_mean",
    "shortcut_history_write_norm_mean",
    "shortcut_direct_evidence_write_norm_mean",
    "shortcut_evidence_relay_write_norm_mean",
    "shortcut_autonomous_history_write_norm_mean",
    "shortcut_relay_completion_mean",
    "shortcut_route_completion_mean",
    "shortcut_route_incompleteness_mean",
    "shortcut_rewired_route_completion_mean",
    "shortcut_endpoint_rewire_gap_mean",
    "shortcut_evidence_relay_support_mean",
    "shortcut_autonomous_support_mean",
    "shortcut_additive_support_error_mean",
    "shortcut_route_candidate_mean",
    "shortcut_route_rewired_control_mean",
)

DEFAULT_MODEL = Path(
    "/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct"
)
DEFAULT_CACHE = Path(
    "/share/home/tm902089733300000/a903202310/lys/research/"
    "Unsupervised-hypergraph/outputs/attention_cache/"
    "fresh_attention_c8847872bedf_20260731T074520Z_p876"
)
DEFAULT_SOURCE_INFO = Path(
    "/share/home/tm902089733300000/a903202310/lys/data/"
    "RAGTruth/dataset/source_info.jsonl"
)
REPORT_DIRECTORY = "shortcut_route_v10"


def _print_report(report: dict) -> None:
    def ci(interval: list[float | None]) -> str:
        if interval[0] is None:
            return "n/a"
        return f"[{interval[0]:.6f},{interval[1]:.6f}]"

    scope = "ALL" if report["capture_complete"] else "PARTIAL"
    prevalence = (
        f"{report['prevalence']:.4%}" if report["prevalence"] is not None else "n/a"
    )
    print(f"\n=== {scope}-{report['task_type'].upper()} MECHANISM DETECTION ===")
    print(
        f"samples={report['samples']} sources={report['sources']} "
        f"tokens={report['tokens']} positives={report['hallucinated_tokens']} "
        f"evaluated_samples={report['evaluated_samples']} "
        f"evaluated_sources={report['evaluated_sources']} "
        f"evaluated_tokens={report['evaluated_tokens']} "
        f"evaluated_positives={report['evaluated_positives']} "
        f"prevalence={prevalence} "
        f"capture_complete={report['capture_complete']}"
    )
    for name in SCORE_ORDER:
        result = report["detection"][name]
        role = "PRIMARY" if name == report["primary_score"] else "control"
        if result["auroc"] is None:
            print(f"{role:9s} {name:30s} AUROC=n/a AP=n/a")
            continue
        print(
            f"{role:9s} {name:30s} "
            f"AUROC={result['auroc']:.6f} CI={ci(result['auroc_ci95'])} "
            f"AP={result['average_precision']:.6f} "
            f"CI={ci(result['average_precision_ci95'])} "
            f"lift={result['ap_lift']:.3f}"
        )
    shortcut = report.get("shortcut_route_detection", {})
    if shortcut:
        print("POST-HOC fixed shortcut-route candidates")
        for name in SHORTCUT_SCORE_NAMES:
            result = shortcut[name]
            if result["auroc"] is None:
                print(f"audit-AUC {name:38s} AUROC=n/a AP=n/a")
                continue
            print(
                f"audit-AUC {name:38s} "
                f"AUROC={result['auroc']:.6f} CI={ci(result['auroc_ci95'])} "
                f"AP={result['average_precision']:.6f} "
                f"CI={ci(result['average_precision_ci95'])}"
            )
    print("POST-HOC matched hallucinated - correct token differences")
    audit = report["group_difference_audit"]
    for name in PRINTED_AUDIT_ORDER:
        if name not in audit["metrics"]:
            continue
        result = audit["metrics"][name]
        difference = result["hallucinated_minus_correct"]
        if difference is None:
            print(f"audit     {name:30s} difference=n/a")
            continue
        print(
            f"audit     {name:30s} difference={difference:.6f} CI={ci(result['ci95'])}"
        )


def _all(args: argparse.Namespace) -> None:
    output = (
        args.output or Path(__file__).resolve().parent / "outputs" / args.model.name
    )
    inputs = capture_all(
        split_roots=(args.cache / "train", args.cache / "test"),
        source_info=args.source_info,
        model_path=args.model,
        output_root=output,
        device=args.device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype],
        limit=args.limit,
    )
    for task in TASK_TYPES:
        task_output = output / REPORT_DIRECTORY / task.casefold()
        report = evaluate_all(
            inputs=inputs[task],
            task_type=task,
            output=task_output / "report.json",
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
        _print_report(report)
        print(f"report: {task_output / 'report.json'}")
        print(f"token scores: {report['token_scores']}")
        print(f"population figures: {report['figures']}")


def _plot_sample(args: argparse.Namespace) -> None:
    result = plot_saved_sample(
        inputs=args.input,
        sample_id=args.sample_id,
        model_path=args.model,
        output=args.output,
    )
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Frozen-model mechanism detection")
    commands = root.add_subparsers(dest="command", required=True)

    all_data = commands.add_parser(
        "all", help="capture all data and evaluate QA, Summary, and Data2txt separately"
    )
    all_data.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    all_data.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    all_data.add_argument("--source-info", type=Path, default=DEFAULT_SOURCE_INFO)
    all_data.add_argument("--output", type=Path)
    all_data.add_argument("--device", default="cuda:0")
    all_data.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    all_data.add_argument("--limit", type=int)
    all_data.add_argument("--bootstrap", type=int, default=1000)
    all_data.add_argument("--seed", type=int, default=20260828)
    all_data.set_defaults(handler=_all)

    sample = commands.add_parser(
        "plot-sample", help="render one saved sample without replaying the model"
    )
    sample.add_argument("--input", action="append", type=Path, required=True)
    sample.add_argument("--sample-id", required=True)
    sample.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    sample.add_argument("--output", type=Path, required=True)
    sample.set_defaults(handler=_plot_sample)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
