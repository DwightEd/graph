"""CLI for the label-free RR topology-dynamics mechanism audit."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .evaluation import evaluate_topology_artifact
from .experiment import (
    TopologyAuditConfig,
    fit_topology_reference,
    score_topology_dataset,
)
from .extractor import TopologyDynamicsConfig


def _topology_arguments(parser):
    parser.add_argument("--spectral-top-k", type=int, default=5)
    parser.add_argument("--block-rows", type=int, default=8192)
    parser.add_argument("--position-bins", type=int, default=8)
    parser.add_argument("--recent-lag-max", type=int, default=4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit", help="fit label-free task/position feature references on train"
    )
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--spectral-reference", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--reference-per-sample", type=int, default=16)
    fit.add_argument("--min-task-bin-rows", type=int, default=8)
    fit.add_argument("--phase-bins", type=int, default=10)
    fit.add_argument("--onset-window", type=int, default=4)
    fit.add_argument("--bootstrap-replicates", type=int, default=1000)
    fit.add_argument("--seed", type=int, default=20260815)
    fit.add_argument("--limit", type=int)
    _topology_arguments(fit)

    score = commands.add_parser(
        "score", help="freeze full token topology features without labels"
    )
    score.add_argument("--split-root", required=True)
    score.add_argument("--spectral-reference", required=True)
    score.add_argument("--topology-reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser(
        "evaluate", help="open labels after features are frozen and audit differences"
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--bootstrap-replicates", type=int)
    evaluate.add_argument("--onset-window", type=int)
    evaluate.add_argument("--phase-bins", type=int)
    evaluate.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def _topology_config(args):
    return TopologyDynamicsConfig(
        spectral_top_k=args.spectral_top_k,
        block_rows=args.block_rows,
        position_bins=args.position_bins,
        recent_lag_max=args.recent_lag_max,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(
            args.train_split, device=args.device, verify_hashes=True
        )
        result = fit_topology_reference(
            dataset,
            args.spectral_reference,
            args.output,
            topology_config=_topology_config(args),
            audit_config=TopologyAuditConfig(
                reference_per_sample=args.reference_per_sample,
                min_task_bin_rows=args.min_task_bin_rows,
                phase_bins=args.phase_bins,
                onset_window=args.onset_window,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            ),
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(
            args.split_root, device=args.device, verify_hashes=True
        )
        result = score_topology_dataset(
            dataset,
            args.spectral_reference,
            args.topology_reference,
            args.output,
            limit=args.limit,
        )
    else:
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        report = evaluate_topology_artifact(
            dataset,
            args.features,
            args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            onset_window=args.onset_window,
            phase_bins=args.phase_bins,
            seed=args.seed,
        )
        result = {
            "output": f"{args.output_dir}/report.json",
            "labels_read": True,
            "primary_feature_metrics": report["feature_metrics_raw"],
            **report["overall"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
