"""CLI for Causal Isomorphism Trajectory Geometry."""

from __future__ import annotations

import argparse
import json

from attention_graph.causal_events import MultiplexEventConfig
from research_dataset import open_research_dataset

from .experiment import evaluate_citg, fit_citg, score_citg
from .geometry import GeometryConfig
from .signatures import SignatureConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit",
        help="fit label-free trajectory geometry and calibration",
    )
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--limit", type=int)
    fit.add_argument("--block-rows", type=int, default=8192)
    fit.add_argument("--layer-bands", type=int, default=8)
    fit.add_argument("--max-rp-events-per-band", type=int, default=2)
    fit.add_argument("--max-rr-events-per-band", type=int, default=4)
    fit.add_argument("--hash-dim", type=int, default=128)
    fit.add_argument("--lag-bins", type=int, default=8)
    fit.add_argument("--weight-bins", type=int, default=5)
    fit.add_argument("--position-buckets", type=int, default=10)
    fit.add_argument("--late-band-transitions", type=int, default=2)
    fit.add_argument("--source-anchor-count", type=int, default=8)
    fit.add_argument("--max-parent-events", type=int, default=8)
    fit.add_argument("--pca-dim", type=int, default=32)
    fit.add_argument("--reference-per-sample", type=int, default=16)
    fit.add_argument("--min-condition-rows", type=int, default=32)
    fit.add_argument("--trim-fraction", type=float, default=1.0)
    fit.add_argument("--calibration-fraction", type=float, default=0.25)
    fit.add_argument("--bootstrap-replicates", type=int, default=1000)
    fit.add_argument(
        "--topology-gate-min-coverage",
        type=float,
        default=0.25,
    )
    fit.add_argument("--seed", type=int, default=20260817)

    score = commands.add_parser(
        "score",
        help="freeze held-out CITG token scores without labels",
    )
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--limit", type=int)

    evaluate = commands.add_parser(
        "evaluate",
        help="open labels only after the CITG artifact is frozen",
    )
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "fit":
        dataset = open_research_dataset(
            args.train_split,
            device=args.device,
            verify_hashes=False,
            retain_embedded_labels=False,
        )
        result = fit_citg(
            dataset,
            args.output_dir,
            event_config=MultiplexEventConfig(
                block_rows=args.block_rows,
                layer_bands=args.layer_bands,
                max_prompt_events_per_band=(
                    args.max_rp_events_per_band
                ),
                max_rr_events_per_band=args.max_rr_events_per_band,
            ),
            signature_config=SignatureConfig(
                hash_dim=args.hash_dim,
                lag_bins=args.lag_bins,
                weight_bins=args.weight_bins,
                position_buckets=args.position_buckets,
                late_band_transitions=args.late_band_transitions,
                source_anchor_count=args.source_anchor_count,
                max_parent_events=args.max_parent_events,
            ),
            geometry_config=GeometryConfig(
                pca_dim=args.pca_dim,
                reference_per_sample=args.reference_per_sample,
                min_condition_rows=args.min_condition_rows,
                trim_fraction=args.trim_fraction,
                calibration_fraction=args.calibration_fraction,
                bootstrap_replicates=args.bootstrap_replicates,
                topology_gate_min_coverage=(
                    args.topology_gate_min_coverage
                ),
                seed=args.seed,
            ),
            limit=args.limit,
        )
    elif args.command == "score":
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=False,
            retain_embedded_labels=False,
        )
        result = score_citg(
            dataset,
            args.reference,
            args.output,
            limit=args.limit,
        )
    else:
        dataset = open_research_dataset(
            args.split_root,
            device=args.device,
            verify_hashes=False,
            retain_embedded_labels=True,
        )
        report = evaluate_citg(
            dataset,
            args.scores,
            args.output,
        )
        result = {
            "output": args.output,
            "labels_read": True,
            **(report["metrics"] or {}),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
