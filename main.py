"""Single CLI for canonical attention data and the unsupervised graph method."""

from __future__ import annotations

import argparse
import json

from attention_graph.graph import GraphBuildConfig
from attention_graph.statistics import collect_statistics, evaluate_statistics
from attention_graph.token_representation import (
    TokenRepresentationConfig,
    discover_token_representations,
    render_saved_sample,
)
from extract import AttentionExtractor, ExtractionConfig
from metadata import enrich_ragtruth_indices
from research_dataset import ResearchDataset, open_research_dataset


def _graph_args(parser):
    parser.add_argument("--selection", choices=("threshold", "global_topk", "typed_topk", "typed_mass_cover"), default="threshold")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mass-cover", type=float, default=0.80)
    parser.add_argument("--max-edges-per-target", type=int)
    parser.add_argument("--query-block", type=int, default=64)


def _graph_config(args):
    return GraphBuildConfig(
        selection=args.selection,
        threshold=args.threshold,
        top_k=args.top_k,
        mass_cover=args.mass_cover,
        max_edges_per_target=args.max_edges_per_target,
        query_block=args.query_block,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Attention-graph hallucination research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="RAGTruth + observer LLM -> canonical attention split")
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--generator-model", default="llama-2-7b-chat")
    p.add_argument("--task-type", default="all")
    p.add_argument("--floor", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("enrich-index")
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--dataset-path", required=True)

    p = sub.add_parser("statistics", help="all-sample label-blind scalar diagnostics")
    p.add_argument("--canonical-split", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    _graph_args(p)

    p = sub.add_parser("evaluate-statistics", help="feature-wise AUC after diagnostics are frozen")
    p.add_argument("--canonical-split", required=True)
    p.add_argument("--statistics", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser(
        "represent-tokens",
        help="full Lookback nodes plus multiscale causal evidence-flow detection",
    )
    p.add_argument("--train-split", required=True)
    p.add_argument("--test-split", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--position-bins", type=int, default=10)
    p.add_argument("--lookback-window", type=int, default=8)
    p.add_argument("--provenance-hops", type=int, default=2)
    p.add_argument("--prompt-bins", type=int, default=16)
    p.add_argument("--graph-head-components", type=int, default=8)
    p.add_argument("--bootstrap-replicates", type=int, default=200)
    p.add_argument("--csr-row-block", type=int, default=4096)
    p.add_argument(
        "--sample-id", action="append", default=[],
        help="render this test sample in full; repeat for multiple samples",
    )
    p.add_argument("--display-mass-cover", type=float, default=0.80)
    p.add_argument("--display-edges-per-type", type=int, default=2)
    p.add_argument("--display-max-edges", type=int, default=300)
    p.add_argument(
        "--display-layer", type=int,
        help="render this layer; default collapses each edge by maximum layer weight",
    )
    p.add_argument("--reference-size", type=int, default=12_000)
    p.add_argument("--subspace-components", type=int, default=32)
    p.add_argument("--tail-fraction", type=float, default=0.05)
    p.add_argument("--anomaly-quantile", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser(
        "render-token-graph",
        help="re-render weighted attention structure from an existing output directory",
    )
    p.add_argument("--test-split", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--display-layer", type=int)
    p.add_argument("--device", default="cpu")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "extract":
        result = AttentionExtractor(ExtractionConfig(
            args.model_path, args.dataset_path, args.output_dir, args.split,
            args.generator_model, args.task_type, args.floor, args.device, args.limit,
        )).run()
    elif args.command == "enrich-index":
        result = enrich_ragtruth_indices(args.canonical_root, args.dataset_path)
    elif args.command == "statistics":
        dataset = ResearchDataset(args.canonical_split, device=args.device, verify_hashes=True)
        result = collect_statistics(
            dataset, output_path=args.output, graph_config=_graph_config(args)
        )
    elif args.command == "evaluate-statistics":
        dataset = ResearchDataset(args.canonical_split, device="cpu", verify_hashes=True)
        result = evaluate_statistics(
            dataset, statistics_path=args.statistics, output_path=args.output
        )
    elif args.command == "represent-tokens":
        train_dataset = open_research_dataset(
            args.train_split,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=False,
        )
        test_dataset = open_research_dataset(
            args.test_split,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=False,
        )
        evaluation_dataset = open_research_dataset(
            args.test_split,
            device="cpu",
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        representation_config = TokenRepresentationConfig(
            position_bins=args.position_bins,
            lookback_window=args.lookback_window,
            provenance_hops=args.provenance_hops,
            prompt_bins=args.prompt_bins,
            graph_head_components=args.graph_head_components,
            bootstrap_replicates=args.bootstrap_replicates,
            csr_row_block=args.csr_row_block,
            sample_ids=tuple(args.sample_id),
            display_mass_cover=args.display_mass_cover,
            display_edges_per_type=args.display_edges_per_type,
            display_max_edges=args.display_max_edges,
            display_layer=args.display_layer,
            reference_size=args.reference_size,
            subspace_components=args.subspace_components,
            tail_fraction=args.tail_fraction,
            anomaly_quantile=args.anomaly_quantile,
            seed=args.seed,
        )
        result = discover_token_representations(
            train_dataset,
            test_dataset,
            evaluation_dataset,
            output_dir=args.output_dir,
            config=representation_config,
        )
    elif args.command == "render-token-graph":
        dataset = open_research_dataset(
            args.test_split,
            device=args.device,
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        result = render_saved_sample(
            dataset, output_dir=args.output_dir,
            sample_id=args.sample_id, layer=args.display_layer,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
