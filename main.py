"""Single CLI for canonical attention data and the unsupervised graph method."""

from __future__ import annotations

import argparse
import json

from archive import AttentionArchiveConverter, AttentionArchiveVerifier, ArchiveConfig
from attention_graph.evaluate import evaluate_scores
from attention_graph.graph import GraphBuildConfig
from attention_graph.mart import fit_mart, score_mart
from attention_graph.patterns import (
    PatternDiscoveryConfig,
    discover_provenance_patterns,
)
from attention_graph.score import load_checkpoint, save_score_records, score_dataset
from attention_graph.statistics import collect_statistics, evaluate_statistics
from attention_graph.train import TrainingConfig, train_unsupervised
from attention_graph.visualize import EmbeddingShiftVisualizer
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

    p = sub.add_parser("archive-attention", help="formal RAGTruth cache -> canonical train/test")
    p.add_argument("--formal-root", required=True)
    p.add_argument("--output-root", required=True)

    p = sub.add_parser("verify-attention")
    p.add_argument("--archive-root", required=True)

    p = sub.add_parser("enrich-index")
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--dataset-path", required=True)

    p = sub.add_parser("train", help="label-blind GNN training on canonical train split")
    p.add_argument("--train-split", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--message-steps", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--target-mask-rate", type=float, default=0.20)
    p.add_argument("--channel-drop-rate", type=float, default=0.10)
    p.add_argument("--target-block-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    _graph_args(p)

    p = sub.add_parser("score", help="frozen checkpoint -> label-free token embeddings and scores")
    p.add_argument("--canonical-split", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--target-block-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("fit-mart", help="train-only non-GNN mechanism baseline")
    p.add_argument("--train-split", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--neighbors", type=int, default=16)
    p.add_argument("--position-bins", type=int, default=8)
    p.add_argument("--reference-size", type=int, default=100000)

    p = sub.add_parser("score-mart", help="frozen MART detector -> label-free token scores")
    p.add_argument("--canonical-split", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("evaluate", help="open labels only after scores are frozen")
    p.add_argument("--canonical-split", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--output", required=True)

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
        "discover-patterns",
        help="training-free discovery of multi-layer prompt-provenance node patterns",
    )
    p.add_argument("--train-split", required=True)
    p.add_argument("--test-split", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--signature-view",
        choices=("prompt_absorption", "response_concentration"),
        default="prompt_absorption",
        help="fit one structural curve at a time; views are never concatenated",
    )
    p.add_argument("--checkpoints", type=int, default=8)
    p.add_argument("--min-patterns", type=int, default=2)
    p.add_argument("--max-patterns", type=int, default=6)
    p.add_argument("--fit-reference-size", type=int, default=30000)
    p.add_argument("--tsne-landmarks", type=int, default=10000)
    p.add_argument("--perplexity", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prototype-hops", type=int, default=3)
    p.add_argument("--prototype-max-incoming", type=int, default=6)
    _graph_args(p)

    p = sub.add_parser("visualize", help="paired before/after t-SNE of frozen GNN node states")
    p.add_argument("--canonical-split", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--domain-field", choices=("data_source", "task_type"), required=True)
    p.add_argument("--source-domain", required=True)
    p.add_argument("--target-domain", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-nodes-per-domain", type=int, default=5000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "extract":
        result = AttentionExtractor(ExtractionConfig(
            args.model_path, args.dataset_path, args.output_dir, args.split,
            args.generator_model, args.task_type, args.floor, args.device, args.limit,
        )).run()
    elif args.command == "archive-attention":
        result = AttentionArchiveConverter(ArchiveConfig(args.formal_root, args.output_root)).run()
    elif args.command == "verify-attention":
        result = AttentionArchiveVerifier(args.archive_root).run()
    elif args.command == "enrich-index":
        result = enrich_ragtruth_indices(args.canonical_root, args.dataset_path)
    elif args.command == "train":
        dataset = ResearchDataset(args.train_split, device=args.device, verify_hashes=True)
        config = TrainingConfig(
            embedding_dim=args.embedding_dim,
            message_steps=args.message_steps,
            dropout=args.dropout,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            target_mask_rate=args.target_mask_rate,
            channel_drop_rate=args.channel_drop_rate,
            target_block_size=args.target_block_size,
            seed=args.seed,
        )
        result = train_unsupervised(
            dataset, output_dir=args.output_dir, graph_config=_graph_config(args), config=config
        )
    elif args.command == "score":
        dataset = ResearchDataset(args.canonical_split, device=args.device, verify_hashes=True)
        model, calibrator, graph_config, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
        expected_channels = int(dataset.manifest["num_layers"]) * int(dataset.manifest["num_heads"])
        if model.num_channels != expected_channels:
            raise ValueError("checkpoint and canonical split have different attention geometry")
        records = score_dataset(
            dataset, model, calibrator, graph_config=graph_config,
            target_block_size=args.target_block_size, seed=args.seed,
        )
        path = save_score_records(records, args.output)
        result = {
            "output": path,
            "tokens": len(records),
            "samples": len(dataset),
            "labels_read": False,
            "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        }
    elif args.command == "fit-mart":
        dataset = ResearchDataset(args.train_split, device=args.device, verify_hashes=True)
        result = fit_mart(
            dataset, output_path=args.output, neighbors=args.neighbors,
            position_bins=args.position_bins, reference_size=args.reference_size,
        )
    elif args.command == "score-mart":
        dataset = ResearchDataset(args.canonical_split, device=args.device, verify_hashes=True)
        result = score_mart(dataset, checkpoint=args.checkpoint, output_path=args.output)
    elif args.command == "evaluate":
        dataset = ResearchDataset(args.canonical_split, device="cpu", verify_hashes=True)
        result = evaluate_scores(dataset, score_path=args.scores, output_path=args.output)
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
    elif args.command == "discover-patterns":
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
            retain_embedded_labels=True,
        )
        pattern_config = PatternDiscoveryConfig(
            signature_view=args.signature_view,
            checkpoints=args.checkpoints,
            min_patterns=args.min_patterns,
            max_patterns=args.max_patterns,
            fit_reference_size=args.fit_reference_size,
            tsne_landmarks=args.tsne_landmarks,
            perplexity=args.perplexity,
            seed=args.seed,
            prototype_hops=args.prototype_hops,
            prototype_max_incoming=args.prototype_max_incoming,
        )
        result = discover_provenance_patterns(
            train_dataset,
            test_dataset,
            test_dataset,
            output_dir=args.output_dir,
            graph_config=_graph_config(args),
            config=pattern_config,
        )
    elif args.command == "visualize":
        dataset = ResearchDataset(args.canonical_split, device=args.device, verify_hashes=True)
        result = EmbeddingShiftVisualizer(
            dataset,
            checkpoint=args.checkpoint,
            domain_field=args.domain_field,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            output_dir=args.output_dir,
            device=args.device,
            max_nodes_per_domain=args.max_nodes_per_domain,
            perplexity=args.perplexity,
            seed=args.seed,
        ).run()
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
