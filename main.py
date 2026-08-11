"""Small CLI for feature extraction, migration and graph construction."""

import argparse
import json

from archive import (
    ArchiveConfig,
    ArtifactInspector,
    AttentionArchiveConverter,
    AttentionArchiveVerifier,
    TraceArchiveConfig,
    TraceArchiveConverter,
)
from build import GRAPH_KINDS, BuildConfig, GraphDatasetBuilder
from extract import AttentionExtractor, ExtractionConfig
from metadata import enrich_ragtruth_indices


def _layers(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(item) for item in value.split(","))


def main(argv=None):
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="command", required=True)

    p = command.add_parser("extract")
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--generator-model", default="llama-2-7b-chat")
    p.add_argument("--task-type", default="all")
    p.add_argument("--floor", type=float, default=0.01)
    p.add_argument("--hidden-layers", default="", help="0-based decoder layers, e.g. 7,15,23,31")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)

    p = command.add_parser("archive-attention")
    p.add_argument("--formal-root", required=True)
    p.add_argument("--output-root", required=True)

    p = command.add_parser("archive-features")
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p = command.add_parser("enrich-index")
    p.add_argument("--canonical-root", required=True)
    p.add_argument("--dataset-path", required=True, help="RAGTruth dataset directory containing response.jsonl and source_info.jsonl")

    p = command.add_parser("inspect")
    p.add_argument("--artifact-dir", required=True)

    p = command.add_parser("verify-attention")
    p.add_argument("--archive-root", required=True)

    p = command.add_parser("build")
    p.add_argument("--cache-dir", required=True, help="one canonical split directory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--kind", choices=GRAPH_KINDS, default="relation_topk_channels")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--k-prompt", type=int, default=8)
    p.add_argument("--k-history", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)

    args = parser.parse_args(argv)
    if args.command == "extract":
        result = AttentionExtractor(ExtractionConfig(
            args.model_path,
            args.dataset_path,
            args.output_dir,
            args.split,
            args.generator_model,
            args.task_type,
            args.floor,
            _layers(args.hidden_layers),
            args.device,
            args.limit,
        )).run()
    elif args.command == "archive-attention":
        result = AttentionArchiveConverter(ArchiveConfig(args.formal_root, args.output_root)).run()
    elif args.command == "archive-features":
        result = TraceArchiveConverter(TraceArchiveConfig(args.trace_dir, args.output_dir)).run()
    elif args.command == "enrich-index":
        result = enrich_ragtruth_indices(args.canonical_root, args.dataset_path)
    elif args.command == "inspect":
        result = ArtifactInspector(args.artifact_dir).run()
    elif args.command == "verify-attention":
        result = AttentionArchiveVerifier(args.archive_root).run()
    else:
        result = GraphDatasetBuilder(BuildConfig(
            args.cache_dir,
            args.output_dir,
            args.kind,
            args.tau,
            args.k_prompt,
            args.k_history,
            args.device,
            args.limit,
        )).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
