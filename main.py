"""Command-line entry point for attention extraction and graph building."""

import argparse
import json

from archive import (
    ArchiveConfig,
    ArtifactInspector,
    AttentionArchiveConverter,
    AttentionArchiveVerifier,
)
from build import GRAPH_KINDS, BuildConfig, GraphDatasetBuilder
from extract import AttentionExtractor, ExtractionConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create label-free attention caches and graphs."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    extract_parser = subcommands.add_parser("extract")
    extract_parser.add_argument("--model-path", required=True)
    extract_parser.add_argument("--dataset-path", required=True)
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--split", required=True, choices=("train", "test"))
    extract_parser.add_argument("--generator-model", default="llama-2-7b-chat")
    extract_parser.add_argument("--task-type", default="all")
    extract_parser.add_argument("--floor", type=float, default=0.01)
    extract_parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--limit", type=int)

    build_parser = subcommands.add_parser("build")
    build_parser.add_argument("--cache-dir", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--kind", choices=GRAPH_KINDS, default="relation_topk_channels")
    build_parser.add_argument("--tau", type=float, default=0.05)
    build_parser.add_argument("--k-prompt", type=int, default=8)
    build_parser.add_argument("--k-history", type=int, default=8)
    build_parser.add_argument("--device", default="cuda")
    build_parser.add_argument("--limit", type=int)
    build_parser.add_argument("--split", choices=("train", "test"))

    inspect_parser = subcommands.add_parser("inspect")
    inspect_parser.add_argument("--artifact-dir", required=True)

    archive_parser = subcommands.add_parser("archive-attention")
    archive_parser.add_argument("--formal-root", required=True)
    archive_parser.add_argument("--output-root", required=True)

    verify_parser = subcommands.add_parser("verify-attention")
    verify_parser.add_argument("--archive-root", required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "extract":
        config = ExtractionConfig(
            model_path=arguments.model_path,
            dataset_path=arguments.dataset_path,
            output_dir=arguments.output_dir,
            split=arguments.split,
            generator_model=arguments.generator_model,
            task_type=arguments.task_type,
            floor=arguments.floor,
            dtype=arguments.dtype,
            device=arguments.device,
            limit=arguments.limit,
        )
        AttentionExtractor(config).run()
        print(f"extract: {arguments.output_dir}")
        return

    if arguments.command == "inspect":
        print(json.dumps(ArtifactInspector(arguments.artifact_dir).run(), indent=2, sort_keys=True))
        return

    if arguments.command == "archive-attention":
        summary = AttentionArchiveConverter(ArchiveConfig(
            formal_root=arguments.formal_root,
            output_root=arguments.output_root,
        )).run()
        print(
            f"archive-attention: {summary['count']} samples, "
            f"{summary['source_bytes']} B -> {summary['payload_bytes']} B "
            f"(size_ratio={summary['size_ratio']:.3f}, "
            f"saved_percent={(1 - summary['size_ratio']) * 100:.1f}%, "
            f"manifest_bytes={summary['manifest_bytes']})"
        )
        return

    if arguments.command == "verify-attention":
        summary = AttentionArchiveVerifier(arguments.archive_root).run()
        print(
            f"verify-attention: {summary['count']} samples "
            f"(train={summary['splits']['train']}, test={summary['splits']['test']})"
        )
        return

    config = BuildConfig(
        cache_dir=arguments.cache_dir,
        output_dir=arguments.output_dir,
        kind=arguments.kind,
        tau=arguments.tau,
        k_prompt=arguments.k_prompt,
        k_history=arguments.k_history,
        device=arguments.device,
        limit=arguments.limit,
        split=arguments.split,
    )
    summary = GraphDatasetBuilder(config).run()
    print(f"build {summary['kind']}: {summary['count']} graphs")


if __name__ == "__main__":
    main()
