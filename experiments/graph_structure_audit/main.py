"""CLI for graph-structure-first attention audits."""

from __future__ import annotations

import argparse

from .config import GraphAuditConfig
from .evaluate import evaluate_graph_audit
from .extract import extract_graph_audit


def _add_config(parser: argparse.ArgumentParser) -> None:
    defaults = GraphAuditConfig()
    parser.add_argument("--prompt-bins", type=int, default=defaults.prompt_bins)
    parser.add_argument("--coalition-top-sources", type=int, default=defaults.coalition_top_sources)
    parser.add_argument("--source-mask-fraction", type=float, default=defaults.source_mask_fraction)
    parser.add_argument("--channel-mask-fraction", type=float, default=defaults.channel_mask_fraction)
    parser.add_argument("--minimum-sources-for-recovery", type=int, default=defaults.minimum_sources_for_recovery)
    parser.add_argument("--minimum-channels-for-recovery", type=int, default=defaults.minimum_channels_for_recovery)
    parser.add_argument("--block-rows", type=int, default=defaults.block_rows)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--no-progress", action="store_true")


def _config(arguments) -> GraphAuditConfig:
    return GraphAuditConfig(
        prompt_bins=arguments.prompt_bins,
        coalition_top_sources=arguments.coalition_top_sources,
        source_mask_fraction=arguments.source_mask_fraction,
        channel_mask_fraction=arguments.channel_mask_fraction,
        minimum_sources_for_recovery=arguments.minimum_sources_for_recovery,
        minimum_channels_for_recovery=arguments.minimum_channels_for_recovery,
        block_rows=arguments.block_rows,
        random_seed=arguments.seed,
        show_progress=not arguments.no_progress,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="build and audit one graph per sample")
    extract.add_argument("--split-root", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--task-type")
    extract.add_argument("--limit", type=int)
    extract.add_argument("--no-raw-graphs", action="store_true")
    _add_config(extract)

    evaluate = commands.add_parser("evaluate", help="open labels after graph statistics and recovery scores are frozen")
    evaluate.add_argument("--split-root", required=True)
    evaluate.add_argument("--tokens", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=20260822)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "extract":
        extract_graph_audit(
            split_root=arguments.split_root,
            output_dir=arguments.output_dir,
            config=_config(arguments),
            task_type=arguments.task_type,
            limit=arguments.limit,
            save_raw_graphs=not arguments.no_raw_graphs,
        )
    else:
        evaluate_graph_audit(
            split_root=arguments.split_root,
            token_path=arguments.tokens,
            output_dir=arguments.output_dir,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )


if __name__ == "__main__":
    main()
