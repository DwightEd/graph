"""Command line entry point for attention multiplex construction."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset

from .pipeline import build_dataset_representations
from .representation import MultiplexConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build attention-only dynamic multiplex token roles"
    )
    parser.add_argument("--attention-split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--block-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude-diagonal", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    dataset = open_research_dataset(
        arguments.attention_split,
        device=arguments.device,
        verify_hashes=True,
        retain_embedded_labels=False,
    )
    result = build_dataset_representations(
        dataset,
        arguments.output_dir,
        config=MultiplexConfig(
            rank=arguments.rank,
            block_rows=arguments.block_rows,
            random_seed=arguments.seed,
            include_diagonal=not arguments.exclude_diagonal,
        ),
        limit=arguments.limit,
        resume=arguments.resume,
        workers=arguments.workers,
        checkpoint_every=arguments.checkpoint_every,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
