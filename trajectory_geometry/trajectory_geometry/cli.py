"""Command line interface."""

from __future__ import annotations

import argparse
import json

from .data import discover_attention_files, load_attention_sample
from .pipeline import extract_many
from .routing import AnchorSpec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trajectory-geometry")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="validate and summarize one cache")
    inspect.add_argument("--attention", required=True)

    extract = commands.add_parser("extract", help="extract route-dynamics vectors")
    extract.add_argument("--attention-root", required=True)
    extract.add_argument("--split", choices=("train", "test"))
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--prompt-bins", type=int, default=8)
    extract.add_argument("--history-lag-edges", default="1,2,4,8,16,32")
    extract.add_argument("--embedding-dim", type=int, default=256)
    extract.add_argument("--seed", type=int, default=20260814)
    extract.add_argument("--csr-row-block", type=int, default=4096)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--save-raw-route", action="store_true")
    return parser


def _spec(arguments: argparse.Namespace) -> AnchorSpec:
    edges = tuple(int(value) for value in arguments.history_lag_edges.split(","))
    spec = AnchorSpec(prompt_bins=arguments.prompt_bins, history_lag_edges=edges)
    spec.validate()
    return spec


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "inspect":
        sample = load_attention_sample(arguments.attention)
        print(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "path": str(sample.path),
                    "layers": sample.layers,
                    "heads": sample.heads,
                    "tokens": sample.token_count,
                    "response_idx": sample.response_idx,
                    "response_tokens": sample.response_tokens,
                    "retained_values": int(sample.values.size),
                    "attention_floor": sample.attention_floor,
                    "labels_read": False,
                },
                indent=2,
            )
        )
        return

    files = discover_attention_files(arguments.attention_root, arguments.split)
    if arguments.limit is not None:
        if arguments.limit < 1:
            raise ValueError("--limit must be positive")
        files = files[: arguments.limit]
    manifest = extract_many(
        files,
        arguments.output_dir,
        spec=_spec(arguments),
        embedding_dim=arguments.embedding_dim,
        seed=arguments.seed,
        csr_row_block=arguments.csr_row_block,
        save_raw_route=arguments.save_raw_route,
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
