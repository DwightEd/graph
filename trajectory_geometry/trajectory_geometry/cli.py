"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import discover_attention_files, load_attention_sample
from .evaluation import GateAEvaluator
from .hidden import discover_hidden_files, pair_attention_hidden
from .pipeline import extract_many
from .routing import AnchorSpec
from .state_model import StateModelConfig
from .state_pipeline import run_graph_state_pipeline


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

    evaluate = commands.add_parser("evaluate", help="score extracted route features then evaluate labels")
    evaluate.add_argument("--train-features", required=True)
    evaluate.add_argument("--test-features", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda")

    state = commands.add_parser(
        "fit-state-model",
        help="fit and export graph-conditioned hidden-state residual vectors",
    )
    state.add_argument("--attention-root", required=True)
    state.add_argument("--hidden-root", required=True)
    state.add_argument("--output-dir", required=True)
    state.add_argument("--projection-dim", type=int, default=16)
    state.add_argument("--projection-reference-rows", type=int, default=12000)
    state.add_argument("--head-components", type=int, default=8)
    state.add_argument("--fit-tokens-per-layer", type=int, default=4096)
    state.add_argument("--fit-fraction", type=float, default=0.8)
    state.add_argument("--trim-fraction", type=float, default=0.9)
    state.add_argument("--ridge", type=float, default=1e-2)
    state.add_argument("--residual-shrinkage", type=float, default=0.1)
    state.add_argument("--minimum-relative-graph-gain", type=float, default=0.01)
    state.add_argument("--bootstrap-replicates", type=int, default=1000)
    state.add_argument("--dct-components", type=int, default=8)
    state.add_argument("--prompt-rewire-bins", type=int, default=8)
    state.add_argument("--csr-row-block", type=int, default=4096)
    state.add_argument("--seed", type=int, default=20260815)
    state.add_argument("--limit-train", type=int)
    state.add_argument("--limit-test", type=int)
    state.add_argument("--skip-train-embeddings", action="store_true")
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

    if arguments.command == "evaluate":
        report = GateAEvaluator(device=arguments.device).evaluate(
            arguments.train_features, arguments.test_features, arguments.output_dir
        )
        print((Path(arguments.output_dir) / "summary.txt").read_text(encoding="utf-8"), end="")
        return

    if arguments.command == "fit-state-model":
        train_attention = discover_attention_files(arguments.attention_root, "train")
        test_attention = discover_attention_files(arguments.attention_root, "test")
        train_hidden = discover_hidden_files(arguments.hidden_root, "train")
        test_hidden = discover_hidden_files(arguments.hidden_root, "test")
        train_pairs = pair_attention_hidden(train_attention, train_hidden)
        test_pairs = pair_attention_hidden(test_attention, test_hidden)
        if arguments.limit_train is not None:
            if arguments.limit_train < 1:
                raise ValueError("--limit-train must be positive")
            train_pairs = train_pairs[: arguments.limit_train]
        if arguments.limit_test is not None:
            if arguments.limit_test < 1:
                raise ValueError("--limit-test must be positive")
            test_pairs = test_pairs[: arguments.limit_test]
        config = StateModelConfig(
            projection_dim=arguments.projection_dim,
            projection_reference_rows=arguments.projection_reference_rows,
            head_components=arguments.head_components,
            fit_tokens_per_layer=arguments.fit_tokens_per_layer,
            fit_fraction=arguments.fit_fraction,
            trim_fraction=arguments.trim_fraction,
            ridge=arguments.ridge,
            residual_shrinkage=arguments.residual_shrinkage,
            minimum_relative_graph_gain=arguments.minimum_relative_graph_gain,
            bootstrap_replicates=arguments.bootstrap_replicates,
            dct_components=arguments.dct_components,
            prompt_rewire_bins=arguments.prompt_rewire_bins,
            csr_row_block=arguments.csr_row_block,
            seed=arguments.seed,
        )
        manifest = run_graph_state_pipeline(
            train_pairs,
            test_pairs,
            arguments.output_dir,
            config=config,
            save_train_embeddings=not arguments.skip_train_embeddings,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
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
