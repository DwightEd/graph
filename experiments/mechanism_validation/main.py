"""Command line for label-free attention-mechanism validation experiments."""

from __future__ import annotations

import argparse

from experiments.mechanism_validation.experiment import (
    build_graphs,
    evaluate_graphs,
    evaluate_mechanisms,
)
from experiments.mechanism_validation.screen import MechanismScreen
from research_dataset import open_research_dataset

DEFAULT_VARIANTS = [
    "exact", "no_edges", "unit_mass", "uniform_on_support", "weight_shuffle",
    "source_rewire", "rp_only", "rr_only", "source_free",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    screen = commands.add_parser("screen")
    screen.add_argument("--split-root", required=True)
    screen.add_argument("--output-dir", required=True)
    screen.add_argument("--device", default="cuda")
    screen.add_argument("--ema-decay", type=float, default=.9)

    mechanisms = commands.add_parser("evaluate-mechanisms")
    mechanisms.add_argument("--train-split", required=True)
    mechanisms.add_argument("--train-features", required=True)
    mechanisms.add_argument("--test-split", required=True)
    mechanisms.add_argument("--test-features", required=True)
    mechanisms.add_argument("--output-dir", required=True)
    mechanisms.add_argument("--bootstrap", type=int, default=200)
    mechanisms.add_argument("--seed", type=int, default=0)
    mechanisms.add_argument("--max-train-tokens", type=int, default=100000)

    graph = commands.add_parser("build-graph")
    graph.add_argument("--split-root", required=True)
    graph.add_argument("--mechanism-features", required=True)
    graph.add_argument("--output-dir", required=True)
    graph.add_argument("--device", default="cuda")
    graph.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=DEFAULT_VARIANTS)
    graph.add_argument("--seed", type=int, default=0)

    graphs = commands.add_parser("evaluate-graphs")
    graphs.add_argument("--train-split", required=True)
    graphs.add_argument("--train-graphs", required=True)
    graphs.add_argument("--test-split", required=True)
    graphs.add_argument("--test-graphs", required=True)
    graphs.add_argument("--output-dir", required=True)
    graphs.add_argument("--seed", type=int, default=0)
    graphs.add_argument("--bootstrap", type=int, default=200)
    graphs.add_argument("--max-train-tokens", type=int, default=100000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "screen":
        dataset = open_research_dataset(args.split_root, device=args.device)
        result = MechanismScreen(dataset, args.output_dir, ema_decay=args.ema_decay).run()
    elif args.command == "evaluate-mechanisms":
        result = evaluate_mechanisms(args.train_split, args.train_features, args.test_split, args.test_features,
                                     args.output_dir, bootstrap=args.bootstrap, seed=args.seed,
                                     max_train_tokens=args.max_train_tokens)
    elif args.command == "build-graph":
        result = build_graphs(args.split_root, args.mechanism_features, args.output_dir, device=args.device,
                              variants=args.variants, seed=args.seed)
    else:
        result = evaluate_graphs(args.train_split, args.train_graphs, args.test_split, args.test_graphs,
                                 args.output_dir, seed=args.seed, max_train_tokens=args.max_train_tokens,
                                 bootstrap=args.bootstrap)
    if args.command == "evaluate-mechanisms":
        top = sorted(result["adjusted_global_mean"].items(), key=lambda item: item[1]["point_delta"]["auroc"], reverse=True)[:5]
        print("adjusted AUROC delta top5:", [(name, round(value["point_delta"]["auroc"], 4)) for name, value in top])
        print(f"results: {args.output_dir}/results.json")
    elif args.command == "evaluate-graphs":
        print("paired cluster intervals:", result["paired_cluster_intervals"])
        print(f"results: {args.output_dir}/results.json")
    else:
        print(f"{args.command}: results written to {args.output_dir}")


if __name__ == "__main__":
    main()
