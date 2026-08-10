import argparse
from pathlib import Path

from tqdm import tqdm

from data import load_feature, save_graph
from extract import extract_ragtruth
from graphs import build_graph


KINDS = ("original", "multiplex", "support", "relation_topk", "hypergraph")


def build_directory(args):
    paths = sorted(Path(args.features).glob("*.pt"))
    if args.limit:
        paths = paths[:args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for path in tqdm(paths, desc=f"build {args.kind}"):
        sample = load_feature(path)
        common = {"node_feature": args.node_feature, "hidden_layer": args.hidden_layer}
        if args.kind in ("original", "hypergraph"):
            common["tau"] = 0.05 if args.tau is None else args.tau
        elif args.kind == "multiplex":
            common["tau"] = args.tau
        elif args.kind == "support":
            common["mass"] = args.mass
        else:
            common["k_prompt"] = args.k_prompt
            common["k_history"] = args.k_history
        graph = build_graph(sample, args.kind, **common)
        save_graph(graph, out / f"{sample['sample_id']}.pt")


def main():
    parser = argparse.ArgumentParser(description="Minimal attention graph research tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--generator-model", default="llama-2-7b-chat")
    p.add_argument("--task", default="all")
    p.add_argument("--floor", type=float, default=0.01)
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--hidden-layers", default="")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("build")
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--tau", type=float)
    p.add_argument("--mass", type=float, default=0.8)
    p.add_argument("--k-prompt", type=int, default=8)
    p.add_argument("--k-history", type=int, default=8)
    p.add_argument("--node-feature", choices=("diagonal", "hidden", "none"), default="diagonal")
    p.add_argument("--hidden-layer", type=int, default=-1)
    p.add_argument("--limit", type=int)

    args = parser.parse_args()
    if args.command == "extract":
        hidden = tuple(int(x) for x in args.hidden_layers.split(",") if x.strip())
        extract_ragtruth(
            args.model, args.dataset, args.output, args.split,
            generator_model=args.generator_model,
            task=args.task,
            floor=args.floor,
            dtype=args.dtype,
            device=args.device,
            hidden_layers=hidden,
            limit=args.limit,
        )
    else:
        build_directory(args)


if __name__ == "__main__":
    main()
