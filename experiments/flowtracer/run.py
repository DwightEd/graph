import argparse
import numpy as np

from .attention import aggregate_attention
from .flow import compute_edge_flow, compute_node_throughput, extract_weighted_paths
from .graph import build_token_dag
from .io import save_sample_graph


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract target-related weighted paths from attention")
    parser.add_argument("--attention", required=True, help=".npy or .npz attention tensor")
    parser.add_argument("--targets", required=True, help="comma-separated target token indices")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args(argv)


def process_sample(attention, targets, token_ids=None, threshold=0.0):
    matrix = aggregate_attention(attention)
    graph = build_token_dag(matrix, token_ids=token_ids, threshold=threshold)
    analysis = compute_edge_flow(graph, targets)
    analysis["throughput"] = compute_node_throughput(graph, analysis["edges"])
    analysis["paths"] = extract_weighted_paths(graph, targets)
    return graph, analysis


def main(argv=None):
    args = parse_args(argv)
    loaded = np.load(args.attention)
    attention = loaded["attention"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
    targets = [int(value) for value in args.targets.split(",") if value.strip()]
    graph, analysis = process_sample(attention, targets, threshold=args.threshold)
    save_sample_graph(args.output, graph, analysis)
    print(f"saved {len(graph['nodes'])} nodes and {len(analysis['edges'])} target-related edges to {args.output}")


if __name__ == "__main__":
    main()
