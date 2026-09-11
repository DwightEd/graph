"""Fixed candidate-conditioned path operators for unsupervised graph detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onset_analysis.analysis import OnsetChoiceAudit, OnsetChoiceConfig
from route_graph.capture import CaptureConfig, FrozenGraphCapture
from route_graph.data import RagtruthPreparer
from route_graph.detector import RouteDetector
from route_graph.evaluation import RouteEvaluator


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="prepare RAGTruth without annotation-based selection"
    )
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--task", choices=["QA", "Summary"], default="QA")
    prepare.add_argument("--generator", required=True)
    prepare.add_argument("--max-sources", type=int, default=32)
    prepare.add_argument("--seed", type=int, default=20260911)
    extract = commands.add_parser(
        "extract", help="capture real edges and candidate states from a local Llama"
    )
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--model", type=Path, required=True)
    extract.add_argument(
        "--end-layers",
        type=int,
        nargs="+",
        default=(),
        help="zero-based; default final layer",
    )
    extract.add_argument("--depths", type=int, nargs="+", default=(1, 2))
    extract.add_argument("--candidates", type=int, default=2)
    extract.add_argument("--device", default="cpu")
    extract.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    extract.add_argument("--max-tokens", type=int, default=2048)
    extract.add_argument("--max-attention-mb", type=int, default=1024)
    detect = commands.add_parser(
        "detect", help="fit source-disjoint references and freeze test scores"
    )
    detect.add_argument("--features", type=Path, required=True)
    detect.add_argument("--output", type=Path, required=True)
    detect.add_argument("--neighbors", type=int, default=3)
    detect.add_argument("--per-source", type=int, default=8)
    evaluate = commands.add_parser(
        "evaluate", help="join frozen full-stream scores with RAGTruth labels"
    )
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=20260911)
    onset = commands.add_parser(
        "onset-audit",
        help="joint onset-choice and lookback analysis of existing traces",
    )
    onset.add_argument("--audit", type=Path, required=True)
    onset.add_argument("--output", type=Path, required=True)
    onset.add_argument("--pre-window", type=int, default=3)
    onset.add_argument("--match-window", type=int, default=64)
    onset.add_argument("--bootstrap", type=int, default=200)
    onset.add_argument("--seed", type=int, default=20260910)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = RagtruthPreparer(
            args.dataset,
            args.output,
            args.task,
            args.generator,
            args.max_sources,
            args.seed,
        ).run()
    elif args.command == "extract":
        result = FrozenGraphCapture(
            CaptureConfig(
                input_path=args.input,
                output_dir=args.output,
                model_path=args.model,
                end_layers=tuple(args.end_layers),
                depths=tuple(args.depths),
                candidates=args.candidates,
                device=args.device,
                dtype=args.dtype,
                max_tokens=args.max_tokens,
                max_attention_mb=args.max_attention_mb,
            )
        ).run()
    elif args.command == "detect":
        result = RouteDetector(
            args.features, args.output, args.neighbors, args.per_source
        ).run()
    elif args.command == "evaluate":
        result = RouteEvaluator(
            args.scores, args.labels, args.output, args.bootstrap, args.seed
        ).run()
    else:
        result = OnsetChoiceAudit(
            OnsetChoiceConfig(
                args.audit,
                args.output,
                args.pre_window,
                args.match_window,
                args.bootstrap,
                args.seed,
            )
        ).run()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
