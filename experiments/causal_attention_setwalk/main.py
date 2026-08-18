"""CLI for causal anonymous SetWalk attention-hypergraph validation."""

from __future__ import annotations

import argparse
import json

from research_dataset import open_research_dataset
from .experiment import evaluate, fit_reference, score_dataset
from .model import ReferenceConfig
from .representation import SetWalkConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit unlabeled SetWalk references")
    fit.add_argument("--train-split", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--fourier-features", type=int, default=8)
    fit.add_argument("--dct-components", type=int, default=3)
    fit.add_argument("--recent-lag-max", type=int, default=4)
    fit.add_argument("--block-rows", type=int, default=8192)
    fit.add_argument("--seed", type=int, default=20260818)
    fit.add_argument("--reference-per-sample", type=int, default=8)
    fit.add_argument("--position-bins", type=int, default=8)
    fit.add_argument("--min-task-bin-rows", type=int, default=8)
    fit.add_argument("--trim-fraction", type=float, default=0.90)
    fit.add_argument("--limit", type=int)

    score = commands.add_parser("score", help="freeze all token representations")
    score.add_argument("--split-root", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda")
    score.add_argument("--limit", type=int)

    audit = commands.add_parser(
        "evaluate", help="open labels after representations are frozen"
    )
    audit.add_argument("--split-root", required=True)
    audit.add_argument("--scores", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--bootstrap-replicates", type=int, default=200)
    audit.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    if arguments.command == "fit":
        dataset = open_research_dataset(
            arguments.train_split, device=arguments.device, verify_hashes=True
        )
        result = fit_reference(
            dataset,
            arguments.output,
            representation_config=SetWalkConfig(
                fourier_features=arguments.fourier_features,
                dct_components=arguments.dct_components,
                recent_lag_max=arguments.recent_lag_max,
                block_rows=arguments.block_rows,
                seed=arguments.seed,
            ),
            reference_config=ReferenceConfig(
                reference_per_sample=arguments.reference_per_sample,
                position_bins=arguments.position_bins,
                min_task_bin_rows=arguments.min_task_bin_rows,
                trim_fraction=arguments.trim_fraction,
            ),
            limit=arguments.limit,
        )
    elif arguments.command == "score":
        dataset = open_research_dataset(
            arguments.split_root, device=arguments.device, verify_hashes=True
        )
        result = score_dataset(
            dataset,
            arguments.reference,
            arguments.output,
            limit=arguments.limit,
        )
    else:
        dataset = open_research_dataset(
            arguments.split_root,
            device=arguments.device,
            verify_hashes=True,
            retain_embedded_labels=True,
        )
        result = evaluate(
            dataset,
            arguments.scores,
            arguments.output_dir,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()

