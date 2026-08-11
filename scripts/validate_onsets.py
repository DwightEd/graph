#!/usr/bin/env python3
"""Run paired onset-aligned validation on one canonical attention split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-split",
        type=Path,
        required=True,
        help="canonical attention split containing manifest, index, attention, and labels",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="directory for the seven result files"
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument(
        "--effect-width", type=int, default=3, help="tokens before/after onset (default: 3)"
    )
    parser.add_argument(
        "--bootstraps", type=int, default=10_000, help="bootstrap draws (default: 10000)"
    )
    parser.add_argument(
        "--permutations", type=int, default=10_000, help="sign-flip draws (default: 10000)"
    )
    parser.add_argument(
        "--rewires", type=int, default=100, help="constrained-rewire draws (default: 100)"
    )
    parser.add_argument(
        "--rewire-burn-in-sweeps",
        type=int,
        default=10,
        help="initial constrained source-swap sweeps (default: 10)",
    )
    parser.add_argument(
        "--rewire-thinning-sweeps",
        type=int,
        default=2,
        help="constrained source-swap sweeps per retained draw (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from onset_experiment import OnsetValidation, ValidationConfig

    config = ValidationConfig(
        canonical_split=args.canonical_split,
        output_dir=args.output_dir,
        device=args.device,
        effect_width=args.effect_width,
        bootstraps=args.bootstraps,
        permutations=args.permutations,
        rewires=args.rewires,
        rewire_burn_in_sweeps=args.rewire_burn_in_sweeps,
        rewire_thinning_sweeps=args.rewire_thinning_sweeps,
        seed=args.seed,
    )
    result = OnsetValidation(config).run()
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "pairs": result["pairs"],
                "events": result["events"],
                "primary_effects": [
                    {
                        key: row[key]
                        for key in (
                            "feature",
                            "mean_effect",
                            "ci_low",
                            "ci_high",
                            "holm_p",
                            "dz",
                        )
                    }
                    for row in result["primary_effects"]
                ],
                "topology_test": result["topology_test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
