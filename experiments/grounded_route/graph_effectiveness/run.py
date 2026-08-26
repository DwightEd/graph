"""Command-line interface for the saved-embedding effectiveness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import AuditConfig, audit
from .data import verify_bundle
from .detectors import DetectorConfig
from .label_free import label_free_audit
from .upper_bound import ProbeConfig


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen GroundedRoute node embeddings"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--index", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--topology-output", required=True)
    verify.add_argument("--seed", type=int, default=20260825)

    diagnose = commands.add_parser("audit")
    diagnose.add_argument("--calibration", required=True)
    diagnose.add_argument("--index", required=True)
    diagnose.add_argument("--test", required=True)
    diagnose.add_argument("--scores")
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument(
        "--control",
        nargs=3,
        action="append",
        default=[],
        metavar=("VARIANT", "CALIBRATION_INDEX", "TEST_INDEX"),
    )
    diagnose.add_argument("--device", default="cpu")
    diagnose.add_argument("--folds", type=int, default=5)
    diagnose.add_argument("--epochs", type=int, default=20)
    diagnose.add_argument("--patience", type=int, default=4)
    diagnose.add_argument("--batch-size", type=int, default=1024)
    diagnose.add_argument("--hidden-dim", type=int, default=128)
    diagnose.add_argument("--components", type=int, default=32)
    diagnose.add_argument("--neighbors", type=int, default=20)
    diagnose.add_argument("--max-reference", type=int, default=20_000)
    diagnose.add_argument("--bootstrap", type=int, default=2_000)
    diagnose.add_argument("--seed", type=int, default=20260825)
    diagnose.add_argument("--seeds", nargs="+", type=int, default=[20260825])
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    if arguments.command == "verify":
        bundle, report = verify_bundle(arguments.index)
        topology = label_free_audit(
            bundle,
            arguments.topology_output,
            seed=arguments.seed,
        )
        report = {**report, "topology": topology}
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {**report, "integrity": str(output.resolve())}
    else:
        calibration = {"real": arguments.calibration}
        test = {"real": arguments.index}
        for variant, calibration_index, test_index in arguments.control:
            if variant in calibration:
                raise ValueError(f"duplicate graph variant: {variant}")
            calibration[variant] = calibration_index
            test[variant] = test_index
        config = AuditConfig(
            detector=DetectorConfig(
                components=arguments.components,
                neighbors=arguments.neighbors,
                max_reference=arguments.max_reference,
                neural_hidden_dim=arguments.hidden_dim,
                neural_epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                seed=arguments.seed,
                neural_seeds=tuple(arguments.seeds),
            ),
            probe=ProbeConfig(
                folds=arguments.folds,
                hidden_dim=arguments.hidden_dim,
                epochs=arguments.epochs,
                patience=arguments.patience,
                batch_size=arguments.batch_size,
                split_seed=arguments.seed,
                seeds=tuple(arguments.seeds),
            ),
            bootstrap_replicates=arguments.bootstrap,
            bootstrap_seed=arguments.seed,
        )
        result = audit(
            calibration,
            test,
            arguments.test,
            arguments.output,
            published_score_path=arguments.scores,
            config=config,
            device=arguments.device,
        )
    print_report(arguments.command, result)


def print_report(command: str, report: dict[str, object]) -> None:
    print(f"{command} completed")
    for name in (
        "integrity",
        "report",
        "graphs",
        "nodes",
        "edges",
        "tokens",
        "positive_tokens",
        "prevalence",
    ):
        if name in report:
            print(f"{name}: {report[name]}")


if __name__ == "__main__":
    main()
