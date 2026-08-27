"""CLI facade for the attention-routing lineage validation stages.

The implementation is split by research step: trace artifacts, label-free
calibration, and post-hoc evaluation.  Public imports remain here so a single
module command is sufficient for remote experiments.
"""

from __future__ import annotations

import argparse

from .lineage_artifacts import (
    ARTIFACT_VERSION,
    DEFAULT_EPSILON,
    TRACE_SCHEMA,
    carrier_changed_fraction,
    complete_lineage_rows,
    complete_trace_rows,
    direct_prompt_lookback,
    encoded_to_token_graph,
    export_trace,
    raw_takeover,
    require_artifact,
    sample_seed,
    trace_graph,
)
from .lineage_evaluation import (
    evaluate_scores,
    load_frozen_labels,
    onset_diagnostics,
    paired_source_delta,
    relay_rescue_diagnostics,
)
from .lineage_scoring import (
    DEFAULT_MINIMUM_REFERENCE_SOURCES,
    DEFAULT_POSITION_BIN_WIDTH,
    SCORE_SCHEMA,
    conditional_high_tail,
    score_traces,
    source_balanced_high_tail,
)

__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_EPSILON",
    "DEFAULT_MINIMUM_REFERENCE_SOURCES",
    "DEFAULT_POSITION_BIN_WIDTH",
    "SCORE_SCHEMA",
    "TRACE_SCHEMA",
    "carrier_changed_fraction",
    "complete_lineage_rows",
    "complete_trace_rows",
    "conditional_high_tail",
    "direct_prompt_lookback",
    "encoded_to_token_graph",
    "evaluate_scores",
    "export_trace",
    "load_frozen_labels",
    "onset_diagnostics",
    "paired_source_delta",
    "raw_takeover",
    "relay_rescue_diagnostics",
    "require_artifact",
    "sample_seed",
    "score_traces",
    "source_balanced_high_tail",
    "trace_graph",
]


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predecessor-aligned attention-routing lineage validator"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("trace")
    command.add_argument("--index", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--seed", type=int, default=20260827)
    command.add_argument("--carrier-rewire-passes", type=int, default=4)
    command.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)

    command = commands.add_parser("score")
    command.add_argument("--calibration-trace", required=True)
    command.add_argument("--test-trace", required=True)
    command.add_argument("--output", required=True)
    command.add_argument(
        "--position-bin-width", type=int, default=DEFAULT_POSITION_BIN_WIDTH
    )
    command.add_argument(
        "--minimum-reference-sources",
        type=int,
        default=DEFAULT_MINIMUM_REFERENCE_SOURCES,
    )

    command = commands.add_parser("evaluate")
    command.add_argument("--test-root", required=True)
    command.add_argument("--scores", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--bootstrap", type=int, default=500)
    command.add_argument("--seed", type=int, default=20260827)
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    if arguments.command == "trace":
        report = export_trace(
            arguments.index,
            arguments.output,
            seed=arguments.seed,
            carrier_rewire_passes=arguments.carrier_rewire_passes,
            epsilon=arguments.epsilon,
        )
    elif arguments.command == "score":
        report = score_traces(
            arguments.calibration_trace,
            arguments.test_trace,
            arguments.output,
            position_bin_width=arguments.position_bin_width,
            minimum_reference_sources=arguments.minimum_reference_sources,
        )
    else:
        report = evaluate_scores(
            arguments.test_root,
            arguments.scores,
            arguments.output,
            bootstrap_replicates=arguments.bootstrap,
            seed=arguments.seed,
        )
    print(f"{arguments.command} completed")
    for name, value in report.items():
        if name in {
            "trace",
            "scores",
            "evaluation",
            "samples",
            "nodes",
            "available_nodes",
            "representation_dimension",
            "routing_representation_dimension",
            "carrier_rewire_changed_fraction",
            "carrier_rewire_nonzero_sample_fraction",
            "lineage_mass_max_error",
            "routing_role_mass_max_error",
        }:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
