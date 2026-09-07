"""Compatibility entry point for the delivered attention-rhythm audit commands.

Use the implementation already merged in attention_rhythm_run rather than
installing a second, incompatible observer/report pair. This entry accepts
--split both, --labels, --heads L:H ... and multiple --sample-id values.
All measurement definitions, FAI defaults (10..100), peak rules, outputs and
per-sample head grouping are those of the canonical runner; see
ATTENTION_RHYTHM_AUDIT.md for the differences from the earlier ZIP delivery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


TASKS = ("QA", "Summary", "Data2txt")


def translate_arguments(argv: list[str]) -> list[str]:
    """Translate CLI spellings only; do not change the measurement algorithm."""
    translated: list[str] = []
    supplied: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, attached = token.partition("=")
        supplied.add(option)
        if option == "--split" and (separator or index + 1 < len(argv)):
            value = attached if separator else argv[index + 1]
            translated.extend(("--split", "all" if value == "both" else value))
            index += 1 if separator else 2
            continue
        if option == "--labels" and not separator:
            translated.append("--evaluate")
            index += 1
            continue
        if option in {"--heads", "--sample-id", "--task"}:
            values = [attached] if separator else []
            index += 1
            while index < len(argv) and not argv[index].startswith("-"):
                values.append(argv[index])
                index += 1
            if not values:
                raise ValueError(f"{option} requires a value")
            if option == "--task":
                if len(values) == 1:
                    translated.extend(("--task", values[0]))
                elif set(values) == set(TASKS):
                    translated.extend(("--task", "all"))
                else:
                    raise ValueError("use one --task or all three task names")
            else:
                name = "--head" if option == "--heads" else "--sample-id"
                for value in values:
                    translated.extend((name, value))
            continue
        translated.append(token)
        index += 1
    # Preserve the delivered command's full-cohort defaults. The canonical
    # attention_rhythm_run entry keeps its original single-sample defaults.
    for option, value in (("--split", "all"), ("--task", "all"),
                          ("--samples-per-task", "0")):
        if option not in supplied:
            translated.extend((option, value))
    return translated


def normalize_scan_directory(args, command_parser) -> None:
    """Accept either the parent scan root or one completed split directory."""
    if args.scans is None:
        return
    manifest = Path(args.scans) / "run_manifest.json"
    if not manifest.is_file():
        return
    with manifest.open(encoding="utf-8") as stream:
        split = json.load(stream)["config"]["split"]
    if split not in {"train", "test"}:
        command_parser.error("scan manifest split must be train or test")
    if args.split not in {split, "all"}:
        command_parser.error("requested split disagrees with --scans directory")
    args.split = split
    args.scans = Path(args.scans).parent


def main(argv: list[str] | None = None):
    from .attention_rhythm_run import parser, run

    command_parser = parser()
    command_parser.prog = "python -m experiments.reanchor_flow.audit_attention_rhythm"
    command_parser.description = __doc__
    try:
        arguments = translate_arguments(list(sys.argv[1:] if argv is None else argv))
    except ValueError as error:
        command_parser.error(str(error))
    args = command_parser.parse_args(arguments)
    normalize_scan_directory(args, command_parser)
    return run(args)


if __name__ == "__main__":
    main()
