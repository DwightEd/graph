"""Unified command line for lookback transport, source allocation and evaluation."""

import argparse
import sys
from pathlib import Path

from .evaluation import OfflineEvaluator


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("command", choices=("lookback", "allocate", "evaluate"))
    command.add_argument("arguments", nargs=argparse.REMAINDER)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "lookback":
        from . import event_run

        return event_run.run(event_run.parser().parse_args(args.arguments))
    if args.command == "allocate":
        from . import run

        return run.run(run.parser().parse_args(args.arguments))

    evaluate_parser = argparse.ArgumentParser(description="evaluate a committed study offline")
    evaluate_parser.add_argument("output", type=Path)
    evaluate_parser.add_argument("--bootstrap", type=int, default=200)
    request = evaluate_parser.parse_args(args.arguments)
    return OfflineEvaluator(request.output, bootstrap=request.bootstrap).run()


if __name__ == "__main__":
    main(sys.argv[1:])
