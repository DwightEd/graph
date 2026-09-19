"""Explicit entry points for mechanism measurement and detector baselines."""

import sys


HELP = """Usage: python main.py COMMAND [arguments]

  flow             Final-target native transport, head interactions and repair
  population       RAGTruth screen/confirm; --phase grounding is a forecast baseline
  regime           Existing unlabeled head-covariance HMM; direction remains a hypothesis
  unsupervised     Existing entropy/local-reuse baseline; no semantic binding readout
  supervised-s10   Historical supervised baseline
  supervised-s11   Historical supervised baseline

Read docs/TRANSPORT_METHOD_20260919.md for the method, literature and remaining gaps.
No command starts an experiment automatically. Each command accepts --help.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(HELP)
        return
    command, arguments = argv[0], argv[1:]
    if command == "flow":
        from experiments.path_conflict.transport import cli
        cli(arguments)
    elif command == "population":
        from experiments.ragtruth_flow.run import main as run
        run(arguments)
    elif command == "supervised-s11":
        from structural_detector.transport_run import main as run
        run(arguments)
    elif command in ("unsupervised", "supervised-s10", "regime"):
        import runpy
        modules = {"unsupervised": "reuse_detector.run",
                   "supervised-s10": "structural_detector.experiment",
                   "regime": "experiments.unsupervised_token_graph.latent_regime"}
        sys.argv = [sys.argv[0], *arguments]
        runpy.run_module(modules[command], run_name="__main__")
    else:
        raise SystemExit(f"Unknown command: {command}. Use --help; S11 requires supervised-s11.")


if __name__ == "__main__":
    main()
