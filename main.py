"""Explicit entry points for mechanism measurement and detector baselines."""

import sys

HELP = """Usage: python main.py COMMAND [arguments]

  flow             Resampled claim pairs: head interactions, repair and persistence
  support          Per-token routing and entropy detection; score cached data, evaluate once
  dynamics         Offline head/source responses, unlabelled state fitting and token AUROC/AP
  transport        Automatic source blocks and vector-conditioned offline budget states
  flow-edges       Earlier final-target edge experiment with branch-correct restoration
  population       RAGTruth screen/confirm; --phase grounding is a forecast baseline
  regime           Existing unlabeled head-covariance HMM; direction remains a hypothesis
  unsupervised     Existing entropy/local-reuse baseline; no semantic binding readout
  supervised-s10   Historical supervised baseline
  supervised-s11   Historical supervised baseline

Read experiments/native_support/README.md for detector inputs and execution costs.
No command starts an experiment automatically. Each command accepts --help.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(HELP)
        return
    command, arguments = argv[0], argv[1:]
    if command in ("support", "dynamics", "transport"):
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "teaching/state_audit/src"))
        if command == "transport":
            from experiments.native_support.transport import main as run
        elif command == "dynamics":
            from experiments.native_support.dynamics import main as run
        else:
            from experiments.native_support.run import main as run
        run(arguments)
    elif command == "flow":
        from experiments.path_conflict.paired import cli
        cli(arguments)
    elif command == "flow-edges":
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
