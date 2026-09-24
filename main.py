"""Explicit entry points for mechanism measurement and detector baselines."""

import sys

HELP = """Usage: python main.py COMMAND [arguments]

  flow             Resampled claim pairs: head interactions, repair and persistence
  support          Per-token routing and entropy detection; score cached data, evaluate once
  dynamics         Offline head/source responses, unlabelled state fitting and token AUROC/AP
  transport        Signed source provenance through native attention, residual and FFN value paths
  transport-pack   Compact existing value-path captures for head and temporal audit; CPU only
  transport-state  Score/evaluate cached choice states and auto-pack; --stage pack reuses completed results
  transport-functions  Original-answer native FFN/RMS audit; no generated banks; coverage + auto-pack
  transport-observable  Native cumulative distribution response and conditional structured anomaly
  transport-readout  Supervised source-held-out probes of cached responses; no model forward
  transport-dual    Unlabelled current/persistent head readout; cache only, causal and offline
  transport-contrast  Frozen source/history four-condition likelihood; no truth-label training
  transport-carriers  Select history messages, measure conditional deletion, save per-token vectors
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
    if command == "transport-pack":
        from experiments.native_support.transport_pack import main as run
        run(arguments)
    elif command in ("support", "dynamics", "transport", "transport-state", "transport-functions", "transport-observable", "transport-readout", "transport-dual", "transport-contrast", "transport-carriers"):
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / "teaching/state_audit/src"))
        if command == "transport-carriers":
            from experiments.native_support.message_carriers.run import main as run
        elif command == "transport-contrast":
            from experiments.native_support.evidence_contrast.run import main as run
        elif command == "transport-dual":
            from experiments.native_support.dual_state.run import main as run
        elif command == "transport-readout":
            from experiments.native_support.readout.run import main as run
        elif command == "transport-observable":
            from experiments.native_support.observable_run import main as run
        elif command == "transport-functions":
            from experiments.native_support.functional_run import main as run
        elif command == "transport-state":
            from experiments.native_support.choice_state_run import main as run
        elif command == "transport":
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
