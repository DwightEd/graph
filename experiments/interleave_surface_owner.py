"""Reuse the audited exclusive pause/restore transaction for surface validation."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from experiments.interleave_soft_graph import (
    GRAPH,
    POPULATION,
    PYTHON,
    REANCHOR,
    checksum,
    execute_and_restore,
    frozen_population,
    pidfd_open,
    pidfd_signal,
    save,
)


def run(args):
    args.output = args.output.resolve()
    args.journal = args.journal.resolve()
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    audit_log, resume_log = args.journal.with_suffix(".audit.log"), args.journal.with_suffix(".population.log")
    if any(p.exists() for p in (args.journal, audit_log, resume_log)):
        raise FileExistsError("preserve prior journals; choose a new path")
    settings_path = args.output / "settings.json"
    prepared = json.loads(settings_path.read_text())
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    with (REANCHOR / "runs/relation_interleave_20260913.lock").open("a") as scheduler:
        fcntl.flock(scheduler, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.output / ".phase.lock").open("a") as phase:
            fcntl.flock(phase, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [str(PYTHON), "-u", "-m", "next_iteration.surface_runner",
                "--preflight", prepared["preflight_path"], "--output", str(args.output),
                "--observer-model", prepared["observer_model"], "--reader-model", prepared["reader_model"],
                "--inherited-lock-fd", str(phase.fileno())]
            subprocess.run(command + ["--stage", "prepare"], cwd=GRAPH, env=env,
                           pass_fds=(phase.fileno(),), check=True)
            if prepared["protocol"]["phase_order"] != ["features", "finite"]:
                raise ValueError("surface phase order differs")
            probe = pidfd_open(os.getpid())
            try:
                pidfd_signal(probe, 0)
            finally:
                os.close(probe)
            frozen = frozen_population()
            before = json.loads((POPULATION / "progress.json").read_text())
            if before["pid"] != args.population_pid or before["status"] != "running":
                raise ValueError("population PID and running progress disagree")
            descriptor = pidfd_open(args.population_pid)
            try:
                proc = Path(f"/proc/{args.population_pid}")
                cmdline = (proc / "cmdline").read_bytes().split(b"\0")
                if (b"decoding.ragtruth_population" not in cmdline
                        or b"outputs/ragtruth_population_20260912" not in cmdline
                        or (proc / "cwd").resolve() != REANCHOR):
                    raise ValueError("PID is not expected population process")
                record = {"status": "prepared", "previous_progress": before, "started_unix": time.time(),
                    "audit_output": str(args.output), "audit_settings_sha256": checksum(settings_path),
                    "frozen_population": frozen, "command": command + ["--stage", "all"],
                    "scheduler_sha256": checksum(Path(__file__)), "signal_target": "Linux_pidfd",
                    "scope": "surface_candidate_supply_validation_native0_no_label_evaluation"}
                save(args.journal, record)
                execute_and_restore(args, audit_log, resume_log, command, env, phase, descriptor, frozen, record)
            finally:
                os.close(descriptor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    run(parser.parse_args())
