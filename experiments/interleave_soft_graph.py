"""Exclusively run an already prepared native audit, then restore population."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import select
import signal
import subprocess
import time
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[1]
REANCHOR = GRAPH.parent / "reanchor"
POPULATION = REANCHOR / "outputs/ragtruth_population_20260912"
PYTHON = GRAPH.parent.parent / "conda_envs/research/bin/python"


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def frozen_population():
    settings = json.loads((POPULATION / "settings.json").read_text())
    for name, expected in settings["code_sha256"].items():
        if checksum(Path(name)) != expected:
            raise ValueError(f"population frozen code differs: {name}")
    manifest = json.loads((POPULATION / "input_manifest.json").read_text())
    if checksum(POPULATION / "inputs.jsonl") != manifest["sha256"]:
        raise ValueError("population generated input differs")
    for name, expected in settings["input_sha256"].items():
        if checksum(Path(settings["dataset"]) / name) != expected:
            raise ValueError(f"population source dataset differs: {name}")
    return {
        name: checksum(POPULATION / name)
        for name in (
            "settings.json",
            "inputs.jsonl",
            "input_manifest.json",
            "length_preflight.json",
        )
    }


def linux_pidfd_syscall(number, *arguments):
    # x86_64 syscall numbers verified against the local Linux UAPI header.
    # https://man7.org/linux/man-pages/man2/pidfd_open.2.html
    # https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("pidfd fallback is verified only on Linux x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return linux_pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_signal(descriptor, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(descriptor, sig)
    return linux_pidfd_syscall(
        424,
        ctypes.c_int(descriptor),
        ctypes.c_int(sig),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )


def fd_alive(descriptor):
    return not bool(select.select([descriptor], [], [], 0)[0])


def stop_child(child, record):
    if child is None or child.poll() is not None:
        return True
    for sig, seconds in (
        (signal.SIGINT, 60),
        (signal.SIGTERM, 20),
        (signal.SIGKILL, 10),
    ):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            record.setdefault("cleanup_timeouts", []).append(int(sig))
    record["cleanup_error"] = "own audit process survived bounded signal escalation"
    return False


def resume_population(record, resume_log, env):
    # Do not start a second worker if another authorized resume already owns
    # the output lock. A matching progress record must confirm that handoff.
    with (POPULATION / ".lock").open("a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            current = json.loads((POPULATION / "progress.json").read_text())
            active_pid = current.get("pid")
            cmdline_path = Path(f"/proc/{active_pid}/cmdline")
            cmdline = (
                cmdline_path.read_bytes().split(b"\0") if cmdline_path.exists() else []
            )
            if (
                current.get("status") in {"loading_model", "running", "evaluating"}
                and b"decoding.ragtruth_population" in cmdline
                and b"outputs/ragtruth_population_20260912" in cmdline
            ):
                record.update(
                    status="population_output_already_owned", observed_progress=current
                )
                return
            raise RuntimeError(
                "population lock contended without a matching active progress record"
            ) from None
    with resume_log.open("x") as stream:
        resumed = subprocess.Popen(
            [
                "bash",
                "scripts/run_ragtruth_population.sh",
                "--output",
                "outputs/ragtruth_population_20260912",
                "--resume",
            ],
            cwd=REANCHOR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record.update(
        status="population_resume_started",
        population_resume_launcher_pid=resumed.pid,
        resumed_unix=time.time(),
        resume_log=str(resume_log),
    )
    record["resume_verification_timeout_seconds"] = 900
    deadline = time.monotonic() + 900
    while resumed.poll() is None and time.monotonic() < deadline:
        current = json.loads((POPULATION / "progress.json").read_text())
        if current.get("pid") == resumed.pid and current.get("status") in {
            "running",
            "loading_model",
            "evaluating",
            "complete",
        }:
            with (POPULATION / ".lock").open("a") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    record.update(
                        status="population_resume_verified", resumed_progress=current
                    )
                    return
        time.sleep(1)
    record.update(
        status="population_resume_unverified", resume_exit_code=resumed.poll()
    )


def run(args):
    with (REANCHOR / "runs/relation_interleave_20260913.lock").open(
        "a"
    ) as scheduler_lock:
        fcntl.flock(scheduler_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)


def run_locked(args):
    args.output = args.output.resolve()
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    audit_log, resume_log = (
        args.journal.with_suffix(".audit.log"),
        args.journal.with_suffix(".population.log"),
    )
    if any(path.exists() for path in (args.journal, audit_log, resume_log)):
        raise FileExistsError("choose a new journal; existing artifacts are preserved")
    settings_path = args.output / "settings.json"
    if not settings_path.exists():
        raise ValueError(
            "output must already be prepared by soft_graph_runner --stage prepare"
        )
    prepared = json.loads(settings_path.read_text())
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with (args.output / ".phase.lock").open("a") as audit_lock:
        # Retain the SAME open-file-description through preflight and actual
        # child execution, eliminating a check-then-start lock race.
        fcntl.flock(audit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [
            str(PYTHON),
            "-u",
            "-m",
            "route_graph.soft_graph_runner",
            "--inputs",
            prepared["input_path"],
            "--output",
            str(args.output),
            "--observer-model",
            prepared["observer_model"],
            "--reader-model",
            prepared["reader_model"],
            "--inherited-lock-fd",
            str(audit_lock.fileno()),
        ]
        if prepared.get("evaluation_manifest") is not None:
            command += [
                "--evaluation-manifest",
                prepared["evaluation_manifest"]["path"],
            ]
        subprocess.run(
            command + ["--stage", "prepare"],
            cwd=GRAPH,
            env=env,
            pass_fds=(audit_lock.fileno(),),
            check=True,
        )
        prepared = json.loads(settings_path.read_text())
        if prepared["protocol"]["phase_order"] != ["A", "B", "C", "D", "merge"]:
            raise ValueError("prepared phase order differs")
        if checksum(Path(prepared["input_path"])) != prepared["input_sha256"]:
            raise ValueError("prepared native inputs changed")
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
            cmdline = (
                Path(f"/proc/{args.population_pid}/cmdline").read_bytes().split(b"\0")
            )
            if (
                b"decoding.ragtruth_population" not in cmdline
                or b"outputs/ragtruth_population_20260912" not in cmdline
            ):
                raise ValueError("PID is not the expected frozen population process")
            record = {
                "status": "prepared",
                "previous_progress": before,
                "started_unix": time.time(),
                "audit_output": str(args.output),
                "audit_settings_sha256": checksum(settings_path),
                "frozen_population": frozen,
                "command": command + ["--stage", "all"],
                "scheduler_sha256": checksum(Path(__file__)),
                "signal_target": "Linux_pidfd",
            }
            save(args.journal, record)
            execute_and_restore(
                args,
                audit_log,
                resume_log,
                command,
                env,
                audit_lock,
                descriptor,
                frozen,
                record,
            )
        finally:
            os.close(descriptor)


def execute_and_restore(
    args, audit_log, resume_log, command, env, audit_lock, descriptor, frozen, record
):
    child = population_lock = None
    signaled = False
    try:
        pidfd_signal(descriptor, signal.SIGINT)
        signaled = True
        record["status"] = "pausing"
        save(args.journal, record)
        deadline = time.monotonic() + 60
        while fd_alive(descriptor) and time.monotonic() < deadline:
            time.sleep(1)
        if fd_alive(descriptor):
            raise RuntimeError("population did not exit; audit was not launched")
        population_lock = (POPULATION / ".lock").open("a")
        try:
            fcntl.flock(population_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record["status"] = "population_lock_handoff_contended"
            raise
        record["completed_population_manifests"] = {
            str(p.relative_to(POPULATION)): checksum(p)
            for p in (POPULATION / "responses").glob("*/manifest.json")
        }
        record["status"] = "population_paused_lock_held"
        save(args.journal, record)
        with audit_log.open("x") as stream:
            child = subprocess.Popen(
                command + ["--stage", "all"],
                cwd=GRAPH,
                env=env,
                pass_fds=(audit_lock.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(status="running_native_audit", audit_pid=child.pid)
            save(args.journal, record)
            record["audit_exit_code"] = child.wait()
    finally:
        stopped = False
        try:
            stopped = stop_child(child, record)
        except (
            OSError,
            subprocess.SubprocessError,
            RuntimeError,
            ValueError,
            KeyError,
        ) as error:
            record["cleanup_error"] = repr(error)
            stopped = child is None or child.poll() is not None
        finally:
            if population_lock is not None:
                population_lock.close()
            if signaled and not fd_alive(descriptor):
                if stopped:
                    try:
                        if frozen_population() != frozen:
                            raise RuntimeError("frozen population metadata changed")
                        for name, expected in record.get(
                            "completed_population_manifests", {}
                        ).items():
                            if checksum(POPULATION / name) != expected:
                                raise RuntimeError(
                                    "completed population manifest changed"
                                )
                        resume_population(record, resume_log, env)
                    except (
                        OSError,
                        subprocess.SubprocessError,
                        RuntimeError,
                        ValueError,
                        KeyError,
                    ) as error:
                        record.update(
                            status="population_resume_failed", resume_error=repr(error)
                        )
                else:
                    record["status"] = "audit_cleanup_unresolved_population_not_resumed"
            save(args.journal, record)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k != "completed_population_manifests"
                    }
                ),
                flush=True,
            )
    if record.get("audit_exit_code") != 0 or record.get("status") not in {
        "population_resume_verified",
        "population_output_already_owned",
    }:
        raise RuntimeError(
            "audit or verified recovery did not complete; inspect the journal"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-pid", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Existing output from soft_graph_runner --stage prepare",
    )
    parser.add_argument("--journal", type=Path, required=True)
    run(parser.parse_args())
