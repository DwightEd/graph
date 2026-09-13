"""CPU-only safety checks for the native-audit interleave scheduler."""

import json
from argparse import Namespace

import pytest

from experiments import interleave_native_audit as scheduler


def test_prepare_failure_does_not_pause_population(tmp_path, monkeypatch):
    """All audit preparation must succeed before the scheduler can send SIGINT."""
    output = tmp_path / "audit"
    output.mkdir()
    (output / "settings.json").write_text(
        json.dumps(
            {
                "input_path": str(tmp_path / "inputs.jsonl"),
                "observer_model": str(tmp_path / "observer"),
                "reader_model": str(tmp_path / "reader"),
            }
        )
    )
    pidfd_signals, process_group_signals = [], []

    def fail_prepare(*_args, **_kwargs):
        raise RuntimeError("synthetic CPU-only prepare failure")

    def pidfd_open_must_not_run(*_args, **_kwargs):
        pytest.fail("prepare failure reached pidfd_open")

    monkeypatch.setattr(scheduler.subprocess, "run", fail_prepare)
    monkeypatch.setattr(
        scheduler,
        "pidfd_open",
        pidfd_open_must_not_run,
    )
    monkeypatch.setattr(
        scheduler,
        "pidfd_signal",
        lambda *args: pidfd_signals.append(args),
    )
    monkeypatch.setattr(
        scheduler.os,
        "killpg",
        lambda *args: process_group_signals.append(args),
    )

    with pytest.raises(RuntimeError, match="prepare failure"):
        scheduler.run_locked(
            Namespace(
                population_pid=12345,
                output=output,
                journal=tmp_path / "runs" / "interleave.json",
            )
        )

    assert pidfd_signals == []
    assert process_group_signals == []


def test_pidfd_fallback_uses_x86_64_linux_syscall_numbers(monkeypatch):
    """The fallback selects the documented pidfd syscalls without invoking libc."""
    calls = []

    def fake_syscall(number, *arguments):
        calls.append((number, arguments))
        return 77

    monkeypatch.delattr(scheduler.os, "pidfd_open", raising=False)
    monkeypatch.delattr(scheduler.signal, "pidfd_send_signal", raising=False)
    monkeypatch.setattr(scheduler, "linux_pidfd_syscall", fake_syscall)

    assert scheduler.pidfd_open(123) == 77
    assert scheduler.pidfd_signal(77, 0) == 77
    assert calls[0][0] == 434
    assert [argument.value for argument in calls[0][1]] == [123, 0]
    assert calls[1][0] == 424
    assert calls[1][1][0].value == 77
    assert calls[1][1][1].value == 0
    assert calls[1][1][2].value is None
    assert calls[1][1][3].value == 0


def test_pidfd_fallback_rejects_an_unverified_platform(monkeypatch):
    """Unknown syscall-number ABIs fail before calling libc."""
    monkeypatch.setattr(scheduler.platform, "system", lambda: "Linux")
    monkeypatch.setattr(scheduler.platform, "machine", lambda: "aarch64")

    with pytest.raises(RuntimeError, match="Linux x86_64"):
        scheduler.linux_pidfd_syscall(434)


def test_resume_verification_waits_past_legacy_timeout_for_loading_model(
    tmp_path, monkeypatch
):
    """A held lock alone is insufficient until the resumed PID becomes active."""
    population = tmp_path / "population"
    population.mkdir()
    (population / ".lock").touch()
    progress = population / "progress.json"
    progress.write_text(json.dumps({"pid": 9001, "status": "preparing"}))
    resume_log = tmp_path / "population.log"
    record, clock, lock_attempts = {}, {"seconds": 0}, []

    class Resumed:
        pid = 9001

        @staticmethod
        def poll():
            return None

    def fake_sleep(seconds):
        assert seconds == 1
        clock["seconds"] += 121
        progress.write_text(json.dumps({"pid": 9001, "status": "loading_model"}))

    def fake_flock(_stream, _operation):
        lock_attempts.append(clock["seconds"])
        if len(lock_attempts) == 2:
            raise BlockingIOError

    monkeypatch.setattr(scheduler, "POPULATION", population)
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *_args, **_kwargs: Resumed())
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: clock["seconds"])
    monkeypatch.setattr(scheduler.time, "sleep", fake_sleep)
    monkeypatch.setattr(scheduler.fcntl, "flock", fake_flock)
    monkeypatch.setattr(
        scheduler,
        "pidfd_signal",
        lambda *_args: pytest.fail("resume verification sent a signal"),
    )

    scheduler.resume_population(record, resume_log, {})

    assert clock["seconds"] > 120
    assert record["resume_verification_timeout_seconds"] == 900
    assert record["status"] == "population_resume_verified"
    assert record["resumed_progress"] == {"pid": 9001, "status": "loading_model"}
    assert lock_attempts == [0, 121]
