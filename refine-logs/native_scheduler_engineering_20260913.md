# Native-audit interleave scheduler review — 2026-09-13

Reviewed `experiments/interleave_native_audit.py` independently against the
single-GPU scheduling requirement, the live frozen RAGTruth population
implementation, and the prior relation interleave script. This review is
limited to scheduler/process/lock/recovery behavior. It does not re-review the
native-audit runner or its measurement method.

## Result

**Do not run or merge the scheduler before the Critical and Required findings
are fixed.** The script preserves completed response directories by design and
does not launch an audit until the target PID has exited and the population
output lock is held. Those safeguards are not enough to guarantee recovery on
all exception paths.

## Critical

1. **A child-stop exception bypasses population recovery** —
   `experiments/interleave_native_audit.py:157-160`.

   The only recovery path is below `os.killpg(...); child.wait(timeout=60)` in
   the same `finally` block. `os.killpg` can raise `ProcessLookupError` after
   `poll()` races with a child exit, and `wait(timeout=60)` can raise
   `TimeoutExpired`. Either exception exits the `finally` block before the
   population lock is closed, the resume command is launched, and the journal
   records recovery. An interrupt during the audit can therefore leave the
   authorized population frozen indefinitely. Put audit termination in a
   nested best-effort cleanup, handle the process-group race and timeout (with
   a bounded escalation), and put lock release plus resume in an outer
   unconditional cleanup. Persist the failed cleanup state before attempting
   resume.

## Required

1. **The frozen-population checksum preflight always rejects the existing
   population contract** — `experiments/interleave_native_audit.py:42-45`.

   The live `outputs/ragtruth_population_20260912/settings.json` stores
   `input_sha256` as a mapping for `source_info.jsonl` and `response.jsonl`.
   The scheduler compares the SHA-256 string of its generated `inputs.jsonl`
   against that mapping, so it always raises `ValueError("population frozen
   input differs")`. The same broken check executes again in recovery. Use
   the population's `input_manifest.json["sha256"]` for `inputs.jsonl`, and
   independently validate each dataset input against its named expected digest
   if that is part of the freeze contract. Add a fixture copied from the real
   settings schema and a positive preflight test.

2. **The advertised preparation step cannot prepare a new audit output** —
   `experiments/interleave_native_audit.py:76-92`.

   `run_locked` reads `args.output/settings.json` before it runs
   `audit_runner --stage prepare`. But that runner is the component that
   creates `settings.json` for a new output. With the currently prepared
   `outputs/native_audit_design_20260913` directory, no `settings.json` is
   present, so the scheduler exits before its intended concrete preflight. Run
   and validate `--stage prepare` first, then load and record the settings; or
   explicitly require a separately prepared output and remove the misleading
   preparation invocation. In either design, require a nonempty settings
   digest, expected input digest, and the intended phase order before SIGINT.

3. **The target identity has a PID-reuse race immediately before SIGINT** —
   `experiments/interleave_native_audit.py:106-129`.

   The code verifies `/proc/<pid>/cmdline`, then later signals the numeric PID.
   If that process exits in between and Linux reuses the PID, the scheduler can
   signal an unrelated process. Capture `/proc/<pid>/stat` start time while
   validating the command, then re-read and compare it immediately before
   `kill`; also fail closed when `/proc` disappears. Record that immutable
   process identity in the journal.

4. **Failure to acquire the original output lock is not represented as a
   recovery state** — `experiments/interleave_native_audit.py:134-165`.

   A contended `.lock` raises before native launch, but `finally` still starts
   a resume launcher whenever the original PID is absent. The launcher PID is
   recorded as `population_resume_started` even though the resumed Python
   process can immediately fail on that same lock. Treat lock contention as a
   distinct, journaled handoff failure. Do not claim recovery started until the
   new process has acquired the output lock and published a matching progress
   record, or provide a bounded supervisor that verifies those conditions.

5. **The preflight does not prove the `all` driver is available before the
   population is stopped** — `experiments/interleave_native_audit.py:101` and
   `route_graph/audit_runner.py:279-280`.

   `prepare` holds `.phase.lock`, whereas the actual `all` run holds
   `.driver.lock`. Another audit can own `.driver.lock` while preparation
   succeeds. This scheduler can then pause the population and launch an audit
   that exits immediately on driver-lock contention. Reserve/check the driver
   lock as part of preflight, using a scheduler-owned protocol that cannot race
   between check and child launch.

## Optional

1. **Use a structured journal state machine.** Record `prepared`,
   `population_paused`, `population_lock_held`, `audit_started`,
   `audit_exited`, and `resume_verified` atomically. This makes manual
   recovery possible after host restart or an uncatchable process termination,
   and avoids overloading `status` with an optimistic launcher event.

2. **Validate the currently expected progress states deliberately.** The
   population publishes `loading_model`, `running`, `evaluating`, and terminal
   states. Requiring exactly `running` is safe but can reject a valid scheduled
   handoff during startup; make the allowed live states explicit and document
   why each is safe to interrupt.

3. **Snapshot all preservation-relevant metadata in the journal.** Completed
   response manifests are checked, which is the important immutable result
   set. Including the existing failure log digest and a listing of partial
   directories would make the promise to preserve untracked results auditable,
   even though the audit itself writes elsewhere.

## Confirmed safeguards

- The shared `relation_interleave_20260913.lock` serializes this scheduler with
  the earlier relation interleave.
- The native audit is not spawned until the target process is observed dead
  and a nonblocking exclusive lock is obtained on the population output.
- The audit process starts in a new session, and a normal interruption targets
  its whole process group.
- Population response publication is atomic directory rename; the scheduler
  snapshots each completed response manifest after the pause and does not
  modify those directories.
- The CPU regression test below verifies that a failed audit preparation sends
  no signal at all.

## Verification

No signal was sent and no GPU task was started. The live population command
was inspected read-only and is the expected dedicated Python process with its
own session. Ran:

```text
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -m pytest -q tests/test_native_scheduler.py
/tmp/research_lint_20260912/bin/ruff check \
  experiments/interleave_native_audit.py tests/test_native_scheduler.py
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -m py_compile experiments/interleave_native_audit.py
```

## Re-review of the rewritten scheduler

The rewrite resolves the earlier Critical child-cleanup escape and the prior
Required findings about the population input schema, preparation ordering, PID
reuse, and audit lock handoff. The scheduler now keeps one `.phase.lock` open
through preflight and audit execution, passes the same open-file-description to
the runner, uses a Linux pidfd for the actual SIGINT target, and uses a bounded
SIGINT/SIGTERM/SIGKILL audit cleanup before attempting recovery. Its launched
resume checks both a matching progress PID and a contended output lock.

**Result: Critical 0, Required 1. Do not launch until the remaining recovery
verification issue is fixed.**

### Required

1. **An arbitrary lock holder plus stale `running` progress is accepted as a
   verified population owner** — `experiments/interleave_native_audit.py:84-96`.

   If the initial output-lock probe is contended, the scheduler accepts any
   `progress.json` whose status is one of `loading_model`, `running`, or
   `evaluating`. It does not verify that `current["pid"]` is alive, is the
   expected `decoding.ragtruth_population` command, or still owns the frozen
   output. A stale pre-pause progress file combined with an unrelated lock
   holder is therefore recorded as `population_output_already_owned`; the final
   success gate accepts that state. Require a live pidfd (or equivalent
   identity-safe process check) for `current["pid"]`, verify its expected
   command/output, and only then accept a contended lock as a recovered
   population. Otherwise report recovery unverified and fail closed.

### Targeted CPU verification

`tests/test_native_scheduler.py` now patches `subprocess.run`, the scheduler's
pidfd wrappers, and `os.killpg`. Its synthetic
prepare failure asserts that the scheduler neither opens a pidfd nor sends a
pidfd or process-group signal. It starts no process and uses no GPU.

## Re-review follow-up: runtime blocker

The active-lock branch now also verifies that the progress PID has the expected
population command and output argument. That resolves the stale-progress
finding above for the scheduler's recovery evidence.

**Result: Critical 0, Required 1.**

### Required

1. **The required interpreter does not expose either pidfd API used by the
   scheduler** — `experiments/interleave_native_audit.py:218,261`.

   The requested interpreter is Python 3.11.15 on Linux 5.15, yet read-only
   inspection returns `hasattr(os, "pidfd_open") == False` and
   `hasattr(signal, "pidfd_send_signal") == False`. After successful CPU
   preflight, the scheduler will raise `AttributeError` before its first
   signal, so it cannot launch. Add a tested Linux syscall wrapper (or run
   under an interpreter that provides both APIs) and fail during the earliest
   preflight with a clear compatibility error if that capability is absent.
   Preserve the pidfd-only target rule; do not fall back to numeric-PID
   `kill`.

## Final pidfd fallback re-review

The scheduler now supplies `pidfd_open` and `pidfd_signal` wrappers. They use
the interpreter APIs when present and otherwise call the local Linux x86_64
UAPI syscall numbers (`pidfd_open=434`, `pidfd_send_signal=424`) through
`ctypes`, rejecting every other platform before a syscall is made. It performs
the required self-pidfd signal-0 capability probe after audit preparation and
before opening or signaling the population pidfd.

**Result: Critical 0, Required 0.**

The targeted CPU tests cover a prepare failure that cannot reach either
scheduler wrapper or a process-group signal, fallback argument/number routing
without invoking libc, and failure on an unverified architecture. They pass
with the requested interpreter. A separate self-only witness opened a pidfd
for the test process, sent signal 0, and reported `result=0`,
`alive_before=True`, and `alive_after=True`. It did not signal any other
process or start GPU work.
