# Native scheduler v2 recovery-timeout review — 2026-09-13

Reviewed the recovery-verification change in
`experiments/interleave_native_audit.py`. Scope is limited to the observed v1
failure: the native audit exited successfully, but the scheduler returned exit
1 before the resumed population had finished validating the large existing
response set. No scheduler implementation or prior journal was modified in
this review.

## Result

**Critical 0, Required 0.** The 120-to-900 second bound fixes the observed
false-negative without weakening the handoff proof.

The v1 journal records `audit_exit_code: 0` and
`population_resume_unverified`. Its preserved population log shows the resumed
worker first rebuilding the 17,790-item roster and validating the already
completed population artifacts before it could publish `loading_model`. The
current resumed PID then published active progress and advanced from the v1
pause at 11,515 completed responses to more than 11,600. This is consistent
with a slow, healthy startup caused by immutable artifact validation, not a
lost worker or an audit failure.

## Assessment

- **Correctness:** `resume_population` now records
  `resume_verification_timeout_seconds: 900` and waits for that bound. It still
  rejects `preparing`: success requires the launcher PID to publish one of
  `loading_model`, `running`, `evaluating`, or `complete` and a nonblocking
  probe must show the population output lock is actually held.
- **Recovery safety:** The longer wait neither launches a second population nor
  relaxes the PID/lock proof. A dead launcher, no active progress, or an
  unlocked output continues to end as `population_resume_unverified`.
- **Readability and architecture:** Keeping the timeout in the journal makes
  the operational policy auditable. The change remains local to recovery
  verification; it does not affect audit phases, GPU loading, artifact
  immutability, or pidfd signaling.
- **Security and performance:** The scheduler continues to use the verified
  PID plus held output lock for handoff. The extra wall time occurs only while
  waiting for a launched resume to prove itself; it avoids treating a known
  O(number-of-completed-artifacts) validation phase as failure.

## Targeted CPU test

Added `test_resume_verification_waits_past_legacy_timeout_for_loading_model`.
It uses a fake launcher, monotonic clock, fake sleep, in-memory progress file,
and simulated lock contention. The population remains `preparing` through 121
synthetic seconds, then publishes matching-PID `loading_model` with a held
lock. The test proves that the scheduler waits past the former 120-second
limit, records the 900-second bound, and verifies only after both active
progress and lock evidence appear. It patches `pidfd_signal` to fail if called;
it starts no process, sends no signal, and uses no GPU.

## Verification

```text
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -m pytest -q tests/test_native_scheduler.py
# 4 passed

/tmp/research_lint_20260912/bin/ruff check \
  experiments/interleave_native_audit.py tests/test_native_scheduler.py
# All checks passed

/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -m py_compile experiments/interleave_native_audit.py
```
