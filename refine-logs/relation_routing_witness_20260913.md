# O5 relation-routing independent execution witness — 2026-09-13

Engineering execution witness only. Scientific review: `REVIEW_UNAVAILABLE`.

## Instruction and environment check

Read `run-experiment/SKILL.md`, its compute environment contract,
`docs/RELATION_ROUTING_PLAN_20260913.md`, and `.aris/compute/local.md` before launch.
No applicable AGENTS.md or CLAUDE.md was found in the repository/ancestor checks.
The existing canonical environment spec hash was `8d044d57`, matching the ledger.
No install, environment rebuild, code edit, protocol edit, GPU retry, or cleanup occurred.
Preflight GPU 0 was the documented RTX 4090 (24,564 MiB), with 17,114 MiB used
by the existing population job. Its PID 19668 and output argument matched the
scheduler's required target. The scheduler, rather than a concurrent model load,
handled its pause and acquired the population lock.

Before launch, saved settings/code hashes and 5,040 already published response
manifest hashes to `/tmp/relation_routing_interleave_witness_20260913_before.json`.
The snapshot was taken while the population was progressing; the scheduler's
subsequent pause snapshot records 5,057 completed responses, zero failures.

## Documented invocation and exit

Executed exactly once from `reanchor`:

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python scripts/interleave_relation_only.py --population-pid 19668 --experiment relation_routing --output outputs/relation_routing_20260913
```

Captured scheduler stdout/stderr in
`/tmp/relation_routing_interleave_witness_20260913.log`.
The scheduler journal is `reanchor/runs/relation_routing_interleave_20260913.json`;
the model log is `reanchor/runs/relation_routing_20260913.log`.
The command and O5 child both exited 0. O5 model PID was 20315.
Started at 2026-09-13 01:32:37.647 Asia/Shanghai; population resume was launched
at 01:33:56.100. The interval from old process stop to resume launch was 76.44 s.
The O5 runner records 55.74 s and 16,507,489,792 bytes (15.37 GiB) peak CUDA allocation.
These are different timers and neither includes the later population resume validation.

## Artifact verification

CPU verification script:
`/tmp/verify_relation_routing_witness_20260913.py`.
Machine-readable check:
`/tmp/relation_routing_interleave_witness_20260913_verification.json`.

- `COMPLETE` exists; 175/175 O5 manifest file hashes verified.
- Exactly 167 unique planned conditions: 132 query conditions, 32 layer conditions,
  `all_queries`, `without_final`, and `sham`.
- Every condition saves finite, complete 128,256-dimensional endpoint logits.
  Independently recomputed endpoint argmax and fixed 14-minus-12 margin agree with rows.
- O5 `query_131_logits.npy` exactly equals the O4 final-query E endpoint.
- The sham endpoint exactly equals the O4 baseline endpoint. The runner additionally
  checked exact full baseline reproduction and full-query sham equality before publishing;
  this witness did not perform a second GPU forward.
- All 167 earlier-query maximum logit errors are zero; full-query sham error is zero.
- Maximum recorded source-mass floating point error is 2.980232238769531e-7.
- O5 executed code hashes and inherited O4 dependency hashes match current files.
- All 5,040 snapshotted population response manifest hashes, its settings hash, and
  all frozen population code hashes remain unchanged. This check was repeated after
  verified resumed progress, at Unix 1789234680.6571908, and again passed.

No document-versus-execution mismatch was observed. The saved endpoints support
further scoped analysis; these checks do not establish scientific validity,
unique lookback localization, or a deployable hallucination detector.

## Population restoration

Scheduler `finally` launched PID 20383 against the same
`outputs/ragtruth_population_20260912` directory with `--resume`.
Resume log: `reanchor/runs/ragtruth_population_resume_relation_routing_20260913.log`.
Restoration is verified, not inferred from process creation. The new PID's first
observed `running` state had 5,057 completed responses at Unix 1789234577.9876566.
A later state at Unix 1789234638.6785016 had 5,101 completed responses and zero
failures. All 44 newly published response manifests had modification times later
than `resumed_unix=1789234436.0998843`. The first new response was 12530; the latest
in this snapshot was 12788. Its four content file hashes were independently checked.
GPU allocation resumed with one model process. The complete restoration evidence
is `/tmp/relation_routing_interleave_witness_20260913_restoration.json`.

The CPU restoration verifier initially used the wrong manifest dictionary key
(`files_sha256` instead of the existing `files`); that verifier-only error was
corrected and the check passed. No execution file changed and no GPU experiment
was repeated. One live progress file read also briefly returned missing during a
publication; immediate subsequent reads and monotonic progress checks succeeded.
