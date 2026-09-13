# Surface owner runner engineering review — 2026-09-13

Scope: bounded code-review-and-quality / run-experiment review of `next_iteration/surface_runner.py` and `experiments/interleave_surface_owner.py`, with direct reads of `surface_verifier.py`, `surface_owner.py`, `surface_graph.py`, `reader_receipt.py`, `frozen_reader.py`, `audit_alignment.py`, and the reused `experiments/interleave_soft_graph.py` transaction. No GPU was started. This is an engineering/protocol review of the launch chain, not a result audit and not evidence of natural-method effectiveness.

Verification performed:

- `CUDA_VISIBLE_DEVICES=` CPU pytest from `graph/`: `tests/test_surface_owner.py tests/test_surface_verifier.py -q` → 21 passed.
- `python -m py_compile next_iteration/surface_runner.py experiments/interleave_surface_owner.py` → passed.
- `python -m next_iteration.surface_runner --help` → passed.
- `python -m experiments.interleave_surface_owner --help` from `graph/` → passed.
- Direct script invocation `python experiments/interleave_surface_owner.py --help` from `graph/` failed with `ModuleNotFoundError: No module named 'experiments'`; this is an invocation-contract issue, not a module syntax issue.

Verdict: no Critical result-integrity blocker. Two Required launch/recovery issues remain before treating this as a robust frozen GPU run. If the immediate launch command is the module invocation and the run is accepted as one-shot/fresh-output only, the core protocol is startable; if same-output recovery through the interleave wrapper is required, fix Required #2 first.

## Required

1. The launch command must be fixed to module mode, or the script import path must be made self-contained.

   Evidence: `experiments/interleave_surface_owner.py` imports `from experiments.interleave_soft_graph import ...` at lines 11-22. Running it as a file path from `graph/` puts `graph/experiments` on `sys.path`, not `graph`, so `python experiments/interleave_surface_owner.py ...` fails before argument parsing. Running `python -m experiments.interleave_surface_owner ...` from `graph/` works.

   Required action: freeze the actual launch command as:

   ```bash
   cd /share/home/tm902089733300000/a903202310/lys/research/graph
   CUDA_VISIBLE_DEVICES=<gpu> /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
     -m experiments.interleave_surface_owner \
     --population-pid <pid> --output <prepared_output> --journal <journal>
   ```

   or change the launcher import/bootstrap so file-path execution is valid. Do not rely on a human remembering this, because the generic run-experiment pattern often uses `python <script>`.

2. `--stage all` is not idempotent after a completed features phase, so same-output recovery through `interleave_surface_owner.py` can fail before finite starts.

   Evidence: `surface_runner.feature_stage` verifies/reuses existing feature chunks and B artifacts, then unconditionally calls `write_json_once(args.output / "feature_summary.json", ...)` at lines 223-235. If a run exits after features have completed and `feature_summary.json` exists, a retry through `interleave_surface_owner.py` still invokes `surface_runner --stage all`; it will revisit `feature_stage` and raise `FileExistsError` on the summary instead of proceeding to finite. A similar preserved-partial case exists if a chunk `.npy` exists but its `.json` receipt was not published yet; the next feature attempt opens the same `.npy` with `xb`.

   Required action: make `feature_summary.json` a verified idempotent artifact: if it exists, read it and check `feature_forward_calls`, document count, capture hash, and native count against the recomputed combined capture. For orphan chunk files, either reject with an explicit “preserved partial; choose new output” status before model load, or write chunks via a temporary file and atomically publish the final `.npy` only with its receipt. The smaller launch-level alternative is for `interleave_surface_owner.py` to detect completed features and call `--stage finite`, but artifact-level idempotence is cleaner.

## Checked and accepted

- Frozen snapshot and upstream binding are materially sound. `verify_preflight` checks the lexical-only preflight manifest, its `samples.jsonl` and `documents.json` artifact hashes, live-vs-preflight code hashes, and original input hash. `surface_runner.prepare` snapshots current `route_graph/*.py`, the relevant `next_iteration` modules, and both interleave scripts into `executed_code`, records model manifests, and rechecks `verify_executed_code` before stages and during feature chunks. B artifacts bind to the exact preflight sample and all feature files; C artifacts bind to the exact B file hash and row hash.

- The new CPU launch preflight closes the previous practical length/label gap. `prepare` now tokenizes all masked documents with the observer tokenizer, records document count/max/total and a length digest in `settings.length_preflight`, verifies all original observer rows with `align_row`, and checks that all finite labels in `SCNIUPFVK` are single Qwen tokens before freezing settings.

- Actual capture and reader counts are auditable. Feature capture records per-document input IDs and vector hashes, and `feature_summary.json` records `feature_forward_calls`, document count, capture hash, and `native_forward_calls: 0`. Finite C artifacts record per-response `reader_calls` from `FrozenReader.calls`, which counts only real model forward/generate calls; `reader_outcomes` separately records returned requests and cache hits. The `target_original` request used by `assess_target` and by the strict candidate checks is structurally identical, so the second use should be a cache return, and `validate_target_assessment` revalidates the receipt before the CN gate is consumed.

- The phase order and model residency are correct for one GPU. The protocol requires `features` then `finite`. The feature model is deleted and CUDA cache cleared before finite loads the reader. No native intervention path is called; all artifacts and protocol fields keep `native_forward_calls: 0` and `labels_used: False`.

- The reused pause/restore transaction is the right shape for the frozen population. The launcher holds the scheduler lock and the audit phase lock, verifies the population PID, status, command line and cwd, opens a pidfd before signaling, waits for exit, holds the population output lock while the audit child runs, records completed population manifests, stops its own audit child on failure, verifies frozen population metadata, and resumes the population via the existing script.

## Non-blocking operational notes

- The code-level CUDA guard checks for at least 20 GiB free on `cuda:0`; it is a runtime guard, not a full cluster-level GPU scheduler. Per run-experiment practice, record `nvidia-smi`/GPU assignment immediately before launch and use explicit `CUDA_VISIBLE_DEVICES=<gpu>` in the module command.

- `code_files()` freezes every `route_graph/*.py`, not only files imported by this path. That is conservative for reproducibility, but it means any unrelated route_graph edit after preparation will abort the run at `verify_executed_code`. Freeze code during the run or narrow the manifest only if this becomes an operational problem.

- The preflight counts, owner ambiguity counts, and future finite pass rates are candidate-supply diagnostics only. They must not be reported as accuracy, label coverage, or native mechanism evidence.

## Closure addendum after summary-recovery fix

Re-review scope: only the `feature_summary.json` recovery change requested after the first review. I did not modify implementation or start GPU.

Additional verification:

- CPU/mock exercised `feature_stage` with feature chunks already present: first summary publication succeeds, an identical existing `feature_summary.json` is accepted without overwrite, and a divergent existing summary raises `ValueError("existing feature summary differs from validated capture")`.
- The prepared `outputs/surface_owner_v1_20260913/settings.json` exists, records the live `next_iteration/surface_runner.py` hash, and includes `length_preflight` with 2377 masked documents, max length 812, total 108701, 36 aligned observer rows, and finite single-token labels validated.

Required #1 is closed by the operational launch contract if the only documented command is `python -m experiments.interleave_surface_owner` from `graph/`. I did not reclassify file-path invocation as valid; it remains invalid by Python import rules, but no blocker remains if the launch doc never uses it.

Required #2 is closed for the actual completed-features resume path. Lines 234-241 now rebuild the capture-derived summary and accept an existing file only when it exactly matches `{feature_forward_calls, documents, capture_sha256, native_forward_calls}`. This removes the previous failure where `--stage all` could not proceed to finite after a completed features phase.

Residual caveat: if a process dies after writing a chunk `.npy` but before publishing its paired chunk `.json`, the next attempt will still enter the pending chunk and fail later on the exclusive `.npy` open. That is a fail-closed partial-output case, not a result-integrity blocker for the fresh prepared run; if same-output recovery from mid-chunk crashes becomes a requirement, publish chunk arrays through a temporary file plus atomic receipt commit.

Final Required status for this bounded launch review: **Critical 0 / Required 0 for fresh frozen launch using the module command and the current prepared output**. Do not report this as effectiveness evidence; it only means the surface owner feature→finite validation chain is protocol-safe to start.
