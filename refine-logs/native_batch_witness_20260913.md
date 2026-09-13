# Native batch doc witness — 2026-09-13

**Documented invocation failed the complete witness gate: wrapper exit code 1.** The audit child exited 0 after all 36 responses passed through A, B, C, D, and merge. The wrapper could not verify population recovery before exiting and recorded `population_resume_unverified`. This is not a successful end-to-end witness or evidence of scientific validity.

## Scope and exact invocation

Fresh agent read only the run-experiment skill, its compute-env-contract reference, the provider ledger `.aris/compute/local.md`, and `docs/NATIVE_AUDIT_RUN_20260913.md`. CPU prepare was already completed by the author. This witness executed the following documented Full natural batch invocation once, without changes, a separate prepare call, a retry, a GPU smoke, installation, code edits, or signals from the witness agent.

Working directory: `/share/home/tm902089733300000/a903202310/lys/research/graph`

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u experiments/interleave_native_audit.py --population-pid 20383 --output outputs/native_audit_v1_20260913 --journal ../reanchor/runs/native_audit_v1_20260913.json
```

Persistent exec session: **38777**. Visible audit child PID: **139735**. Original population PID: **20383**. Recovery launcher/progress PID: **141861**. The wrapper PID was not exposed in the permitted monitored files, so it is not inferred here.

## Observed execution

The initial audit files were absent while population progress continued from 11465 to 11502. The wrapper then recorded `pausing`, `signal_target=Linux_pidfd`, and original population completed count **11515/17790**. The journal froze **11515** completed population manifest entries. After `running_native_audit` and audit PID 139735 appeared, the audit log showed model loading and the first A response.

The audit log contains exactly 36 `running` response records for each phase and these phase-complete records in order:

| Phase | Completed / total | Observed progress completion timestamp (UTC) |
|---|---:|---|
| A | 36 / 36 | 2026-09-12T21:18:16.072Z |
| B | 36 / 36 | 2026-09-12T21:19:27.197Z |
| C | 36 / 36 | 2026-09-12T21:20:18.304Z |
| D | 36 / 36 | 2026-09-12T21:21:23.994Z |
| merge | 36 / 36 | Complete before final audit progress below |

Final audit progress at 2026-09-12T21:23:30.085Z: `status=complete`, `responses=36`, `claim_supported=not_evaluated`. The journal records `audit_exit_code=0`. The audit log has no traceback. It does include generation configuration warnings about ignored temperature/top_p/top_k and the model-specific `do_sample=True` default; these warnings were reported without modifying the invocation or generation settings.

## Failure and recovery evidence

The scheduler journal started at 2026-09-12T20:48:57.997Z. It records recovery launch at 2026-09-12T21:24:05.632Z, recovery PID 141861, and `resume_exit_code=null`. At the actual wrapper exit, population progress was still `preparing`, PID 141861, with preparation started at 2026-09-12T21:24:28.549Z. The wrapper's actual exec return was **exit code 1**, with:

```text
RuntimeError: audit or verified recovery did not complete; inspect the journal
```

Final wrapper journal status remains `population_resume_unverified`. The failure occurs at `experiments/interleave_native_audit.py:397` according to its traceback. The witness did not inspect or patch that source.

At the later read-only observation 2026-09-12T21:27:19.043Z, population progress had advanced from preparing to **loading_model**, PID 141861, completed **11515**, failed **0**, remaining **6275**. Its progress timestamp was 2026-09-12T21:27:18.012Z and the recovery log showed 4/4 model shards loaded. This shows recovery continued after the wrapper exit, but this observation alone does not verify resumed response production or the output lock. No retry or process signal was sent.

A further read-only observation at 2026-09-12T21:28:23.442Z independently confirmed **population status=running**, PID **141861**, completed **11562/17790**, failed **0**, remaining **6228**. This is **47 newly completed responses** beyond the paused count 11515. The recovery log contains corresponding DONE records through response ID 5474, and progress was processing response 5480. Actual response production therefore resumed after the wrapper had already failed its verification deadline. The wrapper's exit code remains 1 and its journal remains `population_resume_unverified`; neither was rewritten. The permitted observations do not independently certify output-lock ownership.

## Documentation versus reality and limits

The document's full-batch invocation was accepted and the A–D/merge sequence ran for the complete frozen roster. Its required end-to-end recovery gate did not pass: the wrapper exited 1 with recovery unverified. An environment-ready or successful doc-witness declaration is therefore not supported by this invocation.

This witness monitored only the five paths specified by the invocation document, without directory recursion. It did not independently rehash frozen population artifacts or probe output locks outside those paths. Manifest preservation is recorded by the scheduler's frozen snapshot; it is not claimed as a separate witness-agent rehash.

Phase completion does not establish candidate coverage, actual native interventions, successful hallucination detection, or semantic-estimate validity. The parent separately reported zero risky/selected windows from A; that analysis was not independently recomputed by this witness, and the report makes no positive native-intervention claim.

Evidence paths:

- Audit progress: `outputs/native_audit_v1_20260913/progress.json`
- Audit log: `../reanchor/runs/native_audit_v1_20260913.audit.log`
- Scheduler journal: `../reanchor/runs/native_audit_v1_20260913.json`
- Recovery log: `../reanchor/runs/native_audit_v1_20260913.population.log`
- Population progress: `../reanchor/outputs/ragtruth_population_20260912/progress.json`
