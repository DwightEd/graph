# Native v2 fresh agent-follows-doc witness — 2026-09-13

The documented full natural batch was executed exactly once. The audit child and persistent scheduler invocation both exited 0. A, B, C, D, and merge each recorded `phase_complete` for all 36 responses. Subsequent read-only monitoring confirmed the restored population actually advanced. This is an execution/environment witness, not evidence of empirical method effectiveness.

## Isolation and invocation

The fresh witness read only the run-experiment skill, its shared compute environment contract, the project provider ledger entry `research@c7b4f607`, and `docs/NATIVE_AUDIT_V2_RUN_20260913.md` before launching. Preparation and engineering review were supplied as completed prerequisites by the parent. Existing environment and weights were reused; no installation, rebuild, additional kernel/GPU job, repair, retry, manual signal, or evaluation invocation was performed.

Working directory: `/share/home/tm902089733300000/a903202310/lys/research/graph`.

Exact invocation, executed once:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u experiments/interleave_native_audit.py --population-pid 141861 --output outputs/native_audit_v2_20260913 --journal ../reanchor/runs/native_audit_v2_20260913.json
```

Persistent exec session: **91196**, retained until natural completion, wrapper exit **0**. Audit child PID: **143662**, journal `audit_exit_code: 0`. Original population PID: **141861**. Restored population PID: **145947**. The wrapper's own OS PID is not published in the permitted monitoring records; no PID is inferred for it.

The wrapper printed prepared-output verification for 36 responses with `settings_sha256: 1980c5bf35406afc048d3b61cda67db4cb76fa799f667ed9d69e77f5b1342cd1`. Journal `audit_settings_sha256` is `8d5dba0df8d473101848442bb0f7280e7a92ed22522a413f5c82285ff644fec4`; these are recorded as named by their respective emitters, without assuming identical hash semantics. Journal scheduler SHA256 is `8122aadfb6f4383d4707dfdd6c0b3e3c3ba8039f4cb928c919b9360f7a22b92d`.

## Observed execution

The command initially remained alive without stdout or a journal; original population progress advanced during this interval. It then verified preparation and created a `running_native_audit` journal. Journal start time: **2026-09-12T21:42:00.802018+00:00**. Its population snapshot contained **11,985 completed manifest hashes**, with **0 failures** and signal target `Linux_pidfd`.

| Phase | Exact terminal log record |
|---|---|
| A | phase_complete, completed_stage 36, total 36 |
| B | phase_complete, completed_stage 36, total 36 |
| C | phase_complete, completed_stage 36, total 36 |
| D | phase_complete, completed_stage 36, total 36 |
| merge | phase_complete, completed_stage 36, total 36 |

Final audit progress recorded `status: complete`, `responses: 36`, `claim_supported: not_evaluated`, PID 143662, at **2026-09-12T22:11:01.094712+00:00**. All phase terminal records were read from the exact documented audit log, including C and merge, which completed between periodic progress polls.

## Restoration and preservation

Journal final status is `population_resume_verified`; restoration launcher PID is 145947; restoration timeout is 900 seconds. Restoration launch timestamp is **2026-09-12T22:11:35.458948+00:00**. The wrapper's own stored verification snapshot was `loading_model`, PID 145947, completed 11,985, failed 0, at **2026-09-12T22:14:39.932053+00:00**. Thus its exit alone did not demonstrate a newly completed response.

A separate read-only observation after wrapper completion confirmed **status running**, **PID 145947**, **12,012 completed / 17,790**, **failed 0**, at **2026-09-12T22:15:41.241630+00:00**: **27 additional completed responses** beyond the snapshot. This closes the actual-advancement observation beyond the wrapper's loading-model snapshot.

The scheduler completed its documented preservation/restore path and retained the 11,985-entry manifest snapshot in the journal. This witness did not independently open or re-hash those population files, nor independently inspect the OS output lock; lock verification is the scheduler's reported check. Monitoring stayed within the five documented progress/log/journal paths. The witness performed no unrelated filesystem edits, deletions, cleanup, or untracked-file replacement; this report is created as a new file.

## Warnings, divergence, and boundaries

No traceback, `Error:`, or `Exception:` marker occurred in the exact audit log, and no wrapper failure occurred. The log did contain these two Transformers messages:

- Generation flags `temperature`, `top_p`, and `top_k` are not valid and may be ignored.
- `generation_config` defaults were modified to model-specific `do_sample: True`.

No flags or source were changed in response. These warnings are retained as observed runtime details, not silently repaired. There was no invocation refusal or retry. The material reporting nuance is that the wrapper marked restoration verified while its stored progress said `loading_model`; actual advancement was established by the subsequent observation above.

Native-forward and selected-window counts, semantic coverage, annotation comparison, and factual-effectiveness claims were not established by these progress records. Evaluation was deliberately left to the parent, as required by the invocation document. Phase completion cannot establish that a native intervention occurred or that any prediction was correct.

Evidence paths (relative to graph):

- `outputs/native_audit_v2_20260913/progress.json`
- `../reanchor/runs/native_audit_v2_20260913.audit.log`
- `../reanchor/runs/native_audit_v2_20260913.json`
- `../reanchor/runs/native_audit_v2_20260913.population.log`
- `../reanchor/outputs/ragtruth_population_20260912/progress.json`
