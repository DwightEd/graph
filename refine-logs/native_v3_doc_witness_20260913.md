# Native v3 fresh document-execution witness — 2026-09-13

Status: the documented wrapper execution and population restoration passed. The run executed **zero native forwards** and produced all-abstention outputs; this is execution evidence, not evidence of effective native interventions or empirical method validity.

The fresh witness read only the run-experiment skill, its shared compute-environment contract, `graph/.aris/compute/local.md`, and `graph/docs/NATIVE_AUDIT_V3_RUN_20260913.md` before launching. The documented Fresh witness command was executed verbatim exactly once. No package installation, rebuild, preparation invocation, evaluation, code modification, retry, or independent GPU workload was performed by this witness.

Working directory: `/share/home/tm902089733300000/a903202310/lys/research/graph`.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u experiments/interleave_native_audit.py --population-pid 145947 --output outputs/native_audit_v3_20260913 --journal ../reanchor/runs/native_audit_v3_20260913.json
```

- Persistent exec session: `65424`; launched with a PTY and retained until completion. No stdin signal or termination was sent.
- Wrapper exit code: **0**, directly returned by the persistent exec session.
- Native audit PID: `147669`; native child exit code: **0**, recorded in the journal.
- Journal start: Unix `1789252425.59781`; native complete progress: Unix `1789257159.8175285`.
- Population at interruption: PID `145947`, completed `12574/17790`, failed `0`.
- Final journal status: `population_resume_verified`; restoration deadline configured as `900` seconds.

## Actual execution and output

The documented optional CPU summary was run after native progress became complete, using the documented research Python. It exited **0** and reported `label_join_performed: false`. No evaluator or annotation-label join was run by this witness.

| Evidence | Observed result |
|---|---:|
| Input responses / SHA256 | 36 / `c2d5712bf5f08ddaf34427eb483e98f38433d9a09847a857efe44bbf0de0b264` |
| Completed A / B / C / D / merge files | 36 / 36 / 36 / 36 / 36 |
| Nonempty B / C / D payloads | 0 / 0 / 0 |
| Selected questions | 0 |
| Risk claims / risk claims with valid contrast | 0 / 0 |
| Actual native `forward_calls`, summed directly over 36 merge artifacts | **0** |
| `native_tokens_processed` | 0 |
| Native groups / origin artifacts | 0 / 0 |
| Semantic-scored words / mechanism-scored words | 0 / 0 |
| Words retained | 4733 |
| Semantic abstentions / mechanism abstentions | 4733 / 4733 |

These forward counts were read from the actual merge fields, not inferred from phase status or model-loading logs. Every B/C/D envelope exists, but each contains an empty `data` object. The model reader's A requests do not count as native interventions.

The optional summary reported 368 question outcomes: 271 `uncertain` and 97 `invalid_question`. Verification statuses were 267 `uncertain` and 4 `supported`; none became a selected question. A accounted for 1064 returned requests: 545 strict JSON, 73 recovered JSON, 175 invalid-JSON failures, and 271 label-logit requests; cache hits were 0. It reported 174 source-reader invalid-JSON errors, 1 extraction invalid-JSON error, and 53 ambiguous source-citation occurrences. Atomic budget status was complete for all 36 responses, with 0 unprocessed units. These are observed semantic-interface failures and abstentions, not runtime success criteria or scientific approval.

## Original population restoration witness

The wrapper resumed the original population workload under PID **153564**. The resumed command was directly observed in `/proc/153564/cmdline` as the documented research Python running `-u -m decoding.ragtruth_population --output outputs/ragtruth_population_20260912 --resume`.

The journal's restoration snapshot was `loading_model`, PID 153564, completed 12574, failed 0, updated Unix `1789257395.1514058`. This is the wrapper's actual verification snapshot; it was not a post-resume advancement witness by itself. This agent subsequently read actual `running` progress and observed advancement:

| Observation | Completed | Failed |
|---|---:|---:|
| Before interruption, PID 145947 | 12574 | 0 |
| Running after restoration, PID 153564, Unix `1789257455.7728155` | 12602 | 0 |
| Later running observation, PID 153564, Unix `1789257497.6337442` | 12623 | 0 |

At the independent process/lock check, `/proc/153564/status` showed state `R (running)`. File descriptor 3 pointed to `/share/home/tm902089733300000/a903202310/lys/research/reanchor/outputs/ragtruth_population_20260912/.lock`; its fdinfo contained `FLOCK ADVISORY WRITE 153564 ... 0 EOF`. This verifies that the actual restored process held the population output lock.

This agent independently recomputed every journal-snapshotted population manifest hash: **12574 checked, 0 mismatches**. The four frozen population files (`settings.json`, `inputs.jsonl`, `input_manifest.json`, `length_preflight.json`) also had **0 mismatches**. All **8** code/document hashes in the frozen population settings matched the existing bytes. Restoration therefore includes actual resumed progress and lock possession while preserving the snapshotted completed work.

## Document versus reality

No operational mismatch was found in the documented working directory, invocation, required paths, input hash, phase publication, wrapper exit, or population-restoration behavior. The unchanged environment was reused; this witness did not rebuild it and does not establish clean-install reproducibility.

The important output limitation is explicit: all 36 phase sets completed, but no candidate was selected and **no native intervention was executed**. The document already cautions that completed phase files do not prove intervention. Any interpretation of this run as validating native causal controls, origin recovery, nonzero mechanism coverage, or scientific effectiveness would exceed the evidence.

The audit log emitted Transformers warnings about ignored generation flags and model-default `do_sample=True`. They caused no invocation failure. This witness does not infer that sampling was enabled from that warning alone. Progress was published at response/phase boundaries; absence of per-token logging was not treated as a stall.

Two auxiliary read-only inspection attempts encountered unavailable `rg` and a mistaken generic word-field assumption (`abstain` versus the actual `semantic_abstain` / `mechanism_abstain`). Subsequent read-only inspection used the actual schema. Neither issue was in the documented workload invocation or optional summary, neither changed an experiment file, and neither represents a wrapper retry or workload failure.

Primary artifacts: `graph/outputs/native_audit_v3_20260913/{progress.json,A,B,C,D,merge}`, `reanchor/runs/native_audit_v3_20260913.json`, `reanchor/runs/native_audit_v3_20260913.audit.log`, `reanchor/runs/native_audit_v3_20260913.population.log`, and `reanchor/outputs/ragtruth_population_20260912/progress.json`.
