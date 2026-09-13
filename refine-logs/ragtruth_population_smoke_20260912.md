# RAGTruth population mechanism: independent execution witness

2026-09-12. Fresh execution witness following `graph/docs/RAGTRUTH_POPULATION_MECHANISM_PLAN_20260912.md` and `graph/.aris/compute/local.md`. Reused `research@8d044d57`; no installs, downloads, or environment rebuild. No applicable `AGENTS.md` or `CLAUDE.md` was found in the repository ancestors and directories touched by this task. This report is execution evidence, not independent scientific approval. `REVIEW_UNAVAILABLE` remains applicable.

## Preflight and command

Before execution the output directory did not exist. Independent `nvidia-smi` check: NVIDIA GeForce RTX 4090, 24564 MiB total, 1 MiB used, 0% utilization, no compute process. The documented command was invoked exactly once for GPU capture from `reanchor`:

```bash
bash scripts/run_ragtruth_population.sh --output outputs/ragtruth_population_smoke_20260912 --smoke
```

Capture exit code: **0**. Process wall time: **47.497 s**. Capture PID: 16518. Log: `/tmp/ragtruth_population_smoke_20260912.log`. The output reached `complete`, 6 completed / 6 total, 0 failures, 0 remaining. Evaluation also finished before process exit.

## Coverage and numerical checks

All six responses have all eight conditions (`base`, `sham`, `source_01`, `source_1`, `history_01`, `history_1`, `source_permute`, `mlp_01`). Total response tokens: **1491**, corresponding to **11928 condition × prediction-token rows**. Each response stores 88 metric arrays. Baseline attention profiles have shape `[32 layers, response tokens, 32 heads]`; source/history message norms and MLP norms have shape `[32, response tokens]`.

| Response | Task | Generator | Input tokens (forward) | Response tokens | Capture seconds | Peak allocated GiB |
|---|---|---|---:|---:|---:|---:|
| 11859 | QA | llama-2-7b-chat | 252 | 47 | 1.088 | 15.145 |
| 17623 | QA | gpt-3.5-turbo-0613 | 1041 | 404 | 2.550 | 15.343 |
| 2234 | Summary | mistral-7B-instruct | 2616 | 431 | 6.408 | 17.151 |
| 3 | Summary | llama-2-7b-chat | 770 | 158 | 1.523 | 15.210 |
| 5661 | Data2txt | llama-2-7b-chat | 752 | 136 | 1.491 | 15.210 |
| 7870 | Data2txt | llama-2-13b-chat | 2248 | 315 | 5.015 | 16.594 |

Maximum actual forward input: **2616 tokens**. Maximum CUDA allocated peak: **17.151 GiB** (18415733760 bytes). These are runner-recorded allocated peaks, not sampled GPU resident-memory maxima. No OOM or truncation was observed.

- Verified all 24 data-file hashes listed by the six response manifests and all six input digests against the frozen roster. Verified `inputs.jsonl` against its manifest.
- Every metric array is finite. All sham metric arrays exactly equal the baseline arrays, including zero JS, logp change, margin change, and argmax change. Sham hidden-state maximum error is 0 for every response.
- All eight conditions preserve the earlier prompt prefix exactly (`prompt_state_max_error = 0`), with the final prompt position serving as the first response prediction query.
- Fixed baseline top-two token IDs are identical across all conditions. Every non-sham condition has at least one nonzero JS effect in each response.
- Both history interventions have zero JS for the first 17 response prediction positions, as expected with the 16-token remote-history distance rule.
- Source endpoint permutation preserves source attention mass within the recorded maximum floating-point error of 4.172325134e-07.

## Real annotation evaluation

Five of six task/generator/split groups contain both positive and negative token labels, so the real ranking-metric computation (including its seeded entry point) was executed successfully, not bypassed through the one-class guard. Each of these five groups has nine ranking metrics for both all-token and through-first-error subsets. The remaining QA/Llama-2-7B group has no error token, and its ranking metrics correctly remain null.

| Task / generator | All tokens | Error tokens | Non-null all-token ranking metrics |
|---|---:|---:|---:|
| Data2txt / llama-2-13b-chat | 315 | 2 | 9 |
| Data2txt / llama-2-7b-chat | 136 | 3 | 9 |
| QA / gpt-3.5-turbo-0613 | 404 | 13 | 9 |
| QA / llama-2-7b-chat | 47 | 0 | 0 |
| Summary / llama-2-7b-chat | 158 | 11 | 9 |
| Summary / mistral-7B-instruct | 431 | 15 | 9 |

The capture and the two resume runs produced three evaluation files. Their group results are exactly identical. Coverage records differ in process/timing state as expected. These tiny groups demonstrate execution coverage only and do not establish useful detector performance. The protocol replays older generators’ answers with a local Llama-3.1 observer; it does not claim to recover their original internal trajectories or validate ownership labels.

## Resume and interrupted-manifest recovery

Both recovery checks used the exact original command with `--resume` appended:

```bash
bash scripts/run_ragtruth_population.sh --output outputs/ragtruth_population_smoke_20260912 --smoke --resume
```

1. Normal resume: exit **0**, **14.813 s**. All 30 response files (including manifests) retained identical hashes. Log contained neither model checkpoint loading nor `DONE` forward records. Log: `/tmp/ragtruth_population_smoke_20260912_resume.log`.
2. Missing-input-manifest recovery: first safely renamed `input_manifest.json` to `input_manifest.witness_backup.json`, retaining the original file. Resume exited **0**, **24.928 s**. It rebuilt a byte-identical `input_manifest.json` (SHA256 `6531cda3eadd4a73e3ee87db8aa4d0a8150dd6f888e33afa8502133e90ff3fef`), preserved all 30 response-file hashes, and again logged no model loading or `DONE` records. The backup is retained. Log: `/tmp/ragtruth_population_smoke_20260912_manifest_resume.log`.

After both recovery checks, an independent GPU check again showed 1 MiB used, 0% utilization, and no compute process. The witness did not modify any implementation or protocol file, did not retry GPU capture, and did not start the population run. The parent was notified that the GPU was available for the authorized full run.

One local audit-script attempt initially used a guessed `saved_tokens` NPZ key and stopped with `KeyError`; the actual archive key is `token_ids`. The audit was corrected to read the real schema and completed on the existing output. This was not a capture failure or a GPU rerun.
