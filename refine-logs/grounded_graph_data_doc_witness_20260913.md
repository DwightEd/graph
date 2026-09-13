# GroundedGraph source-only data: independent document execution witness

Status: CPU preparation exited 0; synthesis was deliberately stopped after a source-occurrence training-data failure was identified. The synthesis shell session exited 130 with `KeyboardInterrupt`. This is an incomplete, negative data-design run, not a passing full-run witness. No adapter training or natural-response evaluation was performed by this witness.

## Authorization and execution scope

Read `docs/GROUNDED_GRAPH_DATA_RUN_20260913.md`, `.aris/compute/local.md`, the closed data engineering review, `/root/.agents/skills/run-experiment/SKILL.md`, and its compute environment contract. The review's R1 closure is Critical 0 / Required 0 with six targeted CPU tests passing. The environment's canonical `.aris/compute/env-spec.json` SHA prefix is `03909e02`, matching the ledger. Existing Python/packages/Qwen3-8B/RTX 4090 are reused; no installs, rebuilds, implementation edits, paused reference run, or retries.

The output directory did not exist before preparation. Both documented Python invocations were executed once, with unchanged arguments and environment assignments, inside a shell with `pipefail` and a `tee` log wrapper. The documented GPU availability command was executed between prepare and synthesis.

## CPU preparation

- Session `50405`, Python PID `174713`, exit 0.
- Log: `refine-logs/grounded_graph_data_prepare_20260913.log`.
- Output: `outputs/grounded_graph_data_v1_20260913`.
- 240 sources: QA, Summary, and Data2txt each contribute 64 training / 16 validation sources.
- 192 training / 48 validation raw source-text SHAs, intersection 0. Recomputed each request's source SHA and the frozen split rule independently.
- 1,920 selected targets; 0 unavailable requests; model forwards 0.
- Input token minimum / median / maximum: 828 / 1,231.5 / 2,958; total 309,114. All requests satisfy the 16,384-token combined context limit with the 4,096-token generation allowance.
- Independently verified all 242 prepared artifact hashes and the settings SHA before GPU launch. Settings bind 50 code files and 240 selected inventory artifacts.
- Independently matched all 240 selected original row IDs to the input roster: all `official_split == train`, and source ID, task, prompt, prompt length, and prompt-only token IDs agree. Response content and hallucination-label fields were not accessed.

Frozen identities:

| Artifact | SHA-256 |
| --- | --- |
| Run settings | `e6539da0a60e5d76d07de5ed22f7335e55239d21f8e30217266e3bc2b099e280` |
| Input roster | `be388df51e83f60beafbcb4075bde85da11a40f0941279e715f0f334e28d67bb` |
| Inventory manifest | `b4766185732788d9c96452b185e12b952a70f45dd4e95a36378b7c5a56573b09` |
| Inventory settings | `b7a5579a668d78a2dfc29bf17016fafb4a17f07185422238479dbce596a1e958` |

## GPU synthesis launch

Preflight: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader` returned `0, 1 MiB, 24564 MiB`. Hardware was independently observed as NVIDIA GeForce RTX 4090.

- Background tool session `34328`, Python PID `175021`, shell PID `175018`.
- Log: `refine-logs/grounded_graph_data_synthesis_20260913.log`.
- Protocol: thinking=true, batch 2, temperature 0.6, top_p 0.95, top_k 20, maximum 4,096 new tokens, fixed batch seeds, local Qwen3-8B, bf16/SDPA.
- Actual GPU allocation observed: 17,152 MiB / 24,564 MiB and 91% utilization while Python PID 175021 was running. The GPU process query reported a different host PID (`4178099`, process name unavailable); do not conflate that number with the container's Python PID.
- Batch 1: 2 / 240 sources, 2,355 actual batched forwards, 80.070 seconds; 6 / 16 mechanically usable templates, 10 rejected templates, two complete generations. Linear remaining ETA was about 159 minutes.
- Batch 2: 4 / 240 sources, 4,463 actual batched forwards, 149.047 seconds; 7 / 32 mechanically usable templates, 25 rejected templates, four complete generations. Linear remaining ETA was about 147 minutes. These are observed throughput estimates, not promised completion times.
- Batch 3 / last durable progress: 6 / 240 sources, 7,298 actual batched forwards, 244.952 seconds; 19 / 48 mechanically usable templates, 29 rejected templates, six complete generations. All six are QA/train; Summary, Data2txt, and validation have no completed synthesis outputs.

## Deliberate stop and resource release

After the root independently identified source-occurrence and relation failures, it explicitly instructed this witness to stop only synthesis PID 175021. At 2026-09-13 14:39:17 +08:00, the witness reread `/proc/175021/cmdline`, required exact agreement with the documented synthesis command, and sent SIGINT to that PID only. Session 34328 then exited 130 with `KeyboardInterrupt` inside Qwen `model.generate` / rotary embedding computation. There was no retry, reinterpretation of compilation, or deletion of partial output.

The interrupted fourth batch has no durable results and its additional forward count is unknown. The recorded 7,298 forwards are the exact count for the three completed batches and only a lower bound on all forwards performed before interruption. Do not sum `batch_actual_forwards` across all six records because both records in a batch repeat the same batch count. The run prepared 1,920 target slots, completed 48 target slots, and left 1,872 uncompleted; the 19 accepted examples are not a full-run success rate.

After exit, neither Python PID 175021 nor shell PID 175018 remained. GPU 0 returned to 1 MiB / 24,564 MiB, 0% utilization, with no compute process listed. The witness independently acquired and released the shared scheduler lock nonblockingly, confirming it was no longer held. No unrelated process was signaled.

The partial directory intentionally has no final `summary.json` or final result `manifest.json`. The last `progress.json` still says `running` because interruption bypassed normal completion; this stale marker must not be treated as proof of a live process.

## Independent post-stop artifact audit

CPU audit session 36131 exited 0 at 2026-09-13 14:41:12 +08:00. Machine-readable evidence, including the six partial-result hashes and all 19 accepted compiled examples, is retained in `refine-logs/grounded_graph_data_partial_audit_20260913.json`. This sidecar is an independent partial-output snapshot, not the runner's missing completion manifest.

- All 242 prepared artifact hashes still match; settings and input roster retain the exact SHA values above.
- All 50 live code files and their 50 executed-code snapshots still match the prepared settings.
- Inventory manifest/settings SHA and settings-object digest match; the upstream input SHA and no-labels inventory protocol remain bound. All 240 selected inventory artifact hashes match both the run settings and parent inventory manifest.
- All 16 local model files match their saved names, sizes, modification times, and SHA-256 hashes: 16,397,461,896 bytes verified.
- Every request's source ID/task/text and inventory SHA agree with its selected source. All 1,920 selected targets have existing inventory owner/member IDs; member raw spans and displayed payloads agree with their referenced source nodes.
- Prepared target census: QA 640 text occurrences; Summary 640 text occurrences; Data2txt 493 literal payloads, 126 text occurrences, and 21 explicit identical-weekday bundles. The 21 prepared bundles were never synthesized in this stopped run.
- Each of the six partial results binds the request hash, settings hash, and execution hash. All compiled examples preserve their exact target value, target span, source owner/member IDs, and source spans; supervision remains `weak_source_template; semantic_relation_unverified`.
- Recomputed counts exactly match durable progress: 48 completed targets, 19 mechanically usable, 29 rejected. Rejection reasons are 11 exact-value repetitions outside the placeholder and 18 missing/multiple placeholders. All six generation statuses are `complete`; there are no completed-result truncations or parse failures.
- Deduplicating the repeated batch metadata gives exactly three completed batches and 7,298 forwards. Interrupted-batch forwards and final-run forwards remain explicitly unknown.

Prepared-manifest SHA: `6568598fdc16073e60cddedfe5748646d6816cc3a3dec0d53c8d34bda05204b5`. Execution SHA: `4fc7ebbeca65bd7d56e0da1dbe81ef5ce62d79feccf0d6adbf47c603af542e13`. Log hashes and each partial-result hash are in the audit sidecar.

There was no CLI/environment divergence before execution: the documented CPU prepare and GPU launch worked in the reused environment. The required full synthesis was intentionally abandoned for the substantive semantic failure below, so this does not certify a successful full document execution or useful training data.

## Interpretation boundary

The source provides exact payloads and source-occurrence owner pointers. Qwen generates natural templates; the program inserts the exact payload. Mechanical template acceptance cannot verify the relation, entity, event, negation, condition, or seven-day scope. These outputs remain weak source-only reconstruction supervision, not hallucination labels, semantic ground truth, detector effectiveness, owner accuracy, or causal-mechanism evidence. Truncations, malformed JSON, unavailable requests, abstentions, and rejected templates remain part of the selected-target denominator.

Observed limits are retained: source 15496 supplied eight fluent sentences without `<VALUE>` and all were rejected; source 12457 includes accepted templates beginning `The <VALUE> ...`, where the prefix alone does not establish the requested entity/property context. Thus complete generation does not establish mechanical usability, and mechanical usability does not establish semantic or pre-value-context fidelity.

Independent inspection confirmed two concrete accepted owner mismatches:

- Source 14308, target `t4`, owner span `[762,766]`: the exact occurrence `Toss` is in passage 3, step 5. The compiled example is `In passage 1, step 4, the fries are tossed to ensure even coating. Toss`. The pointer preserves one occurrence while the sentence asserts a different passage/step; the appended payload is also not integrated as a grammatical factual value.
- Source 12457, target `t0`, owner span `[710,717]`: `passage` introduces passage 2, about autism. The accepted sentence is `The passage discusses the survival time with ALS.`, a proposition from passage 1. Exact string insertion did not preserve occurrence ownership.

These counterexamples justify rejecting this weak-template training path; they do not require natural-response gold labels. No examples were selectively removed or upgraded to semantic labels. All six partial source results and all 240 prepared requests remain intact.
