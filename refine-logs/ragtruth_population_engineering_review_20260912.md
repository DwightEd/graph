# RAGTruth population mechanism engineering review — 2026-09-12

Separate engineering-review agent using
`/root/.agents/skills/code-review-and-quality/SKILL.md`. Scope:
`graph/route_graph/population_mechanism.py`, its tests,
`reanchor/src/decoding/ragtruth_population_io.py`, `ragtruth_population.py`,
`ragtruth_population_evaluate.py`, the launch script, and
`graph/docs/RAGTRUTH_POPULATION_MECHANISM_PLAN_20260912.md`.

The reviewer read implementation and local data, ran CPU checks, and wrote this
report. No GPU work, execution-code edits, or real-result edits were performed.
Temporary synthetic fixtures below are software tests, not research results.

## Verdict

**Engineering review: approve. No unresolved Required issue.**

The reviewed implementation is suitable for the authorized full-population run
after the separate runtime/resume witness. This is not scientific approval.
Independent scientific review remains unavailable: `REVIEW_UNAVAILABLE`.

## Required issues found and resolved

1. **Interrupted input-roster initialization could prevent resume.** Initially,
   `inputs.jsonl` was published before `input_manifest.json`; interruption between
   those writes made resume fail on the absent manifest. The runner now rebuilds
   expected records from the frozen dataset/tokenizer contract, compares them with
   the existing roster, and creates the missing manifest only after equality.
   Independently tested the exact missing-manifest state: recovery succeeded and
   reused an already completed fixture without model loading. A changed
   unmanifested roster was rejected.

2. **Evaluation omitted a required metric argument.** The shared
   `binary_detection_metrics` requires keyword-only `seed`. The evaluator now
   passes `settings['seed']` with `bootstrap=0`. A synthetic two-class evaluation
   independently completed and exercised this metric path successfully.

3. **Evaluation could double-count a copied output directory.** A CPU reproduction
   initially copied response directory `1` to `backup_1`; evaluation reported two
   copies of ID 1 and two responses despite progress saying one completed response.
   The final evaluator verifies the frozen input-roster hash and unique roster IDs,
   requires directory name == record ID and membership in the roster, and invokes
   `verify_response` against the frozen record. Final/static evaluation also checks
   the verified completed count against progress. The duplicate-directory witness
   now raises the intended identity error. Running partial evaluations retain
   explicit scanned IDs/counts rather than treating transient progress as a
   complete cohort.

Additional inspected improvements clear each layer's temporary V/context cache
after attention processing, reset peak CUDA allocation per response, save explicit
oversized IDs, verify execution-code hashes before final evaluation, and write the
completion marker only after evaluation succeeds. These changes preserve the
declared scientific settings and do not alter which labels enter inference.

## Token and causal scope

The runner supplies `ids[:-1]` to the model, then compares states starting at
`P-1` against observed response tokens `ids[P:]`. Consequently every response
token, including the first and last, has the correct preceding prediction state.
No input suffix is truncated to obtain a small readout. The tokenizer protocol is
explicit: BOS plus separately tokenized original prompt and original response.

Interventions begin at query `P-1`. Every condition checks exact equality of
earlier prompt states `[0, P-1)`; the final prompt state is correctly excluded
because it predicts the first response token and is intentionally intervened upon.
Sham checks the entire final-state tensor for exact equality.

Source keys lie inside the prompt mask. Remote-history keys must be response
positions (`key >= P`) and satisfy `key <= query - 16`, excluding self and recent
history. With this convention the first possible remote key is response position
P at query P+16, predicting response step 17. Earlier interventions can propagate
through later response states, so these are full-response fixed-prefix
interventions rather than isolated single-query effects. The plan states this
distinction.

The endpoint permutation is derived from a fixed seed and response ID. It does
not depend on future response text, candidate labels, or attention outcomes.

## Actual A/V/O/MLP operations

Each eager-attention hook obtains that forward's actual attention and projected
values. GQA values are repeated in the model's native head grouping. Source and
remote-history contributions use the separate head tensors in A×V products;
heads are not averaged before computing messages.

Source/history conditions subtract 10% or 100% of the selected current message
from the pre-O context. They do not redistribute other attention mass or claim
to remove all information about the source from residual/history paths.
`source_permute` rearranges only the source columns of the current layer's A and
keeps its V fixed. The current source-weight multiset is preserved; summation
roundoff is recorded as `source_mass_max_error`. Downstream layers use their
current intervened inputs, not stale captured A.

O is replayed over the original complete context shape for non-MLP conditions,
including zero-delta sham. This preserves the matrix multiplication shape needed
for exact equality. MLP attenuation replaces the actual MLP branch output with
its 0.9-scaled value at the selected queries; attention is not independently
changed for that condition. All residual and later-layer computation remains the
real model computation.

Baseline profiles contain attention mass, fp32 V/O source/history message norms,
and actual MLP-update norms. They are explicitly diagnostic summaries, not a
complete high-dimensional graph cache. The model retains only the current eager
attention matrix; temporary layer caches are cleared after use.

Final hidden states are scored with the actual LM head in fixed-size blocks.
The baseline top-two token IDs are selected from the baseline block and reused
for the corresponding altered block. Candidate selection uses no annotations.
The same block protocol is used for baseline and altered logits. Full-vocabulary
JS, observed-token log-probability change, native argmax/ties, entropy, and fixed
candidate-margin changes are measured. Identical logit rows receive exact zero JS
to avoid interpreting positive floating-point roundoff as an intervention effect.

## Independent CPU verification

- Ran `graph/tests/test_population_mechanism.py`: **2 passed in 12.76 seconds**.
  Tests cover actual model changes, exact sham, earlier prompt invariance,
  remote-history visibility, a future-token change under permutation, deterministic
  permutation identity, invalid inputs, and hook cleanup. The parent subsequently
  reported the two tests passing again after the final cache change.
- Additional CPU bf16 tiny-Llama check: full-state sham was exact; default remote
  history first became eligible at prediction step 17; earlier history profiles
  were exactly zero; 24 response readouts were covered with a non-dividing block
  size of 7; baseline top-two IDs matched between altered and sham comparisons;
  no hooks remained installed.
- Independent temporary I/O/evaluation fixtures verified seeded two-class metrics,
  duplicate-directory rejection, recovery after roster publication without its
  manifest, reuse of a completed response without model loading, and rejection
  of a modified unmanifested roster.

These CPU checks do not replace production bf16 GPU validation. The parent
reported a separate six-response/eight-condition GPU smoke succeeding; this
reviewer did not independently perform that GPU run or its resume witness.

## Real data and annotation boundaries

Independently checked every source's evidence interval against its original
prompt: all **2,965** intervals matched exactly once.

| Task | Sources | Responses |
| --- | ---: | ---: |
| QA | 989 | 5,934 |
| Summary | 943 | 5,658 |
| Data2txt | 1,033 | 6,198 |
| Total | 2,965 | 17,790 |

Every source has six response records, one per original generator. The corpus has
15,090 train and 2,700 test responses. No response text is empty. In a separate
evaluation-side schema check, all **14,289** annotation spans had valid character
boundaries and exactly matched their provided text slices. This verifies indexing
consistency, not the factual correctness of the annotations.

The inference adapter copies identity, original source/prompt/response text,
task, generator, official split, offsets, and masks. It does not select samples
using quality flags, hallucination presence, or span positions. Smoke selection
uses task, deterministic metadata ordering, and input length. Source-mask tokens
overlap the uniquely matched evidence interval and exclude special-token IDs.

Only evaluation joins annotations. It checks frozen dataset identity, response
text hash, source ID, generator, split, complete token offsets, and metric
coverage/finiteness. Character-span overlap is converted to token labels using
the observer tokenizer's recorded offsets. Output groups separate task,
original generator, and official split. Source-balanced descriptive rankings use
fixed score directions, without fitting a detector or choosing directions from
outcomes. Conditional source means use grouped sums/counts. The explicitly named
`through_first_error` subset is an evaluation restriction, not an inference input.

The local observer is Llama-3.1-8B; the six recorded generators are different
models. The artifacts correctly call this observer replay and retain
observer/observed-token mismatch. These internal interventions therefore do not
establish what happened inside the original generators. Sensitivity, diagnostic
rankings, source attention, or branch norms do not by themselves establish
hallucination probability, correct ownership, graph-model necessity, or a
validated detector.

## Persistence and execution limits

The reviewed runner uses a single-process file lock, a frozen code/model-stat/
dataset/settings contract, a hashed input roster, and per-response atomic
directory publication. Completed responses are verified before reuse. Failures
and partial directories are preserved; OOMs are recorded while remaining work
continues, and retries require the explicit retry option. Inputs exceeding the
declared limit are recorded rather than silently truncated. A run with failures
cannot claim full completion. Runtime code and cohort integrity failures stop
execution rather than mixing incompatible protocols.

This review approves the engineering implementation within those limits. The
full-population result, actual runtime coverage, GPU resource behavior, and any
scientific interpretation remain separate matters to be established from the
run and its independent witness.
