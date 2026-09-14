# P5: complete-context localized source grounding

Frozen before new scoring on 2026-09-14, after observing P1-P4 development failures.
This is a zero-additional-training frozen Qwen3-8B external judge, not a claim of
unsupervised internal-state discovery, novel graph topology, or original-generator causality.
Full answer is visible: explicitly RETROSPECTIVE, never an online onset detector.

For every whitespace-delimited word (punctuation attached), show the complete
source and answer, task, exact target interval and marked local sentence. Judge
membership in the smallest unsupported factual phrase, including incorrect owner,
time, relation or condition binding. Risk is z(B)-z(A), no label fitting or sign flip.
Each original observer token receives max risk over intersecting words; whitespace-only
tokens attach to the next word, trailing whitespace to the last. Every token retained.
The source/response itself is data, not instructions. No RAGTruth annotation input.

Use only R04 development32 to assess candidate. Pilot uses fixed17019,17020.
Validation32 remains unopened until a candidate is frozen after development.
Existing R04 sources were previously observed in population research; validation
is held-source relative to this development, not pristine official test.
No fixed hyperparameter sweep. Compare frozen P1/P4 and native entropy on IDENTICAL
all-token denominator; report AP, per-task and within-answer results, source bootstrap.
Gate: actual all-token AUROC >0.8 on frozen validation, not onset/task-only >0.8.
Failure is retained; next revision must be separately named and motivated by errors.

Execution revision3: the first16/32 v2 development predictions were stopped and
quarantined from evaluation before annotation join. The witness found single versus
batched BF16 full-prefill risk differed4.75. Both default and math-SDPA BF16 probes
reproduced large differences; switching just attention backend did not solve it.
FP32 linear/activation/KV computation with unchanged BF16 stored weights reduced
single/batch/cache A/B differences below0.0004 on the same fixed first4 queries.
V3 therefore uses layerwise upcasting (no resident32GB FP32 whole-model copy), math
SDPA, TF32 disabled. Stronger fixed guard0.005 for raw logits AND risk; any cache
failure falls back for entire answer, and fallback is itself checked against single
full-prefill, aborting if still above0.005. No natural-label feedback was used.

V1 pilot failed before completing any natural response: canary cached/full logits
differed by0.75 (risk log-odds differed by0.25), above the predeclared raw-logit guard.
V1 output and executed code are preserved. V2 changes only the failure policy:
keep the0.5 guard, but recompute ALL queries for that answer by full prefill when
it fails, including the first batch. No sample omission, relaxed threshold or prompt change.

Common-prefix KV reuse is execution-only. Independent copied caches per suffix batch;
right padding, explicit absolute positions and attention masks. Full-prefill control
on first query of EVERY answer, saved A/B logits; v3 full-prefill fallback above0.005 FP32 logit/odds delta.
Eight binding/support canaries test reader behavior, not natural-data ground truth.
No truncation, random rejection, retry, environment install or repository reset/pull.

Existing environment research@03909e02. Complete documented pilot command:

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_local_grounding.sh pilot /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p5_local_pilot_20260914_v3_fp32
```

After pilot engineering verification, same command with `development` and new
output `.../outputs/p5_local_development_20260914_v3_fp32`. Predictions freeze before
separate `next_iteration.grounding_contrast_evaluate` annotation join.
Do not use Llama token IDs as Qwen token IDs; Qwen only tokenizes its own judge prompts.
Original Llama token IDs and character offsets are carried solely for metric alignment.
