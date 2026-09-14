# P6 new-source confirmation — prepared, NOT yet launched

This document is prepared before the P6 development result. It does not assert
that the candidate passed. The launcher fails closed until a separate freeze
record exists and every scorer/evaluator/input/protocol hash matches that record.

## Scope and immutable computation

Fixed64 responses from32 official-test sources:12 QA,12 Summary,8 Data2txt sources,
two Llama2 generators each. Roster protocol is P6_CONFIRMATION_ROSTER_PROTOCOL.md.
Input SHA256:bc9fabcad838541e8f8892d436a181746c7b200363b9160b73b50ce2291d5101.
Source IDs and exact source contents are disjoint from R04. Prior population
observer measurements exist: this is not globally pristine, and model pretraining
contamination is unknown. Do not use old R04 validation as an untouched endpoint.

P6's full scoring/model-execution AST is identical to the development code:
167d7199508cc2e55dad878e1e2f173b562ea95692ab835175fd58ede03b5b38.
The adapter changes only the pinned input, allowed phase, expected denominator,
and provenance metadata. test_confirmation_protocol.py verifies all outer scoring
functions/constants and the entire model execution try block. FP32 per-layer
arithmetic, unchanged BF16 stored weights, math SDPA, no TF32; batch4. Draft
generation is greedy,384 new tokens maximum; truncated drafts remain visible and
explicitly fallible. Raw B-minus-A logits; same word-to-original-token mapping.
Full SOURCE and ANSWER remain visible. Retrospective external zero-additional-
training semantic verifier, NOT online detection/original-generator causality.

All64 form one indivisible evaluation endpoint. No partial-cohort label join,
sample filtering, best-task selection, score sign flip or confirmation-driven
tuning. AUROC must be strictly above0.8 on all original tokens. Report AP,
source-cluster95% interval, source-balanced and within-answer metrics, all tasks,
and preexisting entropy/control scores. An interval reaching0.8 is not a passing
point estimate. A pass is bounded to this cohort, not a universal guarantee;
source uncertainty and previous sequential selection remain disclosed.

## Actual documented command (fresh output, only after freeze)

Reuse Python B/conda_envs/research/bin/python and research@03909e02, no installs.
One RTX4090; no other GPU job may run. Model weights B/models/Qwen3-8B are fixed.
B=/share/home/tm902089733300000/a903202310/lys. The exact executable command is:

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_local_grounding_confirmation.sh /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p6_review_confirmation_20260914_v1
```

The fresh document witness must execute that command once, with stdout/stderr in
graph/runs/p6_review_confirmation_20260914_v1.log and subprocess exit code saved.
No labels or scores from another candidate are read by the witness. It may inspect
engineering completion/hashes/alignment/cache/draft truncation, not performance.
Every original response must be represented. A failure is retained, never silently
resumed or replaced. Report actual elapsed/peak bytes from the manifest; do not
estimate completion. Runtime may be tens of minutes; monitor progress without
starting a competing GPU process. Do not rerun a second pilot on these64 examples.

## After complete hash-frozen predictions, separate CPU evaluation

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.confirmation_reference_export --inputs outputs/p6_confirmation_roster_20260914_v1/inputs.jsonl --population ../reanchor/outputs/ragtruth_population_20260912 --phase validation --output outputs/p6_population_reference_confirmation_20260914_v1
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.confirmation_assess --predictions outputs/p6_review_confirmation_20260914_v1 --reference-predictions outputs/p6_population_reference_confirmation_20260914_v1 --annotations /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl --primary reviewed_source_risk --output outputs/p6_review_confirmation_20260914_v1/evaluation.json
```

Frozen predictions must exist before reference extraction and annotation join.
The evaluator refuses a partial roster or wrong input SHA before opening labels.
Failed or successful confirmation is retained and reported; further iterations
must not reuse this cohort as if it remained an untouched holdout.
