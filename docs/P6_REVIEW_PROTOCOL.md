# P6: explicit evidence audit before localization

2026-09-14. P5 frozen development AUROC0.809356 failed held-source validation
AUROC0.747436. Do not reclassify the failed validation as a success. P6 is a
separate exploratory method; final confirmation must use a new fixed source holdout.
Current development stays the original R04 development32. P5 validation errors
will not be inspected individually for this revision; its aggregate failure is known.

Hypothesis: independent word questions inadequately maintain claim-level evidence
bindings. First let the SAME frozen Qwen3 write a source-grounded, fallible whole-answer
audit (max384 generated tokens, no thinking mode); supply this explicitly marked
draft to each otherwise identical P5 word question. SOURCE remains the only evidence;
the draft is never ground truth. Store the entire draft, generated IDs, length and
truncation status; partial drafts retained and marked, no answer rejection or filtering.
Primary score remains raw zB-zA, no blend, trained coefficients, sign-flip or normalization.
Every original token retained; same word mapping and complete-response retrospective scope.
This tests explicit claim/evidence organization, not original-generator mechanism or graph necessity.

P5 without draft is the exact semantic ablation. Use paired all-token/AP/source/within-answer
evaluation and inspect all development cases. No promise of novelty from prompting.
If not improved, preserve failure and change the representation, not the held-out threshold.
New held-source roster must be label-free and frozen before its scoring/label join.

Reuse stable per-layer FP32 arithmetic, unchanged BF16 stored weights, mathSDPA, TF32off,
cache guard0.005 and checked whole-answer fallback. No package/environment changes.
Doc-execution pilot is the same two predetermined R04 examples, no natural labels:

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_local_grounding_review.sh pilot /share/home/tm902089733300000/a903202310/lys/research/graph/outputs/p6_review_pilot_20260914_v1
```

After engineering check, development output is `outputs/p6_review_development_20260914_v1`.
Do NOT launch validation using the original R04 inputs; a new confirmation roster is required.
