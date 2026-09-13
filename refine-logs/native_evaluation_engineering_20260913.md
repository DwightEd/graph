# Native-audit evaluation engineering review — 2026-09-13

Reviewed `experiments/evaluate_native_audit.py` as an evaluation-only consumer
of a completed frozen A–D run. This review did not read real RAGTruth labels or
run any GPU work.

## Result

**Critical 0, Required 4, Optional 1.** The evaluator correctly waits for the
method `progress.json` to report `complete`, checks the frozen input roster and
recursively validated merge/A/D artifacts before joining annotations, retains
all final response words, and explicitly states that lookback-node accuracy is
not identifiable without independent node ground truth. The remaining issues
can misbind labels/scores or make a reported operating point non-frozen.

### Critical

None.

### Required

1. **The reported operating threshold is a literal, not the frozen run
   protocol** — `experiments/evaluate_native_audit.py:17-57`. Detection uses
   `scores >= 0.8` directly, never reads
   `settings["protocol"]["semantic_threshold"]`, and does not serialize the
   applied value with the metrics. A complete run with a different frozen
   protocol would be evaluated at a silently substituted threshold. Read and
   validate the threshold from the saved settings, pass it into `metrics()`,
   and write it into the report.

2. **Semantic-A scores are paired to labels by array position without checking
   their word offsets** — `experiments/evaluate_native_audit.py:108-132`.
   The code verifies `result["words"]` against every response word, but only
   checks the shape of `result["semantic_A_words"]`. A same-length reordered
   or stale A list silently associates scores/abstentions with different
   annotation spans. Require its `(start, end)` sequence to equal the full
   response word sequence, and verify its coverage-state/abstention fields
   before constructing score arrays.

3. **The annotation artifact is neither atomically hash-bound nor an exact
   roster join** — `experiments/evaluate_native_audit.py:60-84, 97-105`.
   It hashes `labels_path`, then opens it again for parsing, so the used bytes
   can differ from the checked hash. It also rejects duplicate IDs but does not
   require `set(annotation_ids) == set(roster_ids)`, silently ignoring extra
   labels and failing missing labels only later through `KeyError`. Read and
   hash one immutable byte snapshot; require exact bidirectional roster
   equality before joining. The `--expected-labels-sha256` value is only a
   caller-provided string, so also record/verify a pre-frozen annotation
   manifest or other provenance object that supplies that expected hash.

4. **The new evaluator fails repository lint** —
   `experiments/evaluate_native_audit.py:3-14`. Ruff reports `I001` for import
   ordering. Apply the formatter-approved import separation and rerun lint.

### Optional

1. Ranking AUROC/AUPRC uses equal total weight per source, while coverage and
   fixed-threshold precision/recall are deliberately raw-word counts. This is
   visible from their field names and the report does not call those latter
   values source-balanced. Adding source-weighted fixed-threshold counterparts
   or per-source summaries would make comparison with the source-balanced
   ranking easier, especially once more than six sources are available.

## Confirmed behavior

- Labels are opened only inside `evaluate()` after the completed-run gate; A–D
  inputs, candidate selection, controls, and thresholds are not passed labels.
- Every final response word is required to have exactly the whitespace-word
  offsets of the frozen response. Scores must be finite and equal in count to
  this denominator. Metrics retain abstained words in `all_words`,
  `annotated_error_words`, and recall; coverage and abstention counts are
  reported separately.
- AUROC/AUPRC uses `binary_detection_metrics(..., source_balanced=True)` and
  the output states that confidence intervals are not estimated from six
  sources.
- The report preserves `unique_lookback_accuracy` as not identifiable and
  labels its scientific review as not performed. It does not fabricate node or
  causal-edge ground truth from mechanism certificates.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_evaluate_native_audit_review.py \\
  tests/test_audit_pipeline_review.py
5 passed in 16.12s

/tmp/research_lint_20260912/bin/ruff check \\
  experiments/evaluate_native_audit.py tests/test_evaluate_native_audit_review.py
I001: import block is un-sorted or un-formatted
```

Added CPU-only `tests/test_evaluate_native_audit_review.py`, which verifies
that an abstained positive remains in the all-word denominator and that recall
includes abstentions. No implementation or GPU work was performed.

## Final re-review: frozen annotation manifest — 2026-09-13

**Result: Critical 0, Required 0. There are no blockers.**

The four Required findings are resolved.

- `metrics()` now accepts the threshold explicitly; evaluation passes
  `settings["protocol"]["semantic_threshold"]`, and every metric record
  serializes `frozen_threshold`.
- Both final words and `semantic_A_words` must have the exact complete
  whitespace-word offset sequence before labels are projected. Thus all three
  score streams share the same full-word denominator.
- The evaluator reads label bytes once, hashes that exact buffer, and parses
  only that buffer. A frozen `evaluation_manifest` supplies the label hash and
  binds it to the frozen input roster and absolute label path. The runner
  freezes the manifest path/hash in settings and rechecks it across phases;
  the scheduler propagates that same path to its prepare/all invocations.
- The full RAGTruth annotation file is correctly allowed to contain IDs beyond
  the 36-response roster. It rejects duplicate global annotation IDs and every
  frozen roster ID must be present and match source, generator, official split,
  and response digest. Requiring a bidirectional equality would incorrectly
  demand a copied, truncated label file.
- Ruff import ordering is clean.

The existing source-balanced AUROC/AUPRC remains separate from raw-word
coverage and fixed-threshold count fields, and the evaluator still states that
node-level lookback accuracy is not identifiable without external node labels.

### Final verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_evaluate_native_audit_review.py \\
  tests/test_audit_pipeline_review.py tests/test_native_scheduler.py
8 passed in 17.02s

/tmp/research_lint_20260912/bin/ruff check [evaluator, runner, scheduler, regressions]
All checks passed!
```

No labels were read and no GPU or implementation work was performed during
this re-review.
