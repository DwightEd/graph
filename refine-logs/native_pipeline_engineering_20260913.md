# Native A–D pipeline engineering review — 2026-09-13

Reviewed `route_graph/audit_prepare.py`, `audit_semantics.py`,
`audit_output.py`, `audit_runner.py`, `audit_artifacts.py`, and
`audit_protocol.py`, together with the A/C interfaces in
`evidence_anchor.py`, `frozen_reader.py`, and `audit_alignment.py`. This is an
orchestration review only. It does not re-review the D measurement operators.

## Result

**Critical 0, Required 6, Optional 1.** The semantic inputs are label-blind,
the N candidate labels use every frozen source text unit, and the selected
recovery control preserves E rather than creating an error-continuation claim.
The remaining Required issues concern the immutable/resumable execution
contract and final coverage accounting.

### Critical

None.

### Required

1. **`all` and a one-phase invocation can run concurrently** —
   `route_graph/audit_runner.py:348-350`. `--stage all` locks `.driver.lock`,
   whereas A/B/C/D/merge each lock `.phase.lock`. Thus an operator can start
   `all` and, for example, `--stage B`; both locks succeed, contrary to the
   one-model/one-GPU phase protocol. It can overlap model work and race an
   immutable artifact publication. Use one run-wide lock for every invocation,
   or acquire a common lock plus a phase lock in a fixed order.

2. **Resume accepts an incomplete or substituted phase chain** —
   `route_graph/audit_runner.py:122-156`. `read_artifact()` verifies only the
   entries actually present in `saved["upstream"]`; it never requires B to
   name A, C to name A+B, D to name A+B+C, or merge to name all four. A
   self-consistent B file with `{}` as upstream therefore passes and can feed
   later phases. Validate the exact dependency set from `publish()` and
   recursively validate each required artifact with the same row and settings
   hash before reuse.

3. **D does not bind B's frozen candidate/result to A's exact B/A event** —
   `route_graph/audit_phase_native.py:28-41, 81-103`. B serializes its
   `contrast`, but D rebuilds from `question["events"][0]` and only compares
   baseline token log-probabilities. It never asserts that
   `proposal["contrast"]` equals the selected A event, nor that the frozen
   candidate record was made for that event. This breaks the stated
   cross-stage B/A/candidate provenance guarantee after a stale or misplaced
   cache entry. Compare the complete serialized contrast (prefix,
   continuations, prompt length) and bind the frozen selection to its digest
   before any B record is reused.

4. **Raw donor artifacts are not revalidated when D or merge resumes** —
   `route_graph/audit_artifacts.py:66-77` and
   `route_graph/audit_runner.py:122-132`. The D result keeps only paths and an
   NPZ hash. `donor_writer()` checks bytes while writing, but subsequent stage
   reads check the D JSON checksum, not the referenced manifest/NPZ. The final
   graph can consequently cite changed or missing raw donor values while its
   stage artifacts all validate. Verify each referenced donor manifest and
   NPZ hash on D/merge resume, and ensure the manifest's layer digests match
   the raw values.

5. **The final word output loses A's explicit coverage states** —
   `route_graph/audit_output.py:117-142, 168-182`. A records every word as
   `assertion`, `nonassertion`, or `coverage_unknown`; merge reconstructs a
   new `words` list without that field or state counts. Consumers cannot tell
   an intentionally nonassertive word from an assertion that extraction failed:
   both become a score of 0.5 with abstention. Copy `coverage_state` into each
   final word and report counts for all three states alongside scored-word
   counts.

6. **Model/tokenizer bytes are not rehashed on resume** —
   `route_graph/audit_runner.py:76-92`. The initial manifest records SHA-256
   per relevant model file, but restart validation compares only each saved
   file's size and `mtime_ns`. A changed file with restored metadata therefore
   reuses semantic/native artifacts produced by different weights or tokenizer
   rules. Compare a newly computed manifest, including SHA-256, with the saved
   one before accepting a resumed run.

### Optional

1. **Make the no-gold contract syntactically closed** —
   `route_graph/audit_runner.py:96-119`. Current A/C calls pass only task,
   response, source, and questions, so I found no active label leak. The input
   check is nevertheless a short blacklist (`labels`, `hallucination_labels`,
   `quality`, `incorrect`) rather than the strict response schema used by
   `route_graph.data.read_responses()`. Allowlisting the exact method fields,
   or invoking that reader, would make the no-gold assurance mechanically
   robust against dataset-specific annotation names.

## Confirmed behavior

- A's extractor receives `task` and `response`; source QA is built through the
  blind `source_qa_request()` interface and does not receive the claimed
  response answer or natural labels.
- For N, C labels every frozen source text node and
  `null_source_unit_check` requires each one to return `not_stated`.
  Malformed/missing labels default to `uncertain`; full source is supplied to
  each candidate-label request, so a context-limit failure does not become a
  NULL result.
- A recovery is constructed only for current `supported` plus a prior C-score
  question, after the separate all-conditions-equivalent check and a
  single-slot grammar check. It reserves one of four native slots, leaving at
  most three risk claims plus one recovery control.
- `merge_response()` requires both prior and current claims to be unsupported
  before emitting `error_history_continuation`. A supported correction with a
  measured positive history effect is kept as
  `supported_recovery_dependency`, which is the required control behavior.
- Settings freeze the full input hash, code hashes, model manifests, runtime
  versions, dtype, and attention implementation. Input and code are rehashed
  on resume; Required item 6 closes the remaining model-byte gap.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_audit_pipeline_review.py \\
  tests/test_evidence_anchor_review.py tests/test_audit_alignment_review.py \\
  tests/test_causal_groups.py
9 passed in 16.66s

/tmp/research_lint_20260912/bin/ruff check [reviewed modules and dependencies]
All checks passed!

python -m compileall -q [reviewed modules]
compileall: OK
```

Added CPU-only `tests/test_audit_pipeline_review.py`: it verifies the N
denominator fails when one frozen source unit is uncertain, and verifies a
supported recovery control is not labeled `error_history_continuation`. No GPU
work or implementation changes were made.

## Re-review: immutable phase chain and mapped recovery — 2026-09-13

**Result: Critical 0, Required 0, Optional 1. There are no blockers in the
reviewed scope.**

All six Required findings above are resolved.

- Every invocation now holds the same `.phase.lock`. `--stage all` executes
  its phases in one process; the scheduler passes the already-locked identical
  open-file description through `--inherited-lock-fd`, and the runner verifies
  its device/inode before accepting it. This prevents both all-vs-phase and
  scheduler handoff races.
- `read_artifact()` now requires the exact fixed upstream key set, checks each
  recorded hash, and recursively validates every required dependency before a
  stage may resume.
- B records its canonical full contrast and question ID. D rejects a proposal
  unless both are identical to A's selected event, and rejects C unless its
  question ID and frozen-pool digest match B. Thus cached B candidates and C
  labels cannot be applied to another event or pool.
- D resume verifies each top-level and alternate-template donor reference is
  under the current run's donor root, validates manifest and NPZ hashes, and
  recomputes every stored raw layer byte digest.
- Merge carries A's `coverage_state` into every final word and publishes the
  `assertion`/`nonassertion`/`coverage_unknown` counts.
- Resume recomputes a complete file manifest, including SHA-256, for both
  model/tokenizer directories. It no longer accepts same-size,
  restored-mtime byte changes.
- C now binds the prior question's ID and answer slot without passing its
  labels. D maps a linked history origin to that exact prior answer slot; merge
  requires both that mapping and a validated slot relation for either strong
  error continuation or supported recovery. The recovery stays an E/control,
  never an error-continuation label.

The Optional strict input allowlist remains a hardening opportunity; no active
gold-label path was found.

### Re-review verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_audit_pipeline_review.py \\
  tests/test_evidence_anchor_review.py tests/test_audit_alignment_review.py \\
  tests/test_causal_groups.py
12 passed in 12.13s

/tmp/research_lint_20260912/bin/ruff check [reviewed modules and regression]
All checks passed!
```

The updated CPU regression fixture now includes the prior C score, linked
prior-question ID and answer span, mapped history-origin tokens, and complete
word coverage. It additionally verifies recursive-upstream rejection and that
restored model metadata cannot conceal changed model bytes. No GPU work or
implementation changes were made in this re-review.
