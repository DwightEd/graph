# Native-v3 integrity audit and result-to-claim gate — 2026-09-13

Audit backend: **local collaboration fallback, read-only**. No Codex MCP,
GPT-5.5, or other external reviewer backend was invoked or claimed. This report
combines a deterministic artifact audit with a result-to-claim assessment.

## Integrity status: PASS; claim status: negative / unsupported

The frozen run and annotation-only evaluation are internally consistent. The
result does not support claims of automatic localization, source attribution,
causal mechanism attribution, or error-continuation attribution. It supports
only the narrower operational observation that this frozen v3 protocol completed
and abstained on every evaluated word.

## Deterministic integrity checks

- The frozen input file SHA-256 equals `settings.input_sha256`; it contains 36
  unique response IDs. The pre-frozen evaluation manifest has the same input
  hash and records `annotation_read: false`.
- The full RAGTruth annotation file bytes hash to
  `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`,
  exactly matching both the pre-frozen manifest and `evaluation.json`.
  Evaluation then verifies each frozen roster row's ID, source, generator,
  split, and response-text hash against that full annotation file.
- All five stages have exactly 36 artifacts with the exact roster ID set. I
  independently recomputed each of the 180 artifacts' settings hash, row hash,
  content hash, required upstream set, and upstream file hashes: **0
  mismatches**.
- `progress.json` records 36 responses and `status: complete` at 07:52:39;
  `evaluation.json` was written at 07:53:33. The evaluator explicitly rejects
  non-complete runs, requires the frozen annotation manifest, checks raw label
  bytes before parsing, and verifies full response word offsets before joining
  labels. Its output labels the join
  `evaluation_only_after_complete`.
- Annotation labels are absent from protocol settings (`labels_used: false`)
  and from A/B/C/D/merge inputs. They are loaded only in
  `experiments/evaluate_native_audit.py`. The post-A source-reader barrier
  diagnostic is not in the frozen code manifest, has `labels_used: false` and
  `predictions_modified: false`, and is excluded from every result below.
- No metric is normalized by a model-output maximum. Source-balanced AUROC/AUPRC
  are reported alongside zero coverage and zero valid bootstrap replicates;
  their constant-score values are not usable performance evidence.

## Observed completed run

| Quantity | Recomputed value |
| --- | ---: |
| Responses / sources | 36 / 6 |
| Full response-word denominator | 4,733 |
| Annotated error words | 260 |
| A questions | 368 |
| A question states | 271 uncertain, 97 invalid_question |
| Selected native questions | 0 |
| A reader requests | 1,064 |
| B/C/D candidate entries | 0 / 0 / 0 |
| Native forward calls / tokens | 0 / 0 |
| Words scored, A/C/mechanism | 0 / 0 / 0 |
| Words abstained, A/C/mechanism | 4,733 / 4,733 / 4,733 |
| Detected words at frozen 0.8 threshold | 0 |
| Recall including abstentions | 0.0 |

The preserved full-word source denominator has 1,666 assertion, 135
nonassertion, and 2,932 `coverage_unknown` words. All three task groups remain
at zero scoring coverage. There are no mechanism labels, positions, origins, or
native certificates; all 368 claim nodes say
`not_selected_or_contrast_unavailable`.

## Claim gate

| Intended claim | Verdict | Evidence |
| --- | --- | --- |
| Automatic word-level localization/detection | **No** | 0/4,733 scored, 0/260 error-word coverage, recall 0.0. |
| Source-event / source-answer attribution | **No** | No E/C/N anchor reached; 271 uncertain and 97 invalid questions. |
| Native causal route, origin, or MLP attribution | **No** | No selected question and no B/C/D native call or certificate. |
| Conditional history or error-continuation attribution | **No** | No relation was eligible for native validation; no mechanism/edge certificate exists. |
| Frozen v3 pipeline completed conservatively | **Yes, narrow operational claim** | Five stages and evaluation completed with checksums intact; all eligible outputs abstained. |

This negative result does **not** establish that source information is absent
from the observer, that the proposed native operators fail, or that the
post-A what-if reader-barrier diagnostic predicts a repaired system. It shows
that this frozen semantic-anchor interface did not produce any eligible native
measurement on this exploratory six-source roster.

## Scope and pending operational status

The evaluation is a real-GT, post-completion join for 36 responses from six
train sources. It is exploratory, observer replay rather than an original
generator trace, has no confidence intervals from only six sources, and has no
independent node ground truth for unique lookback accuracy. It cannot justify a
general effectiveness or causal claim.

At audit time the phase artifacts and evaluation are complete, but the external
documentation/interleave wrapper (PID 147551) is still running while population
resume work is active. Wrapper finalization is therefore **pending**; this is
neither a run failure nor evidence of successful population restoration.

No GPU process was started or modified for this audit.
