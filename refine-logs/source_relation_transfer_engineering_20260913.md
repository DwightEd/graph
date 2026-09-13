# SourceRel natural transfer engineering review

2026-09-13. Independent bounded `code-review-and-quality` review of `next_iteration/source_relation_transfer.py`. Scope was limited to the new M2 transfer file plus targeted CPU review tests; frozen source data/features/train modules were read only for interface contracts. No GPU, labels, downloads, or implementation changes were used.

## R1 targeted re-review

Critical: 0.

Required: 0.

The previous Required finding is closed. `prepare()` no longer stores the hash partition under the misleading `source_reconstruction_internal_split` name for all natural transfer rows. It now separates:

- `source_hash_partition`: deterministic source reconstruction hash partition, not a claim of actual training membership.
- `source_reconstruction_training_eligible`: `True` only when the natural row is official train.
- `source_reconstruction_training_split`: the source hash split for official train rows, otherwise `not_in_source_reconstruction_training`.

`predict()` now writes `actual_training_source_membership` by joining the exact frozen training settings `internal_split` by `source_id`, with missing sources reported as `not_in_source_reconstruction_training`. That is the right authority for whether the owner head actually trained on a source. This closes the leakage/provenance ambiguity without changing transfer feature texts, candidate pools, checkpoint loading, scores, or labels.

## Checks that remain accepted

- `prepare()` uses `route_graph.audit_runner.rows`, so it inherits the exact 14-field schema check, duplicate response-ID rejection, digit artifact IDs, and response SHA validation.
- The transfer query masks only the selected current slot span. Earlier same-surface occurrences stay visible by design and are not treated as aliases or positives.
- Natural transfer stores `positive_ids: None`, `semantic_role_status: unverified`, `query_supervision: None`, `labels_read: False`, `training_on_responses: False`, and `native_forwards: 0`. I found no response-derived positive owner labels, SCNI labels, or native certificates in this file.
- Candidate pools are same broad type and source reuse is allowed. Empty pools, out-of-domain tasks, untyped slots, and overlength query/pool documents remain in the denominator through explicit unresolved statuses.
- `predict()` checks the natural feature parent protocol, original input hash, feature manifest hash, training manifest hash, source-only checkpoint completeness, `sanity_only == False`, checkpoint/settings digest, checkpoint `TYPES`, and source/natural encoder model-file identity.
- The result-denominator guard compares both slot-ID set and result length against the response graph slots before writing each response artifact.

## Targeted tests

Updated `tests/test_source_relation_transfer_review.py` so the metadata regression matches the R1 contract and includes an official test Data2txt row. The test now verifies that official train rows are eligible and keep their hash split as the training split, while official test rows keep a hash partition but get `source_reconstruction_training_eligible: False` and `source_reconstruction_training_split: not_in_source_reconstruction_training`.

Command run:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_source_relation_transfer_review.py -q
```

Result: `2 passed in 13.22s`.

## Remaining boundary

This module is still only candidate supply. Its outputs must not be reported as natural owner ground truth, factual SCNI, or a native mechanism certificate. The current artifacts preserve that boundary.
