# SourceRel-Mini feature inventory engineering review — 2026-09-13

Scope: bounded `code-review-and-quality` review of `next_iteration/source_relation_features.py` against `refine-logs/SOURCEREL_EXPERIMENT_PLAN.md`. I reviewed only this module and wrote CPU review tests. I did not edit implementation, start GPU work, read labels, download models, or review the trainer.

## Verdict

Critical: 0.

Required: 1 before GPU feature encoding or trainer consumption of this feature inventory.

## Required finding

1. Over-limit documents are retained but their unavailable status is not propagated to the query/candidate pool bindings.

   `prepare_inventory()` records document-level status and `length_preflight.unavailable_documents`, and `encode()` correctly leaves unavailable documents as NaN rows. But `bindings.json` still contains queries whose own document, positive candidate, or candidate pool member may point to an unavailable document, with no `feature_status`, `unavailable_reason`, filtered candidate pool, or query-level exclusion marker.

   This violates the module protocol: `over_limit: retain inventory; mark unavailable; exclude whole affected query pool`. It also creates a downstream correctness risk: the trainer can silently consume a query with a NaN positive or a candidate pool whose valid mask no longer contains the positive. That would either crash later, train on invalid masks, or make source-validation NCE incomparable across runs.

   Minimal fix: during `prepare_inventory()` or the immediate feature-consumption loader, build a document-status map and for every query check:

   - query document is encodable;
   - every positive ID resolves to a candidate document that is encodable;
   - candidate IDs resolve to known candidates;
   - after excluding unavailable candidates, each query still has at least one encodable positive and one valid negative if the training objective needs a negative.

   Then either remove affected queries from the trainable binding while counting them, or keep them with an explicit status such as `feature_status: unavailable_length` / `excluded_by_feature_preflight`. The important point is that unavailable feature rows cannot reach trainer loss construction as ordinary examples.

## Accepted checks

- Full tokenization preflight is explicit: `prepare()` tokenizes every exact query/candidate view with `add_special_tokens=True`, stores `input_ids`, `token_count`, and `status`, and does not truncate.
- `encode()` re-tokenizes every document and rejects any tokenizer ID mismatch before model loading/encoding.
- The model and tokenizer snapshot is strong: `model_manifest(model_path)` records all files in the model directory, and `verify_prepared(..., verify_model=True)` checks it before encoding.
- Code snapshot binding is strong for this feature run: `freeze_code()` copies route_graph plus relevant next_iteration files, and `verify_code()` compares both live and snapshotted code hashes.
- Parent data binding is present: `verify_prepared()` checks the parent data manifest hash and then calls `verify_data()` to validate data artifacts and their settings object digest.
- Source splits are carried into `bindings.json` as supplied by the frozen source data records. The feature module does not reshuffle sources.
- The representation is correctly scoped as auxiliary canonical-view causal hidden state, not original generation trace. `encode()` uses `model.model(...).last_hidden_state` and selects `mask.sum(-1)-1`, the last non-pad token after right padding.
- No labels, reader calls, native effects, or hallucination supervision enter this module.
- No overwrite of completed/started feature arrays: `encode()` rejects existing `features.partial.npy`, `features.npy`, or `manifest.json`. The fcntl lock prevents concurrent encoders. A failed attempt before creating `features.partial.npy` could still retry in the same prepared directory; that is acceptable if intentional, but the error text currently says “no overwrite/retry” more broadly than the guard enforces.

## Tests added and run

I added `tests/test_source_relation_features_review.py` with three CPU tests:

1. over-limit documents are retained with full `input_ids` and counted by length preflight;
2. prepared inventories are fresh-only and document tampering is detected by manifest hashes;
3. queries affected by unavailable documents must be removed or explicitly marked unavailable.

Command:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_source_relation_features_review.py -q
```

Result: `2 passed, 1 failed`.

The failing test is `test_unavailable_documents_exclude_or_mark_the_whole_query_pool`, which demonstrates the Required finding above.

## Required status

Do not start GPU feature encoding or trainer consumption from this inventory until unavailable-document propagation is fixed or the trainer has an equally strict verified exclusion gate with persisted counts.

## Targeted R1 closure

R1 status: closed.

Targeted inspection confirmed the fix:

- `prepare_inventory()` now deep-copies bindings and writes `query["feature_status"]` from the complete query + candidate document pool. If any referenced document is `unavailable_length`, the whole query is marked `unavailable_length`.
- Documents themselves are still retained with full `input_ids` and length status; no truncation is introduced.
- `source_relation_train.source_items()` independently excludes any query whose query document or candidate-pool document is unavailable before tensor assembly, and records `whole_query_pool_unavailable_length`.

I corrected the review tests to check the actual contract rather than a too-strong re-signed-artifact tamper case. Final targeted run:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_source_relation_features_review.py -q
```

Result: `4 passed`.

Current Required status for `source_relation_features.py`: 0.
