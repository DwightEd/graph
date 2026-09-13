# SourceRel-Mini engineering review — 2026-09-13

Scope: bounded `code-review-and-quality` review of `next_iteration/source_relation_data.py` and `next_iteration/source_relation_model.py` against `refine-logs/SOURCEREL_EXPERIMENT_PLAN.md`. I wrote targeted CPU tests only. I did not edit implementation files, start GPU work, read labels, download models, or review broader native code.

## Verdict

Critical: 0.

Required: 1 before freezing SourceRel training data from population inputs.

Required before M1/GPU feature encoding: 1 protocol gate must be present in the feature runner or added to the data artifact summary.

## Required findings

1. `source_relation_data.prepare()` accepts malformed frozen rows whose `response_sha256` does not match `response`.

   The code checks the 14-field allowlist and source ID digits, but it does not reuse `audit_runner.rows()` or independently validate `response_sha256`, response ID uniqueness/digit constraints, or response text integrity. SourceRel training uses only the source text, so this does not leak gold labels, but it does weaken the provenance contract: a row that is not a valid frozen observer row can still decide which source enters train/validation and can be written into a sealed training-data run.

   Evidence: I added `tests/test_source_relation_mini_review.py::test_prepare_rejects_frozen_row_response_hash_mismatch`; it fails because `prepare()` does not raise on a row with `response="actual response"` and `response_sha256="0"*64`.

   Minimal fix: mirror typed-hours prepare and call `route_graph.audit_runner.rows(args)` before filtering to official-train Data2txt sources, or add the same checks locally: exact field allowlist, duplicate response ID rejection, digit response IDs, and `sha256(response.encode()) == response_sha256`. Keep the source-ID consistency check afterward.

2. Before feature encoding/GPU, the implementation must run and freeze a query/candidate length preflight over the actual encoder tokenizer.

   This is not yet a `source_relation_data.py` logic bug, because the plan intentionally preserves all immediate scalar siblings without truncation. But that choice means query/candidate text can include long sibling fields such as review text even when the target scalar itself is short. The current data artifact does not enforce or summarize tokenizer lengths. The M1 feature runner must record counts, max/percentile token lengths, over-4096 failures, and skipped examples before any GPU training/feature run. If that runner is not guaranteed, add a CPU tokenizer-length summary to data prepare.

## Accepted checks

- The training target is honestly source-field pointer reconstruction, not natural ownership ground truth or support/conflict classification. The protocol and output fields state no labels, no reader calls, no native effects.
- Data is filtered to official `train` and `Data2txt`, with source de-duplication by `source_id` and a deterministic source-hash validation split.
- `compile_source()` uses the actual literal graph root: `record_root_id = graph["literal_graph"]["root"]`, and verifies that the root node has empty path and dict kind. This fixes the prior owner metadata class of bug for SourceRel data.
- Query text masks the selected scalar value but preserves independent homographs. This matches the refined dependency-closure decision: do not globally delete the same string/number/unit from unrelated fields.
- Encoder text avoids list indices and occurrence/source IDs. `_field_words()` drops integer path components, and the serialized query/candidate text does not include occurrence IDs or absolute source offsets.
- Identical masked contexts are treated as multi-positive rather than arbitrary-index positives. The implementation computes positives by identical query view within the same value-type pool and deduplicates identical query IDs.
- Candidates are same-source and same broad value type. Homograph negatives and other-record-same-field flags are recorded for review/metrics.
- Pair loss is disabled rather than tautologically trained from visible graph edges. This matches the refinement: topology is fixed provenance unless the target relation is masked.
- `SourceRelMini` matches the intended minimal model: two bias-free projections, type scaling, cosine/temperature score, owner NCE only, role mass derived from candidate softmax, and owner beam with source reuse allowed.
- The additive typed owner metadata correction file exists at `outputs/typed_hours_owner_metadata_correction_20260913.json`, contains 370 corrections, reports `model_forwards: 0`, `labels_read: false`, and leaves old run artifacts unmodified.

## Tests added and run

I added `tests/test_source_relation_mini_review.py` with four targeted checks:

1. actual root ID is used and independent homograph values survive as negatives;
2. duplicate masked contexts become multi-positive and list indices do not enter encoder text;
3. malformed frozen-row response hashes should be rejected by prepare;
4. model valid masks, NCE, role mass, and reusable owner beam behave as specified.

Command:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_source_relation_mini_review.py -q
```

Result: `3 passed, 1 failed`.

The single failure is the Required provenance bug above. The other three tests passed.

## Optional cleanup

`SourceRelMini.forward()` could validate `query_types` dtype/range and input finiteness before the embedding/projection call. The current code will fail or propagate invalid values later, but a clear `ValueError` would make trainer failures easier to diagnose. This is not a blocker for the current data provenance fix.

## Targeted R1 closure

R1 status: closed for the original input-integrity Required.

The implementation now uses the validated frozen-row input path, and the targeted review suite passes:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_source_relation_mini_review.py -q
```

Result: `4 passed`.

The remaining tokenizer-length preflight is still a pre-GPU protocol gate for the feature runner, not a blocker for the corrected CPU data prepare path.

## Final R1 test correction

The review test itself was corrected to avoid relying on the random initial ranking of an untrained `SourceRelMini` head. The model test now sets deterministic projection weights before asserting beam order.

Final targeted run:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_source_relation_mini_review.py -q
```

Result: `4 passed`.

Current Required status for SourceRel data/model code review: 0 for the R1 input-validation issue. The tokenizer-length preflight remains a required gate before GPU feature encoding, as stated in the plan.
