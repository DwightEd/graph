# SourceRel-Mini trainer engineering review — 2026-09-13

## Scope

Reviewed `next_iteration/source_relation_train.py` against
`refine-logs/SOURCEREL_EXPERIMENT_PLAN.md`, the frozen source-reconstruction
interfaces, and `SourceRelMini`. This was a bounded CPU engineering review. I
added `tests/test_source_relation_train_review.py`; I did not change trainer
implementation, use a GPU, read hallucination labels, or claim model quality.

The frozen data summary is consistent with the stated input contract: 883
Data2txt official-train sources, split into 720 source-train and 163
source-validation sources, with 5,760 and 1,304 selected queries respectively.
The target is actual source-field pointer reconstruction. It is explicitly not
natural ownership, factuality, support/conflict, or RAGTruth annotation ground
truth.

## Required found and closed

### R1 — fit TF-IDF only on retained source-train query pools

The initial implementation fit TF-IDF on every train binding document, even an
unavailable query/pool that `source_items` had excluded from actual examples.
That violated the plan's identical retained-view/pool baseline contract.

The fix derives the fit document IDs from the retained `source_train` items'
`qidx` and `cidx`, records their count and digest, and records that neither
validation nor unavailable/excluded pools contributed. It keeps validation
transformation/evaluation intact without fitting its vocabulary.

The new fixture has one train pool, one whole unavailable pool, and one
validation pool. It proves the unavailable pool is reported as excluded, only
three retained train documents are fitted, validation is not fitted, and both
partitions still receive predictions.

## Confirmed contracts

- `source_items` accepts each source exactly once and only with an immutable
  `source_train` or `source_validation` split. Training and checkpoint
  selection consume separate source lists.
- A query whose own document or any candidate document is unavailable is
  excluded as a **whole pool** with a retained exclusion record. It cannot turn
  into a shortened candidate set or a partially supervised example.
- Candidate pools and positives are checked against the source-local candidate
  map and broad value type. Multi-positive rows remain multi-positive in both
  masks and `owner_nce`; no arbitrary same-context index is selected.
- `batch_masks` creates block-diagonal valid/positive matrices. A source's
  query cannot score a candidate from another source even when the sources
  share a training batch.
- The train loop rejects non-finite loss and non-finite parameter gradients.
  Validation uses query-mean owner-NCE and snapshots the lowest value before
  final prediction/checkpoint serialization.
- The output keeps hard-negative subsets, candidate IDs and reconstruction
  scope, while writing `natural_ownership_accuracy: null`; it does not present
  pointer recovery as natural ownership ground truth.
- Fresh-output, prepared/feature manifest, parent reconstruction kind, code
  snapshot and end-of-run upstream revalidation boundaries prevent an output
  from being silently resumed under changed features/code.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -m pytest -q tests/test_source_relation_train_review.py
# 2 passed in 12.53s

/tmp/research_lint_20260912/bin/ruff check \
  next_iteration/source_relation_train.py \
  tests/test_source_relation_train_review.py
# All checks passed
```

The tests cover retained-pool TF-IDF fitting, unavailable/validation exclusion,
cross-source masking, multi-positive loss, and finite CPU gradients. They are
interface tests, not evidence that the head learns a useful natural relation.

## Verdict

**Critical 0 / Required 0.** The trainer is ready for its planned frozen
feature sanity stage. Any eventual reconstruction metric must remain scoped to
source-field pointers and source-disjoint validation; it cannot be promoted to
natural ownership, RAGTruth factuality detection, or native causal evidence.
