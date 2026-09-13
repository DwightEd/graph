# Restoration v2 pipeline engineering review — 2026-09-13

Scope: `next_iteration/grounded_graph_restoration_train.py`, `next_iteration/grounded_graph_restoration_predict.py`, and `next_iteration/grounded_graph_restoration_evaluate.py`, with read-only inspection of `grounded_graph_restoration_features.py` because these modules depend on its `query_empty/full_query` contract. I added CPU-only tests in `tests/test_grounded_graph_restoration_pipeline_review.py`. I did not edit implementation files, run GPU, read labels, or mutate frozen v1 artifacts.

Verdict for the reviewed files: **Critical 0, Required 0**.

## Basis for C0/R0

- **All-source-token erasure contract is bound before training/prediction.** `grounded_graph_restoration_features.intervention()` recomputes the source-token set from tokenizer offsets with positive overlap against `source_span`, compares it to stored `source_offsets`, preserves query positions as pre-token positions, records whole-token boundary crossings, and saves original/executed input digests. `load_example()` exposes `query` as `H_empty` and `full_query` as the original source-full query, so train/predict do not relabel the original packet.

- **Training uses the intended single-variable v2 kernel.** `restoration_train.py` accepts only `source_reconstruction` feature parents, uses the restoration representation, reuses the v1 adapter/measure/frozen-head kernels and hyperparameters, preserves source train/validation SHA separation, writes epoch-0 `initial.pt`, and selects by the same v1 validation objective for the planned single-variable comparison. It does not touch natural labels.

- **Prediction separates the two v2 score families.** `score_one()` computes base logp/entropy from `full_query`, empty logp from `H_empty`, and adapter restore logp from the restoration adapter. It writes both `*_difference = logp_full - logp_restore` and `*_restoration_difference = logp_empty - logp_restore`; the CPU test demonstrates these can diverge under the same adapter output. Graph/no-edge/permuted branches are emitted separately.

- **Whole word denominator is retained.** `words()` emits every non-whitespace response word and marks unavailable entries explicitly. Available word scores are token scores weighted by character overlap. `run()` writes one JSON record for every feature entry and a `.npz` only for available entries, with manifest census matching that rule.

- **Evaluation reads labels late and validates identity.** `restoration_evaluate.py` first verifies complete frozen prediction artifacts, feature parent binding, live/snapshot code, canonical label-free population settings, population roster, response/source/generator/split identity, exact word roster, and annotation byte hash. It copies the evaluator before label access and only then reads `response.jsonl`. It reports fixed high-risk scores separately without score flips or fitted combinations.

- **No semantic or native overclaim is encoded.** Protocols and outputs keep `labels_read=False` before evaluation, `native_route_claim=False`, and evaluation scope explicitly says reused development panel unless official-test entries are supplied. Owner accuracy is marked unavailable.

## Tests added/run

Added: `tests/test_grounded_graph_restoration_pipeline_review.py`.

Command from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest \
  tests/test_grounded_graph_restoration_pipeline_review.py -q
```

Result: **5 passed in 12.71s**.

Covered by tests:

- positive-overlap source erasure and unchanged non-source/response-prefix IDs;
- boundary-crossing whole-token receipt without pretending character-local erasure;
- full-gap and empty-restoration score definitions remain distinct;
- source-balanced ranking behavior and single-class `None` metrics;
- evaluation rejects a population that already used labels before copying/reading labels.

## Non-blocking boundaries to preserve in run reports

- The source48 mechanism gate described in the v2 plan is not enforced by these three files; root is implementing that as a separate source-dependence diagnostic. Natural prediction/evaluation should not be interpreted as graph evidence-use success unless that separate diagnostic shows positive source-heldout anchor gain and fixed-query graph-payload erasure drop. This is a pending external artifact, not a defect in the reviewed train/predict/evaluate code.

- Unavailable words are kept in the denominator with zero-filled scores. That is appropriate for denominator honesty, but if `unavailable_words > 0`, result tables should also expose availability counts prominently; zero-filled NLL values must not be described as measured confidence.

- The edge permutation is a destructive destination rewiring control, not a semantic counterfactual. It is suitable as a topology-sensitivity diagnostic only.

- The development-panel evaluation remains late-label evaluation on reused data. It can diagnose whether v2 fixes the v1 failure pattern, but it is not an independent official-test claim until the frozen score set is carried to held-out official test.
