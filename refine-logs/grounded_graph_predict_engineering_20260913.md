# Grounded graph natural prediction bounded engineering review — 2026-09-13

Scope: `next_iteration/grounded_graph_predict.py`, plus the already-reviewed adapter/train contracts it consumes. I added CPU-only tests in `tests/test_grounded_graph_predict_review.py`. I did not edit implementation files, read labels, run feature preparation, or start GPU work.

Verdict: **Critical 0, Required 0** for prediction freeze readiness under the current protocol.

## Verified

- **Score direction:** `score_one()` computes `graph_difference = logp_base - logp_graph` and `no_edges_difference = logp_base - logp_no_edges`, matching the fixed “larger is risk” direction. Base NLL is `-logp_base`.
- **Full word denominator:** `words()` emits every non-whitespace response word. Available word scores are token scores weighted by character overlap; unavailable words remain present with status `unavailable` and zero neutral differences.
- **Exact checkpoint binding:** `load_adapter()` rejects checkpoint arm/epoch/vocabulary/settings mismatches and accepts epoch-0 `initial.pt` under the selected-checkpoint contract.
- **No training-input label path:** `run()` refuses feature parents that are not `natural_response` before GPU/model work. Output records keep `labels_read=False` and `native_route_claim=False`; the module does not evaluate labels or claim semantic truth.
- **Graph/no-edge/permutation calls:** available entries run graph, no-edge, and destination-permuted graph adapter passes from cached features; observer forwards remain zero. The permutation helper does not mutate the original edge tensor and produces deterministic within-node-index destination rewiring from the supplied seed.
- **Training/source overlap reporting:** prediction records report whether the natural source SHA was present in source-reconstruction train/validation or unseen, instead of hiding transfer overlap.
- **Snapshot binding:** prediction settings bind feature and training manifest SHAs, selected checkpoints, live/snapshot code, model files through feature verification, and selected checkpoint hashes through `load_adapter()`.

## Tests added/run

Added: `tests/test_grounded_graph_predict_review.py`.

Combined targeted command from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest \
  tests/test_grounded_graph_train_review.py tests/test_grounded_graph_predict_review.py -q
```

Result: **9 passed in 13.76s**.

## Non-blocking boundaries

- `permuted_graph_difference` is a destructive destination-rewiring topology control, not a degree-preserving or semantics-preserving graph perturbation. This is acceptable because the protocol names it as a fixed edge-destination permutation; reports should not interpret it as a matched semantic counterfactual.
- Unavailable entries receive zero scores for all score names and explicit unavailable word status. Downstream evaluation must respect that status rather than treating zero NLL/entropy as a real model confidence value.
- The module predicts from cached natural-response features. It does not construct B/A, route evidence, or certify original-generator causality.
