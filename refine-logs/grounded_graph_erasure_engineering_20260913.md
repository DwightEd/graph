# Grounded graph owner-payload erasure control review — 2026-09-13

Scope: `next_iteration/grounded_graph_erasure.py` plus the already-reviewed train/predict contracts it consumes. I added CPU-only tests in `tests/test_grounded_graph_erasure_review.py`. I did not edit implementation files, run GPU, mutate frozen features, or read labels.

Verdict: **Critical 0, Required 0** for running the erasure control after the training/prediction artifacts exist.

## Verified

- **Same-position source-token erasure:** `erase_capture()` replaces only prompt source token positions belonging to observed target-owner nodes with the single-token space replacement. It keeps response-prefix tokens unchanged and preserves sequence length/positions.
- **Actual recomputation:** the test fake model exercises the final-norm hook contract. The receipt records one observer forward, the executed input IDs, original/executed input hashes, erased prompt token indices, owner indices, and array hashes/shapes.
- **Original coordinates held fixed:** source pooling uses the original node prompt-token coordinate sets against the recomputed hidden states, matching the intended control estimand.
- **Branch separation:** `graph_source_erased` can keep original query states while replacing source node features; `full_source_erased` uses the recomputed query states from the perturbed input. These are different estimands and should stay separately reported.
- **Anchor-only measurement:** `measure()` evaluates only pointer-supervised anchor queries and reports base logp, adapter logp gain, and summed coordinate pointer probability for graph and no-edge arms.
- **Run-level denominator:** the runner requires exactly 48 source-validation entries. I checked the current reconstruction summary read-only: `sources_without_pointer` is 0, so the implemented anchor requirement is satisfied for this planned validation control.
- **No semantic overclaim:** protocol and summary set `semantic_necessity_claim=False`, `native_route_claim=False`, and `labels_read=False`.

## Tests added/run

Added: `tests/test_grounded_graph_erasure_review.py`.

Combined command from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest \
  tests/test_grounded_graph_train_review.py \
  tests/test_grounded_graph_predict_review.py \
  tests/test_grounded_graph_erasure_review.py -q
```

Result: **11 passed in 17.11s**.

## Scientific-control boundaries

- A gain drop under erasure supports dependence on the erased source payload under this trained adapter. It does not prove semantic necessity, original-generator routing, or correct natural ownership.
- A surviving gain is not automatically “formatting only,” because unselected duplicate/near-duplicate source evidence can still carry the same answer. The protocol’s redundant-evidence warning is necessary and should be repeated in results.
- For QA/Summary contexts, erasing a context owner can remove more than a single scalar payload. Interpret those rows as context-evidence ablations, not single-value payload ablations.
- The space-token replacement is a same-position perturbation, not a natural missing-evidence source. It is appropriate for this anti-copy diagnostic, but should not be used as a semantic absence proof.

## Non-blocking note

`erase_capture()` itself does not independently check that the model is frozen/eval. The runner loads the model as `eval().requires_grad_(False)`, so this is not a launch blocker. If the helper is reused outside this runner, mirror the stricter `capture()` guard.
