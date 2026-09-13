# Ownership factorial engineering review — 2026-09-12

Reviewer: separate engineering-review agent using `/root/.agents/skills/code-review-and-quality/SKILL.md`.
Scope: `graph/route_graph/ownership.py`, `graph/tests/test_ownership.py`,
`reanchor/src/decoding/ownership_factorial.py`, `reanchor/scripts/run_ownership_factorial.sh`,
and `graph/docs/OWNERSHIP_FACTORIAL_PLAN_20260912.md`.

This is an engineering review, not independent scientific approval. The scientific
review backend is unavailable. No GPU inference, code edits, or raw-result edits
were performed by this reviewer.

## Verdict

Request changes for hook cleanup and completion of the predeclared world-level
distribution reporting. The current single-patch experiment is not invalidated by
the cleanup issue, and world JS can be derived from persisted logits without
rerunning GPU inference. No causal-index error or fake-ground-truth/detection claim
was found in the reviewed implementation and plan.

## Required

1. **Remove installed hooks if patch registration or validation fails.**
   In `ownership.py:123–192`, hooks are registered before entering the `try/finally`.
   A valid layer-0 patch followed by a layer-1 patch whose sources are not visible
   at its query raises the intended `ValueError` but leaves all three layer-0 hooks
   attached. A later forward on the same model is then silently intervened upon.
   A CPU reproduction with the same small Llama architecture as the tests observed
   hook counts **0 before / 3 after the rejected call**. `capture_source_factors`
   has the analogous exposure when a later requested layer is invalid.
   Put registration inside the cleanup boundary, or validate all specifications
   before registering anything. Verify that an invalid second specification leaves
   hook counts and subsequent unpatched logits unchanged.

2. **Complete the B1 full-vocabulary JS report.**
   The plan promises full-vocabulary JS alongside the source×history readouts, but
   `ownership_factorial.py:162–170` writes only candidate margin/preference, top-5,
   and entropy for each world. `distribution_effect` is used for local patches and
   the correct-onion control, not for the four B1 worlds. Publish clearly labeled
   JS contrasts at both grill and onion readouts, including source changes at each
   fixed history and history changes at each fixed source. The existing
   `s*h*_logits.npy` files are sufficient; a derived analysis artifact can complete
   the current run while preserving raw outputs.

## Optional

- **Strengthen execution provenance.** Include `route_graph/adoption.py`, which
  supplies the JS computation, and the launch script in the code hash/snapshot
  list. Current snapshots include the ownership module, driver, and
  `adoption_probe.py`; this does not fully identify the metric implementation.
- **Keep the native control distribution central in reporting.** The control
  predicts the ordinary token ` over`, so the logged 14-versus-12 preference is
  incidental there. Its JS, native top token, and saved-token log-probability change
  are the relevant control measurements. The runner already provides these.

## Verified behavior and limits

- Read tests first. The supplied three tests check independent X/E effects,
  receiving source-mass preservation, a nonadditive X+E example, exact same-world
  exchange, MLP swaps, and unchanged earlier predictions. The parent reported
  **3 passed in 25.91 seconds**; this reviewer did not redundantly rerun that suite.
  The real-model test uses a small random CPU Llama in float32, so the production
  bf16 exact-sham check remains a separate runtime requirement enforced by the
  runner.
- Independently loaded both real traces and the local tokenizer on CPU. Both have
  prompt length **535**, **427** source positions spanning **64–499**. S1 changes
  exactly positions **76 and 341**, swaps **12/14**, preserves the complete token
  multiset, and leaves the answer prefix unchanged. Their decoded contexts are
  respectively bratwurst cooking before removal and bratwurst grilling.
- H1 changes exactly absolute position **634 = 535 + 99**. Grill prediction t99
  reads query **633** and the observed target is **14**. Onion prediction t131
  reads query **665** and its observed target is **12**. Thus H1 is in the future
  of the grill prediction and visible to the onion prediction. The driver tests
  equality through the grill readout before reporting effects.
- Correct-onion trace `00013` reads query **660 = 535 + 125**, predicting t126
  ` over` in “continue cooking the onions over low heat.” Its source-only swap
  is correctly constructed and compared using native full-vocabulary effects.
- `source_context_delta` implements X with receiving A and donor V, E with
  receiving V and rescaled donor A, and X+E with donor V and rescaled donor A.
  Scaling preserves receiving source mass separately for each query/head; GQA
  repeats KV heads in the expected order. Zero donor mass with positive receiving
  mass is rejected. The delta changes only the source contribution before O.
- Replaying O over its original complete input shape supports exact same-world
  equality under the same numeric path. Attention, residual, MLP, and later layers
  then run normally; MLP patches replace the actual MLP update. Donor MLP effects
  are correctly described as context-dependent, not pure parameter knowledge.
- Main and interaction effects use the full `[source, history]` 2×2 matrix with
  the declared signs. Default execution enumerates **64** local interventions,
  **4** shams, **4** worlds, and **2** control forwards. Layers and scopes are
  fixed before results; lower-layer CLI subsets are explicitly recorded.
- No labels are fit, no artificial result is presented as natural ground truth,
  and no AUROC, detector accuracy, repair, or unique-node-location claim is made.
  The onion number has no asserted uniquely correct replacement. Interpretation
  is limited to two observed windows from one response and one control response.
- Readability, architecture, security, and performance are reasonable for this
  bounded local experiment: no new dependencies, offline model loading, fixed
  loops, bounded thread counts, fresh output directories, and preserved failures.
  The review does not establish scientific validity or generalization.
