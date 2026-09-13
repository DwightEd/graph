# Independent engineering review: O5 relation routing

Date: 2026-09-13. Reviewer: delegated engineering reviewer, applying `code-review-and-quality`. Scope: `graph/docs/RELATION_ROUTING_PLAN_20260913.md`, `reanchor/src/decoding/relation_routing.py`, its launcher, and the scheduler changes against `reanchor/results/interleave_relation_only_executed_O4_20260913.py`. Prior O4 and population reviews remain separate and unchanged.

**Disposition: no Required findings in the reviewed version.** Engineering review permits the independent execution witness to proceed. Scientific review remains **REVIEW_UNAVAILABLE**. This report does not certify a scientific claim or report an O5 GPU result.

Only CPU reads and checks were performed, with four-thread limits for the environment Python. No model GPU forward, population signal, implementation edit, or raw-output edit was performed. The sole written artifact is this report.

## Required

None outstanding in this scope.

## Protocol and implementation checks

The driver fixes receiving input to O4 `real_after_0` and donor factors to `real_after_1`. It creates 132 single-query E conditions, 32 single-layer E conditions at the final query, two joint conditions (all queries and all except final), and one same-world final-query E sham: 167 conditions. The single-query scan now manipulates E, so it can address the intervention that changed O4; the preceding X scan is not relabeled as a routing localization experiment.

For the verified input, prompt length is 549 and the response endpoint is step 131. Prediction queries are positions 548 through 679 inclusive. Patch queries use `prompt + response_step - 1`; the scored observed token is at `prompt + endpoint`. The receiver and donor retain the identical response token sequence and identical 450 source positions. Each of the 32 layers has attention shape `[32, 132, 450]` and V shape `[450, 8, 128]`; the layer and query slices match these axes. Candidate encodings are exactly `14 -> [975]` and `12 -> [717]`, and the endpoint observed token is 717.

The reused ownership implementation computes each E delta using the current receiving attention and current receiving V at that layer. It renormalizes the frozen donor source attention to each receiving head's current source mass. Donor V is passed for the common shape and finiteness validation but is not substituted for E. Earlier modifications can therefore affect subsequent receiving computations, while frozen donor A defines the intervention. The scope remains all source positions; it is not a newly narrowed numerical-endpoint swap. Other context and indirect propagation remain in the actual forward.

Before interventions, the driver requires exact equality of the complete freshly computed baseline against the saved O4 baseline, all 132 rows and the full vocabulary. For every condition it checks every prediction row strictly preceding the earliest patched response step. The same-world sham requires exact equality of all 132 rows. The final-query all-layer E condition requires exact equality of its complete endpoint vocabulary vector to the saved O4 final E vector. These are appropriate execution guards; their execution is the separate GPU witness's responsibility.

Every condition saves its full endpoint vocabulary logits, native prediction/tie information, the fixed candidate margin, distribution effect including JS, and intervention diagnostics. The margin order remains 14 minus 12; source-world expected values are not injected into the intervention. Final raw O4 E logits independently yield margin -0.875, agreeing with the saved O4 patch record. The complete saved baseline has shape `[132, 128256]` and final E has shape `[128256]`.

## Provenance and scheduler

The CPU audit independently recomputed and matched all seven O4 file hashes consumed by this driver, all nine inherited code hashes, and all 16 saved model-file metadata records. It inspected all 128 A/V array headers across the two worlds and 32 layers. The driver records these dependencies, the prior manifest hash, execution environment, and copies of its execution sources; it refuses an existing output directory and writes completion only after the condition loop and manifest.

The model identity check is the stated file-name/size/mtime check, not a new weight-content hash. This matches O4 provenance and is supplemented by the exact baseline and final-E runtime comparisons. No stronger weight-hash claim is made here.

The scheduler diff only adds a two-choice `--experiment` dispatcher and separate journal/log names. The default remains O4. The archived O4 executed scheduler is preserved. The shared nonblocking exclusive interleave lock, population PID/output/code checks, population run lock while the mechanism child executes, child cleanup, and release-before-resume sequence remain unchanged. Both experiment choices share the same lock, so this change does not reopen the previously repaired cross-scheduler race. Runtime confirmation that the population resumes and completes new work belongs to the independent witness.

Python AST parsing and launcher `bash -n` passed. The inherited mechanism implementation was unchanged; its earlier CPU regression checks are documented in the prior review rather than represented as newly run tests.

## Optional follow-up and claim boundaries

The downstream recall analysis should verify `candidate_queries.json` against the frozen O4 manifest before using its preselected source-rise top eight. That file is not consumed by this capture driver, so its absence from the driver's seven-file list is not a capture defect. The score, direction, and budget should remain the O4 frozen choices, including missed sufficient queries and undefined recall for an empty sufficient set.

A flip under one E condition establishes sufficiency of that concrete intervention on this fixed example. Comparing joint all-query E with joint E excluding the final query tests necessity within that joint intervention. Neither result identifies a unique earliest screening event or establishes population-level hallucination detection. Joint effects without single-query flips should be described at that intervention scope; they do not alone prove a unique distributed representation. The plan appropriately labels this as O4-driven exploratory follow-up and retains the constructed-source truth versus natural-error distinction.

## Reviewed source SHA-256

- Plan: `85cb1ac263dc2a97949328cbf6304eadaf70c5cb8f5045a6bb35df37eb4e635f`
- Driver: `b11c3a38d355483fa21f2682a54a8b4bc6ad3c6cc360efa9584b48c4fb76aa38`
- Launcher: `3f34d98a7aa1f001b5ea087ba26b1a4e251a79e2bf47d8d582068080921e1f88`
- Scheduler: `1e62bee161d661b303d46781b5c28123045dedceeac9b37fbd30d81426d4e70e`
