# Run-centric transition visualization

This workflow supersedes the earlier use of a single whole-response t-SNE as the main mechanism diagnostic.

## Why

A hallucination span may not occupy a fixed static cluster. It can instead appear as a local transition away from the response's normal graph-state trajectory. The workflow therefore separates four issues that were previously conflated:

1. **generator provenance** — which model generated the RAGTruth response;
2. **observer provenance** — which model produced the analyzed attention/hidden states;
3. **response-position drift** — structural statistics naturally change as more history becomes available;
4. **run-local change** — what changes immediately before and during one hallucination span.

## Model provenance

`TransitionSampleVisualizer.provenance()` reports:

- `observer_model` from the canonical manifest;
- the canonical/selected RAGTruth `generator_model`;
- all generator models present in the archive;
- whether generator and observer names match.

Controls are always selected from the same generator as the hallucinated sample. The first matching priority is task + data source, then task, then response length.

The repository extraction pipeline already stores these two roles separately. `ExtractionConfig.model_path` selects the observer model used to teacher-force the response and extract attention, while `generator_model` filters the original RAGTruth response generator.

## Static state changes

The 32-D rich representation is rearranged into six non-overlapping conceptual blocks:

- grounding;
- self-history;
- sparsity;
- concentration;
- locality;
- layer routing.

Within each block, features are robust-scaled. Each block is divided by `sqrt(block_dimension)`, preventing a concept from dominating the distance simply because more hand-crafted coordinates were assigned to it.

The previous cumulative locality coordinates are converted to mutually exclusive bins:

- lag = 1;
- lag 2–4;
- lag 5–8;
- lag 9–16;
- lag > 16.

This removes triple-counting of very local history attention.

## Position residualization

For a selected hallucinated sample, fully correct same-generator controls are pooled. At every normalized response position, a local median/MAD baseline is estimated from nearby control nodes. The selected response is represented by residuals from that baseline.

This answers a more useful question than raw feature plotting:

> Is the current prompt/history/sparsity/concentration state unusual **for this stage of generation**?

Position residualization is a diagnostic control that uses known-correct examples. It is not presented as a label-free detector.

## Dynamic graph state

Consecutive response nodes are compared using:

- total static-state velocity;
- velocity per structural block;
- trailing-window robust deviation;
- trailing-window deviation per block;
- Jensen–Shannon divergence of incoming source-weight distributions;
- prompt-source Jensen–Shannon divergence;
- response-history-source Jensen–Shannon divergence;
- source-neighborhood change (`1 - Jaccard`);
- early/middle/late routing shift.

These quantities target a graph-state-transition hypothesis rather than a static-class hypothesis.

## Run-centric windows

Every hallucination span is treated separately.

For a run `[start, end)`, its clean pre-window is truncated after the previous hallucination span and its clean post-window is truncated before the next hallucination span. Nearby spans are therefore not silently merged into a global `pre_error` or `post_error` class.

For sample `10071`, whose spans are `[81,84)` and `[85,87)`, the second span has only token 84 as a clean one-token pre-window. The workflow records that this is insufficient for a strong pre/error centroid comparison instead of treating tokens from the first hallucination as normal pre-error context.

## Correct-control null

For each run, two observed quantities are compared with pseudo-onsets in fully correct same-generator responses at matched normalized response positions:

- pre→error centroid shift in the position-controlled, block-balanced state;
- onset trailing-window transition magnitude.

The output reports an empirical percentile. This is more interpretable than quoting an uncalibrated Euclidean distance such as `3.07`.

These percentiles are still case-study diagnostics, not population-level p-values. A paper-level claim should aggregate run-level effects across samples and use sample/source-aware resampling or hierarchical statistics.

## Visual outputs

`TransitionSampleVisualizer.visualize()` creates, for each hallucination run:

- `run_<k>_transition.png`
  - position-residual static PCA;
  - position-residual static t-SNE;
  - rolling transition magnitude over response position;
  - block-wise rolling deviation heatmap;
- `run_<k>_control_null.png`
  - correct-control null for centroid shift;
  - correct-control null for onset transition;
- `matched_controls_shared_axes.png`
  - selected sample and same-generator controls in one joint t-SNE with shared axes;
- `transition_metadata.json`
  - provenance, run boundaries, controls, observed metrics, null distributions, and empirical percentiles.

The deterministic PCA panel should be checked before interpreting t-SNE. A separation visible only in t-SNE is weak evidence.

## RAGTruth and Llama-3.1

The original RAGTruth corpus does not contain responses generated by Llama-3.1-8B. If the canonical archive uses Llama-3.1-8B as `observer_model`, it is analyzing RAGTruth text generated by another model under teacher forcing.

That setting is valid for a **proxy/analyzer** study, but it should not be described as the internal generation dynamics of Llama-3.1.

For a clean same-generator mechanism experiment:

1. reuse RAGTruth prompts/source contexts;
2. generate new responses with Llama-3.1-8B-Instruct;
3. extract attention from the same generation model;
4. obtain new hallucination annotations for those new responses.

Original RAGTruth span labels cannot be transferred to newly generated Llama-3.1 responses because the response text changes.
