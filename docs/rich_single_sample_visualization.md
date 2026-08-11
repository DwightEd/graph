# Rich single-sample graph-state visualization

## Goal

This workflow is for **case-study analysis of one response graph**, not pooled dataset visualization. Its purpose is to test whether a hallucination span is associated with an interpretable change in graph state and whether that change begins before the labeled error onset.

Use:

```python
from rich_visualization import RichSampleVisualizer

viewer = RichSampleVisualizer(DATA_ROOT, output_root=OUTPUT_ROOT)
sample_id = viewer.error_sample_ids[0]
result = viewer.visualize(sample_id)
```

The main response-node projection keeps one point per response token. The representation is deterministic; no GNN or supervised encoder is trained.

## 32-D response-node state

The original 12 structural features are retained, then expanded with information that the coarse representation previously discarded.

### Original structure

- incoming mass
- prompt mass share
- normalized entropy
- history lag
- total / prompt / history degree
- total / prompt / history density
- history edge share
- channel edge density

### Grounding and self-history strength

- prompt mass
- history mass
- prompt entropy
- history entropy

Keeping absolute prompt/history mass as well as their ratios distinguishes a weak graph from a strong graph with the same prompt share.

### Concentration

- top-1 share
- top-3 share
- HHI concentration
- prompt top-1 share
- history top-1 share

These variables target the hypothesis that hallucination can retain fewer but more concentrated relations.

### Locality

- history-lag standard deviation
- history mass within lag <= 1, <= 4, <= 8
- history mass at lag > 16

This preserves more of the distance distribution than a single mean lag.

### Layer-group routing

Layers are divided into early / middle / late thirds. For every response target, prompt and response-history attention mass is measured independently in each group:

- early / middle / late prompt mass
- early / middle / late history mass

This avoids collapsing all layer behavior into a single mean before visualization.

The response representation uses only incoming relations for the current response position. It does not use future response queries, so it remains suitable for reasoning about causal/online detection signals.

## Four response phases

Hallucination labels are not used to fit the projection. After t-SNE coordinates are computed, response tokens are annotated as:

- `far_normal`
- `pre_error`: configurable window immediately before an error span
- `hallucination`
- `post_error`: configurable window immediately after an error span

This is intended to test a state-transition hypothesis rather than only binary separation.

## Generated diagnostics

`visualize()` writes the following under `outputs/single_sample_rich/<sample_id>/` by default when the notebook is used.

### `rich_response_tsne.png`

Four panels use the **same response-node coordinates**:

1. far-normal / pre-error / hallucination / post-error markers;
2. normalized response position, to expose position confounding;
3. prompt grounding (`prompt_mass_share`);
4. concentration (`hhi`).

### `projection_stability.png`

Repeats the response projection at multiple t-SNE perplexities. A visual cluster that disappears under small parameter changes should not be treated as robust evidence.

### `source_role_tsne.png`

Includes **prompt and response tokens together** using a separate 12-D source-role representation: how strongly, how often, and at what later response stages each token is used as an attention source.

This view is descriptive and future-looking. Do not use it as an online detector feature.

### `rich_feature_heatmap.png`

Shows robust-scaled graph-state variables around the first hallucination onset. This is often more useful than t-SNE for seeing exactly which structural channels change and when.

### `rich_feature_differences.png`

Ranks standardized feature changes between the immediately preceding pre-error window and the first hallucination span.

### `matched_control_joint_tsne.png`

Fits one joint projection over the selected hallucinated response and a matched fully correct response. The correct control is chosen by same source when possible, then task/data source, then response length.

## Quantitative checks

`separation_metrics()` works in the original robust-scaled 32-D feature space rather than the t-SNE plane. It reports:

- silhouette score for hallucination vs non-hallucination nodes when enough nodes are available;
- Euclidean centroid distance between hallucination and pre-error nodes.

These are diagnostic quantities, not a population-level significance test. Claims about general hallucination behavior require aggregation across samples with sample/source-aware statistical analysis.

## How to interpret the hypothesis

Evidence consistent with the current hypothesis would include a repeated pattern such as:

- prompt mass / prompt density decreasing;
- history mass / history edge share increasing;
- near-history shares increasing and far-history share decreasing;
- degree or channel density decreasing;
- top-1 / top-3 / HHI increasing while entropy decreases;
- a consistent early/middle/late layer-group routing change;
- pre-error nodes shifting toward the hallucination region before the labeled onset.

A visually clean t-SNE split is **not required** for the hypothesis to be meaningful. The heatmap, original-space feature differences, stability view, and matched-control projection are designed to distinguish a genuine structural transition from a t-SNE artifact or response-position effect.
