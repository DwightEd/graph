# Trajectory Geometry

This is the primary, label-free research project for studying whether an LLM's
attention-conditioned internal-state transition remains coherent while it
generates a response.

The implemented main model is one graph dynamical system rather than a feature
mixture:

```text
compressed attention CSR + consecutive hidden states
  -> shared train-only hidden PCA/whitening
  -> node-control / true-graph / causally-rewired state equations
  -> sample-held-out reconstruction gate
  -> per-token cross-layer residual DCT vectors
```

All state equations are solved by closed-form trimmed ridge. No labels, GNN,
gradient training, or backpropagation are used.

The earlier route-dynamics diagnostic reads the formal sparse attention cache
and produces one attention-only embedding per response token. It:

- preserves every layer and head until the final fixed projection;
- maps variable token sources into comparable prompt/history/self/unresolved
  anchors;
- treats censored attention as an explicit unresolved-mass category;
- measures temporal drift, depth drift, head disagreement, and route
  acceleration;
- uses a deterministic CountSketch projection rather than learned weights or
  layer/head averaging;
- never reads hallucination labels.

The method and staged implementation gates are specified in [METHOD.md](METHOD.md).

## Run the graph state model

The hidden root must contain train/test sidecars aligned by sample id:

```bash
HIDDEN_ROOT=/actual/hidden_cache bash run_graph_state_model.sh
```

Smoke test:

```bash
LIMIT_TRAIN=10 LIMIT_TEST=5 \
HIDDEN_ROOT=/actual/hidden_cache \
bash run_graph_state_model.sh
```

Accepted hidden arrays are full-sequence consecutive states with shape
`[layers+1,tokens,hidden_dim]` (or its token-major transpose). A cache with
only block outputs `[layers,tokens,hidden_dim]` is aligned by skipping the first
attention layer. Response-only and non-consecutive selected-layer caches are
rejected because they cannot support prompt-source messages and real per-layer
transitions.

The output `manifest.json` reports only label-free prediction MSE and whether
the true graph beats both node control and causally rewired topology. Per-token
NPZ files contain the three residual embeddings, layerwise prediction error,
`graph_gain`, and `rewire_gap`. This stage intentionally produces no AUROC or
2-D plot.

## Install

```bash
cd trajectory_geometry
python -m pip install -e .
```

## Inspect one existing cache file

```bash
python -m trajectory_geometry.cli inspect \
  --attention "$FORMAL_ROOT/train/attention_10005.pt"
```

## Extract the attention-only route diagnostic

```bash
LIMIT=5 bash run_route_dynamics.sh
```

The runner defaults to the same formal cache used by
`run_token_representation.sh`:

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/
outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

Override it only when needed:

```bash
FORMAL_ROOT=/actual/cache/root bash run_route_dynamics.sh
```

The command searches the selected split recursively, prints progress for every
sample, and writes a `manifest.json` only after all selected samples finish.
Use `--limit 5` for a smoke test and `--save-raw-route` only for small audits.

The data class exposes two recovery interfaces:

- `iter_sparse_row_blocks(4096)` returns bounded CSR blocks and is used by the
  main encoder;
- `iter_dense_rows()` returns one `[tokens]` thresholded row at a time, fills
  every absent sparse entry with zero, and never allocates `[L,H,R,N]`.

An absent entry means “below the cache extraction floor,” not a known original
attention value of exactly zero. The dense iterator follows the requested zero
fill convention; route statistics additionally retain the total censored mass
as `unresolved_mass` so that truncation is not mistaken for model certainty.

## Output per sample

Each compressed NPZ contains:

- `route_embedding [response_tokens, embedding_dim]`;
- token-level `temporal_js`, `depth_js`, `head_js`, `route_acceleration`;
- token-level prompt/history/self/unresolved route mass;
- anchor names and exact configuration metadata;
- optionally `raw_route_mass [layers, heads, response_tokens, anchors]`.

These are representations and diagnostics, not hallucination scores. Detection
and labels belong to the later frozen evaluation stage.

## Evaluate the extracted feature effect

After both train and test extraction completes, one foreground command runs the
entire frozen Gate-A evaluation; it does not extract attention again:

```bash
bash run_feature_effects.sh \
  /share/home/tm902089733300000/a903202310/lys/data/feature_extraction/trajectory_geometry/route_v1/train \
  /share/home/tm902089733300000/a903202310/lys/data/feature_extraction/trajectory_geometry/route_v1/test
```

It fits label-free conditional kNN detectors on train routes, writes
`detector_state.pt` and `scores_label_free.npz`, then opens test labels only for
evaluation. The output directory contains `results.json` and a compact
`summary.txt`. The fixed comparison views are nuisance, low prompt mass, route
mass, route dynamics, route embedding, their summary, and their full mean.
