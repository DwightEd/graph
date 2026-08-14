# Trajectory Geometry

This is a standalone, label-free research project for studying whether an LLM's
attention routes and internal-state trajectory remain coherent while it
generates a response. It does not import, modify, or train the existing token
graph project.

The first implemented stage reads the existing formal sparse attention cache
directly and produces one route-dynamics embedding per response token. It:

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

## Extract a complete split

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
