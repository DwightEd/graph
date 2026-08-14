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
cd trajectory-geometry
python -m pip install -e .
```

## Inspect one existing cache file

```bash
python -m trajectory_geometry.cli inspect \
  --attention /path/to/attention_10005.pt
```

## Extract a complete split

```bash
python -m trajectory_geometry.cli extract \
  --attention-root /path/to/attention_cache \
  --split train \
  --output-dir /path/to/trajectory_geometry/train
```

The command searches the selected split recursively, prints progress for every
sample, and writes a `manifest.json` only after all selected samples finish.
Use `--limit 5` for a smoke test and `--save-raw-route` only for small audits.

## Output per sample

Each compressed NPZ contains:

- `route_embedding [response_tokens, embedding_dim]`;
- token-level `temporal_js`, `depth_js`, `head_js`, `route_acceleration`;
- token-level prompt/history/self/unresolved route mass;
- anchor names and exact configuration metadata;
- optionally `raw_route_mass [layers, heads, response_tokens, anchors]`.

These are representations and diagnostics, not hallucination scores. Detection
and labels belong to the later frozen evaluation stage.
