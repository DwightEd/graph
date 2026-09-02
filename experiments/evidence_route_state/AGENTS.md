# Evidence route state rules

Read and follow `../grounded_route/iclr/ENGINEERING_RULES.md` before changing
this directory.

- This project has one method: a single-forward, head-resolved attention-write
  DAG followed by label-free route-state tracking.
- The computation edge targets query position `q`; the evaluated token is
  position `q + 1`. Keep both indices explicit.
- A local edge is derived from the actual dynamic `A`, `V`, GQA mapping, and
  the matching head block of the current layer's `W_O`. Reconstruct the native
  attention write before using the edge.
- Never average heads before head/source routes have been constructed. Never
  infer a cross-layer head identity.
- Propagate prompt ancestry through actual graph endpoints. Split response
  history into evidence-rooted relay and unrooted feedback; predictor self is
  a separate route.
- Route concentration is not a hallucination label. The primary state is the
  conjunction of route contraction and persistent unrooted feedback.
- MLP is a same-token nonlinear update, not a parameter-knowledge source.
  MLP quantities are diagnostic unless an explicit experiment validates them.
- Capture, graph construction, state fitting, and scoring must not read
  hallucination labels. `evaluate.py` is the only label-opening module.
- Keep the code path `data -> capture -> messages -> graph -> lineage -> state
  -> detector`; `run.py` only orchestrates it.
- Do not add file hashes, model-directory scans, schema migration, duplicate
  pipelines, defensive utility layers, supervised regressors, AE/GNN variants,
  or four-branch intervention code.
- Tests must target predictor alignment, AVWO reconstruction, BF16 arithmetic,
  GQA/head identity, multi-hop ancestry, legitimate narrow focus, topology
  controls, and label isolation.
