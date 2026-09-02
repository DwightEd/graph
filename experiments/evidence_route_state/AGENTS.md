# Evidence route state rules

Read and follow `../grounded_route/iclr/ENGINEERING_RULES.md` before changing
this directory.

- This directory contains one formal method. Do not add `v2`, `new`, legacy,
  or parallel detector paths.
- The computation event is query position `q`; its evaluated token is `q + 1`.
  Keep `query_position` and `prediction_position` explicit everywhere.
- The four origins are fixed and ordered as `evidence`, `prompt`, `response`,
  and `endogenous`. They form an additive residual ledger, not four latent
  classes and not hallucination labels.
- Initialize evidence and other-prompt token embeddings in their matching
  registers, prior response-token embeddings in `response`, and zero in
  `endogenous`.
- Propagate registers through the observed native RMS scale, dynamic `V`,
  native attention gates, GQA mapping, and the matching per-layer/head `W_O`.
  The four channel writes must reconstruct the native attention write.
- Put the native MLP write into `endogenous`. This register is internal
  nonlinear state, not a claim about parameter knowledge. Keep an explicit
  numerical closure check at every layer boundary.
- Never average heads or layers before constructing the registered route and
  its Gram tensor. Never infer a cross-layer head identity.
- Every causal source endpoint contributes to `route_topology`. Sparse edges
  are allowed only for visualization and cannot affect the detector.
- The detector consumes the complete structured graph frame: final registered
  node embeddings, residual Gram, head-write Gram, route topology, MLP
  relation, and readout-margin contribution. Do not replace this frame with a
  feature list, PCA, random projection, learned adapter, HMM, supervised
  regressor, autoencoder, or GNN.
- The primary score is a source-disjoint, label-free conditional transition
  energy under the full-tensor product metric. It must preserve eight actual
  next-state prototypes per task, position decile, and prompt-length quartile
  instead of averaging them into one center.
- `data.py`, capture, register construction, graph construction, metric
  fitting, neighbor selection, and calibration must not open hallucination
  labels. `evaluate.py` is the only label-opening module.
- Teacher forcing uses the one observed response only. Do not add response
  pairs, knockout branches, patching, deletion replay, or counterfactual
  generation to the primary pipeline.
- The historical QA functional-route-collapse equation remains a locked
  control. It is not an input to the primary detector and is not renamed as a
  new mechanism.
- Keep the formal path readable as `data -> capture -> registered messages ->
  graph frame -> product metric -> conditional energy -> evaluate`; `run.py`
  only orchestrates it.
- Do not add file hashes, model-directory scans, schema migration, defensive
  utility layers, duplicate pipelines, or hidden fallbacks.
- Tests must cover predictor alignment, exact register closure, native AVWO
  reconstruction, BF16 arithmetic, GQA/head identity, MLP assignment, dense
  endpoint use, layer order, source isolation, label isolation, and a correct
  narrow-focus example.
