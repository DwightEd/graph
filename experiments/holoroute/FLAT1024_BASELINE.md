# Flat-1024 Baseline

`flat-1024` is the mandatory no-topology control for HoloRoute. It receives the
same all-layer attention values but no depth, relay, query-set or diamond
adjacency.

For each exact token pair `(source, target)`, all layer/head values are grouped
into

\[
X_{s,t}\in\mathbb R^{L\times H}.
\]

For Llama-3.1-8B, `L=H=32`, so the raw value vector has 1024 coordinates. The
model also receives the observation mask, a layer-presence mask, source role,
causal lag and normalized source/target positions. These masks preserve the
cache-censoring contract but do not introduce relations between different token
pairs.

## Self-supervision

Training masks complete layer blocks `X[s,t,l,:]`, never individual scalar
channels. A residual MLP sees the remaining flattened pair tensor and predicts
the masked head vector. Scoring partitions every existing pair-layer block into
deterministic folds so every block is evaluated once. Block errors are averaged
for the response token targeted by the pair.

The baseline uses the same source-group train/validation/calibration split, the
same nuisance variables, the same conditional density and the same post-hoc
evaluation as HoloRoute. Its checkpoint reports the raw feature dimension and
parameter count.

## Interpretation

- `HoloRoute > flat-1024`: graph organization contributes beyond all-layer data.
- `HoloRoute ~= flat-1024`: gains likely come from the high-dimensional
  attention tensor rather than depth/relay/query topology.
- `flat-1024 > HoloRoute`: the current graph inductive bias is harmful or its
  self-supervised tasks are misaligned.

The hidden dimension may be changed to approximately match the HoloRoute
parameter count. Parameter count must be reported with every comparison.
