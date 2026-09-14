# Unsupervised Token Graph Detector

## Goal

Model one complete prompt+response as one causal token graph and produce a
label-free token risk score. RAGTruth token labels are joined only after the
score freeze for evaluation; they are never used to fit features, thresholds,
graph topology, or model weights.

## Mainline

1. Load the existing RAGTruth observer cache. Prefer the formal sparse CSR
   response-attention cache from `reanchor`; keep the legacy dense cache as a
   compatibility path.
2. Build one graph per response. Nodes are prompt and response tokens. Edges
   are causal attention edges retained above the frozen attention floor. Node
   features contain position/segment information, attention diagonal, causal
   in/out statistics, and optional frozen hidden-state features.
3. Fit a source-balanced graph autoencoder on unlabeled reference responses.
   The encoder predicts each node from its local topology and node features;
   the decoder reconstructs node features and edge-weight summaries. No
   hallucination labels or response annotations enter this phase.
4. Convert reconstruction error plus topology surprise into a frozen token
   anomaly score. Calibrate only an unlabeled source-level alarm budget on a
   disjoint calibration roster.
5. Join RAGTruth labels after prediction artifacts and scorer code are frozen.
   Report all-token, through-first-error, strict-post-first, source-balanced
   AUROC/AUPRC and coverage. Do not call the score a semantic proof.

## Controls

- node-only autoencoder without edges;
- shuffled destination edges preserving per-query mass;
- raw entropy/NLL when available;
- source-disjoint reference versus calibration versus official test.

## Explicit boundaries

This is an unsupervised anomaly-ranking detector, not a hallucination
classifier and not a causal interpretation of the original generator. The
graph is an attention-derived diagnostic representation. Unknown or invalid
cache rows remain unavailable and are not silently dropped.

## Planned modules

| Module | Responsibility |
|---|---|
| `data.py` | formal CSR/dense cache loading and response records |
| `graph.py` | one-response causal graph and tensor features |
| `model.py` | label-free graph autoencoder |
| `score.py` | frozen anomaly scores and unsupervised calibration |
| `evaluate.py` | post-freeze RAGTruth token evaluation |
| `run.py` | prepare, fit, score, evaluate phases |
