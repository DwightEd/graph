# Current research status

## Active hypothesis

GroundedRoute tests whether token-level attention routing contains reusable
structure beyond prompt/response mass and position. The concrete hypothesis is
that response tokens differ in how much prompt-origin routing survives through
response relays, and that exact source endpoints help identify this structure.

This is not yet evidence that hallucinations form a stable error attractor. The
current detector only tests whether the learned token embedding is useful for
label-free one-class detection.

## Active graph

One prompt-response sample produces one independent `TokenGraph`:

```text
node: token
edge: (source, target, layer, head, retained weight)
mass: retained + diagonal + unresolved = 1 per response/layer/head row
```

Prompt-query rows are unavailable and are not fabricated. No top-k pruning or
layer/head averaging occurs at the graph boundary.

## Active model

GroundedRoute contains:

- a row-stochastic learned correspondence between heads in adjacent layers;
- conserved prompt-origin, response-closed, and unresolved lineage;
- layer-ordered, head-aware messages along exact source endpoints;
- a right-shifted prefix state that predicts the next retained endpoint;
- one frozen embedding per token;
- a PCA-whitened kNN detector that reads only response-token embeddings.

The encoder is trained without hallucination labels. It does not use masked
reconstruction, and its endpoint prediction loss is not reused as the anomaly
score.

## Required controls

All representation variants must freeze the same `z[token,d]` artifact and use
the same downstream detector:

1. local layer-head representation without endpoints;
2. causal sequence representation without attention endpoints;
3. full token graph;
4. endpoint-fixed weight shuffle;
5. role/degree/coarse-log-lag-matched endpoint rewire.

The endpoint control does not preserve exact lag. All three active variants
(`real`, `weight_shuffle`, and `endpoint_rewire`) must use the same frozen
source split, seed, training budget, and detector. Control checkpoints, indices,
and scores record the actual changed-edge fraction and reject an ineffective
intervention below the configured threshold.

HoloRoute and Flat-1024 remain reconstruction baselines, but they do not replace
the causal-sequence control.

## Evidence currently available

The code path, artifacts, and synthetic invariance tests are implemented. No
full GroundedRoute RAGTruth result is recorded yet. Historical HoloRoute gains
do not validate the new token representation.

## Required acceptance gates

A graph contribution is allowed only when all of the following are reported:

1. full and truncated-prefix encodings are numerically equivalent;
2. changing current/future rows cannot change the current route prediction;
3. mathematical edge-storage permutations leave embeddings unchanged;
4. the full graph improves token AUPRC over the causal-sequence control;
5. real endpoints outperform degree/role/lag-matched rewiring;
6. real source-weight assignment outperforms endpoint-fixed weight shuffle;
7. all variants use the same source-disjoint fit/calibration/test protocol and
   the same frozen embedding detector;
8. paired source-group bootstrap is reported over at least five seeds;
9. QA, Summary, and Data2txt are evaluated separately.

## Stop rules

```text
graph ~= causal sequence   -> remove the graph contribution claim
real ~= endpoint rewire    -> remove exact-endpoint and relay-path claims
real ~= weight shuffle     -> remove the strong-edge landing claim
graph ~= role-only         -> method is a prompt/response routing summary
prefix test fails          -> remove online/causal language
position dominates         -> reject the detector regardless of raw AUROC
```

## Next experiment

Run the full QA representation pipeline from one frozen commit:

```bash
bash experiments/grounded_route/run.sh
```

Then run the implemented weight-shuffle and endpoint-rewire controls with the
same split, encoder capacity, training budget, and PCA-kNN detector. The
causal-sequence baseline remains a required addition before making a graph
contribution claim. Archive the graph specs, checkpoint, embedding indices,
scores, evaluation, and commit SHA before changing the method.
