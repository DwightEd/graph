# Saved-Embedding Graph Effectiveness Audit

## Representation boundary

GroundedRoute has already completed graph construction and message passing when
it writes an attributed graph:

```text
typed edges + neighbouring nodes
-> GroundedRouteEncoder message passing
-> node_embedding [N,D]
-> node-only anomaly detector
```

The aggregation is part of the graph method. Therefore every representation
detector in this audit reads only `node_embedding`; it never performs a second
round of message passing over `edge_index`. A separately named offline
position baseline reads only normalized token position and response length to
measure annotation/position shortcuts; it is not a representation detector.

Concretely, the encoder forms a typed message for every retained edge from the
source state, attention weight, layer, head, source role, lag bucket and
prompt/response lineage. At layer \(l\), it performs

\[
m_{t,l,h}=\sum_{s<t} a_{t,s}^{l,h}\,
\phi(z_s,e_l,e_h,e_{role},e_{lag},g_s,a_{t,s}^{l,h}),
\]

adds diagonal and unresolved-mass messages, pools the head-specific cells and
updates the target with a GRU. Repeating this over Transformer layers gives

\[
z_t=\operatorname{LayerNorm}(z_t^{(L)}),
\]

which is exactly the saved `node_embedding[t]` consumed below. Thus edge and
neighbour information is present only through the construction's learned
aggregation; it is not copied into a handcrafted detector feature vector.

The saved edges remain useful for artifact integrity and encoder intervention
checks, but they are not detector inputs. A post-hoc GNN that improves over the
saved embedding would mean the encoder left information unaggregated; it would
not validate the intended final representation.

## What the current result means

The current detector is:

```text
calibration node embeddings
-> median/MAD normalization
-> whitened PCA
-> mean 20-nearest-neighbour distance
-> one anomaly score per response node
```

It does not read saved edges. An AUROC near 0.5 rejects this particular
one-class distance assumption. It does not by itself distinguish:

- a graph encoder whose embedding contains no hallucination signal; from
- an informative embedding whose normal and abnormal nodes are not separated
  by local low density.

## Three tests

### 1. Label-free artifact and aggregation audit

Before labels are opened, the module verifies sidecar SHA-256 identities,
node/index alignment, causal typed endpoints, edge uniqueness, finite values,
row-mass conservation and lineage conservation.

It also records the layer/head-resolved alignment between final node vectors
and their actual neighbours:

\[
C_{lh}(G)=
\frac{\sum_{e:r(e)=(l,h)}w_e\cos(z_{s_e},z_{t_e})}
     {\sum_{e:r(e)=(l,h)}w_e}.
\]

Comparing this value with a matched endpoint rewire checks whether the final
embedding geometry remembers exact neighbours. This is a mechanism sanity
check, not a hallucination score.

### 2. Node-only unsupervised detector benchmark

Every detector is fitted without labels on the source-disjoint calibration
embeddings and then frozen before scoring test embeddings:

```text
pca_knn          current whitened-PCA kNN baseline
isolation_forest global partition anomaly detector
lof              local-density novelty detector
one_class_svm    nonlinear one-class boundary
autoencoder      embedding reconstruction error
deep_svdd        neural one-class hypersphere distance
```

Only `pca_knn` uses the current whitened projection. The other readers operate
on the full robust-scaled node vector, so they can expose a PCA bottleneck
instead of inheriting it.

The benchmark answers whether the failure is specific to PCA-kNN. Test labels
are opened only after all scores have been saved.

`pca_knn` remains the pre-registered confirmatory detector. The other five are
reported as exploratory sensitivity analyses; choosing the best one after
seeing test labels is not accepted as a confirmatory result.

### 3. Node-only supervised readability ceiling

Labels are necessary to ask whether hallucination information is readable from
the representation at all. In an isolated diagnostic, the audit runs:

```text
linear_position / position_mlp
                architecture-matched annotation-bias controls
linear_node     balanced linear probe on node_embedding
node_mlp        compact nonlinear probe on node_embedding
```

Predictions are out of fold. Outer test folds and inner early-stopping folds
are disjoint by `source_id`; no source can appear in both training and scoring.
The result is an empirical supervised readability ceiling, not an unsupervised
result and not a new proposed detector.

The position controls are explicitly offline because they include final response
length. It is a nuisance control, not a deployable causal detector; the primary
readers remain node-embedding-only.

## Testing the construction itself

The real construction must be compared with controls *before* the final node
embedding is frozen. GroundedRoute already supports full-pipeline
`real`, `no_message`, `endpoint_rewire` and `weight_shuffle` controls. Each
encoder is trained and run separately, then passed to exactly the same
node-only detectors and probes:

```text
real graph -> message passing -> real node_embedding -> detector
same graph -> row-local updates only -> no-message node_embedding -> same detector
rewired graph -> message passing -> rewired node_embedding -> same detector
shuffled graph -> message passing -> shuffled node_embedding -> same detector
```

Post-hoc rewiring beside a fixed real embedding cannot test whether the
encoder's aggregation was effective. It can only test residual adjacency
information and is not used for the primary result.

The registered comparisons are:

```text
real - no_message
    source-neighbour state aggregation improves the final representation

real - endpoint_rewire
    exact source endpoints improve the final representation

real - weight_shuffle
    pairing strong weights with their exact endpoints improves it
```

Comparisons use aligned response rows and a paired source-cluster bootstrap.
The same detector hyperparameters, calibration policy, folds, seeds and
training budget are used for every encoded variant.

Before a comparison is admitted, the audit loads each checkpoint and verifies
that model, learning, split, seed and budget fields match except for the named
intervention. It also checks the saved control edges against the real sidecars.
The no-message control must receive the exact same graph. Endpoint and weight
controls must change at least 10% of training, calibration and test edges
globally, and at least 80% of calibration/test samples by 5% or more.
Every control must have finite training history and a non-collapsed calibration
embedding; parameter count and all non-intervention training fields must match.

A paired-run gate additionally requires real performance to be absolutely
above random under the source bootstrap. A paper-level stability claim needs
at least three paired encoder seeds; the current one-run report deliberately
marks this requirement as unfulfilled even when its within-run gate passes.
The joint construction summary passes only when all three registered controls
pass the same predeclared reader; no single ablation is promoted as proof of
the complete construction.

## Label boundary

Feature inputs are restricted to the saved calibration/test `index.npz` files
and their content-addressed graph sidecars. The canonical test split is opened
through `FrozenEvaluation` only to align labels by `(sample_id, token_index)`.
No attention tensor is consumed as a feature and no graph is reconstructed.
For formal caches whose labels are embedded in the sample files, the canonical
cache is reopened only through the evaluation API to retrieve labels and verify
response identity; this label-only exception is recorded in the report.

Outputs containing anomaly scores or out-of-fold predictions never copy the
labels. Reports explicitly separate:

```text
labels_read=false                         artifact and score freezing
labels_used_during=posthoc_evaluation     unsupervised metrics
labels_used_during=source_grouped_probe   supervised ceiling only
```

## Interpretation

```text
node MLP strong, all unsupervised detectors random
    graph representation has signal; anomaly objective is mismatched

some unsupervised detector succeeds, PCA-kNN random
    PCA-kNN was the bottleneck

real beats separately encoded no-message view
    neighbour-state aggregation contributes useful node-representation signal

real beats separately encoded rewire/shuffle variants
    exact endpoints and endpoint-weight pairing contribute useful signal

real ~= separately encoded controls
    do not claim exact endpoints or endpoint-weight pairing are useful

position MLP ~= node MLP
    apparent signal is largely annotation-position bias

all node-only probes random
    the saved representation exposes no stable signal to these registered
    readers; changing only the downstream anomaly detector is unlikely to help
```

No construction or detector is guaranteed to work on an unknown dataset. The
supervised ceiling and full-pipeline controls are the shortest reliable way to
separate representation failure from detector failure without turning the
main method into a supervised model.

Together, the controls separate neighbour aggregation, exact endpoints and
endpoint-weight pairing. `no_message` is a parameter-matched row-local
embedding ablation, not a sequence-language baseline. Its endpoint objective
and saved lineage still use topology, while its `node_embedding` path does not;
neither lineage nor edges enter the downstream detector. Removing neighbour
inputs also restricts functional capacity, so the fully capacity-matched
`endpoint_rewire` comparison is required before attributing a gain to correct
neighbours. A broader graph-versus-language-model claim needs a separately
designed sequence control.
