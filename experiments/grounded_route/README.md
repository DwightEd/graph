# GroundedRoute

GroundedRoute is an attention-only, label-free representation prototype for
token-level hallucination research. It produces one reusable embedding for
every prompt and response token, preserves the exact retained attention
endpoints and all layer/head identities, and applies a separate one-class
detector only after representation learning.

It is not a masked graph autoencoder. Its self-supervised task predicts the
source endpoint of a retained attention edge from the causal prefix before the
current attention row is observed.

## Method

One prompt-response sample becomes a typed causal token graph:

```text
node       token position
edge       source token -> response token
edge type  (layer, head)
edge value retained attention weight
```

The sparse cache retains all 32 x 32 layer/head identities. Missing edges stay
censored below the cache floor; they are not converted into observed zeros.
Prompt-query rows are unavailable and are never fabricated.

The model first propagates a conserved three-state path lineage through the
graph:

```text
P  prompt-origin path mass
C  response-closed path mass
U  unresolved/censored path mass
```

For head correspondence between adjacent layers,

\[
B_l[h,h'] = \operatorname{softmax}_{h'} \beta_l[h,h'].
\]

If \(\pi_{s,l-1,h'}\) is the source lineage, the current-head source
lineage is

\[
\bar\pi_{s,l,h}=\sum_{h'}B_l[h,h']\pi_{s,l-1,h'}.
\]

After the cache-tolerance correction, each response row conserves normalized
retained, diagonal, and unresolved mass:

\[
\pi_{t,l,h}=
\sum_{p\in P}\tilde a_{tp}^{lh}e_P+
\sum_{s\in R_{<t}}\tilde a_{ts}^{lh}\bar\pi_{s,l,h}+
\tilde d_t^{lh}\bar\pi_{t,l,h}+u_t^{lh}e_U,
\qquad
\sum_s\tilde a_{ts}^{lh}+\tilde d_t^{lh}+u_t^{lh}=1.
\]

Thus an RR edge can carry prompt-origin mass instead of being treated as
automatically ungrounded. This lineage is injected into the typed edge message;
it is not a collection of post-hoc scalar features.

For response token `t`, the endpoint predictor uses only `G_<t` to distinguish
an observed source from role- and lag-matched causal non-edges. After observing
the current row, the encoder updates the token state through its exact typed
neighbors. The resulting `node_embedding[N,D]` is the method output; endpoint
loss is a self-supervised training objective rather than the anomaly score.

For each retained endpoint \(s^+\), training samples causal non-edges \(s^-\)
with the same source role and logarithmic lag bucket. With

\[
\ell_\theta(s\mid t,l,h)=\frac{q_{t,l,h}^{\mathsf T}k_s}{\sqrt D},
\]

the implemented normalized route loss is

\[
\mathcal L_{route}=
\frac{\sum_e a_e\,
\operatorname{softplus}\!\left(
\ell_\theta(s_e^-)-\ell_\theta(s_e^+)\right)}
{\sum_e a_e}.
\]

The query \(q_t\) is computed from a right-shifted recurrent prefix state, so
it cannot read token \(t\)'s current attention row. The current candidate
edge's weight is used only as the loss weight and is not an endpoint-score
input. Once a row is observed, its retained weights do enter later node
embeddings through both `log1p(weight)` edge-value encoding and multiplicative
message weighting. The anomaly score is not this loss; it is computed later
from the frozen embedding.

The first detector is intentionally simple and independent of graph features:

```text
calibration node embeddings
-> median/MAD normalization
-> whitened PCA
-> k-nearest-neighbor distance
-> one scalar per response token
```

It reads only learned embeddings. It does not read token position, degree,
attention statistics, lineage, or hallucination labels.

## Code map

```text
graph.py       canonical sparse attention -> typed TokenGraph
model.py       TokenGraph -> one embedding per token
learning.py    causal endpoint prediction objective
artifacts.py   graph spec, checkpoint, encoded graph and score contracts
detection.py   PCA-whitened kNN
pipeline.py    build, fit, encode and detect orchestration
evaluate.py    labels opened after score freezing
run.py         command-line interface
run.sh         complete experiment
controls.py    matched endpoint and weight interventions
```

The same `fit -> encode -> detect` path supports three frozen graph variants:
`real`, `weight_shuffle`, and `endpoint_rewire`. Every variant uses the same
source split, seed, model capacity, optimization budget, and embedding-only
detector. Its checkpoint, embedding index, and score record the variant and the
actual changed-edge fraction; a control is rejected when its aggregate change
falls below the configured minimum.

The two controls are applied deterministically per sample:

- `shuffle_weights_keep_endpoints` keeps endpoints, support, row mass, role
  mass, and each group weight multiset while changing which endpoint receives
  a strong edge;
- `rewire_endpoints_keep_roles` performs causal degree-preserving double swaps
  within layer/head/role/coarse-log-lag strata while weights stay in their
  original rows. It preserves only the logarithmic lag bucket, not exact lag.

If the real graph does not outperform these controls after retraining and with
the same detector, the exact-endpoint/multi-hop claim must be removed.

## Run

From the repository root:

```bash
TRAIN_SPLIT=/path/to/attention/train \
TEST_SPLIT=/path/to/attention/test \
VARIANT=real OUT=experiments/grounded_route/outputs/qa/real \
TASK=QA DEVICE=cuda EPOCHS=8 \
bash experiments/grounded_route/run.sh
```

The explicit stages are also available:

```bash
python -m experiments.grounded_route.run build \
  --data /path/to/train --output train_graph.json --task QA

python -m experiments.grounded_route.run fit \
  --spec train_graph.json --checkpoint model.pt \
  --variant real --device cuda

python -m experiments.grounded_route.run encode \
  --spec train_graph.json --checkpoint model.pt \
  --scope calibration --output calibration --variant real --device cuda

python -m experiments.grounded_route.run build \
  --data /path/to/test --output test_graph.json --task QA

python -m experiments.grounded_route.run encode \
  --spec test_graph.json --checkpoint model.pt \
  --scope all --output test --variant real --device cuda

python -m experiments.grounded_route.run detect \
  --calibration calibration/index.npz --test test/index.npz \
  --reference detector.npz --scores scores.npz

python -m experiments.grounded_route.run evaluate \
  --test /path/to/test --scores scores.npz --output evaluation.json
```

## Artifacts

```text
train_graph.json          lightweight data selection; no copied training graph
model.pt                  encoder checkpoint and source-disjoint split
calibration/graphs/*.pt   calibration graph + node embeddings
calibration/index.npz     calibration response-node embeddings
test/graphs/*.pt          self-contained encoded token graphs
test/index.npz            merged response-node embeddings
detector.npz              frozen PCA-kNN reference
scores.npz                one label-free scalar per response token
evaluation.json           post-hoc AUROC/AUPRC and source bootstrap
```

Each encoded graph contains `node_embedding[N,D]`, typed endpoints,
`diagonal[R,L,H]`, `unresolved[R,L,H]`, and the single interpretable lineage
tensor `[R,L,H,3]`. It does not contain labels or a collection of detector
residuals. Each index records the SHA-256 of every referenced graph sidecar.

The implementation keeps the full sparse topology on CPU and transfers only
one layer of edges to the compute device at a time. Training checkpoints each
layer step, so device-side edge activations scale with the largest layer rather
than the whole graph. Preserving every retained endpoint still requires
\(O(E)\) host memory; this is the explicit cost of avoiding top-k pruning.

## Research status

This implementation establishes a clean representation-learning and artifact
boundary. It does not yet establish that kNN distance detects a stable error
attractor, nor does it make a SOTA claim. That question requires real-data
comparisons, matched endpoint interventions, multiple seeds, and separate QA,
Summary, and Data2txt evaluation.
