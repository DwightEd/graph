# Attention Hypernetwork Operator-Code Validation

This experiment tests a mechanism suggested by *Attention as a Hypernetwork*:
for one layer and one query-key pair, the vector over attention heads is not
only a scalar route weight. It is a pair-specific code that combines reusable
head operators.

For layer `l`, query `q`, source `k` and query head `h`,

\[
z_{q,k}^{(l)}=
[a_{1,q,k}^{(l)},\ldots,a_{H,q,k}^{(l)}],
\qquad
B_h^{(l)}=W_{O,h}^{(l)}W_{V,kv(h)}^{(l)}.
\]

The linear value-path operator selected by the code is

\[
W^{(l)}(z)=\sum_h z_h B_h^{(l)}.
\]

Two codes are compared with the head-operator Gram matrix

\[
G_{h,g}^{(l)}=\langle B_h^{(l)},B_g^{(l)}\rangle_F,
\qquad
\|W(z)-W(z')\|_F^2=(z-z')^T G^{(l)}(z-z').
\]

This is a forward value-path operator geometry. It is not a full Transformer
Jacobian, a logit-specific causal effect, or proof that attention alone is the
true functional contribution.

## Why the extracted matrices are cached

The LLM parameters are frozen. `operator_geometry.pt` is therefore extracted
once and reused for every dataset and every mechanism run that uses the same
model checkpoint. It stores:

```text
per-layer operator Gram                 [L,H,H]
unit-head normalized Gram               [L,H,H]
Gram square-root factors                [L,H,H]
head operator Frobenius norms           [L,H]
query-head -> KV-head mapping            [H]
model geometry and provenance metadata
```

The code does **not** save every explicit
`B_h = W_O,h @ W_V,h` matrix by default. For a 32-layer, 32-head model with a
4096-dimensional residual stream, explicit `D x D` matrices would require tens
of gigabytes even in half precision. The Gram artifact is only a few hundred
kilobytes and is sufficient for all operator-code distances in this experiment.

Set `SAVE_BASIS=1` to additionally save one factorized basis file per layer:

```text
W_O head blocks       [H,D,d]
unique W_V KV blocks  [H_kv,d,D]
query-head/KV mapping
optional projection biases
```

These factors can later be combined with captured layer hidden states to obtain
actual message vectors without materializing the full `B_h` matrices.

## Three-stage protocol

```text
[1] frozen LLM weights
      -> extract and cache operator geometry

[2] attention cache, labels sealed
      -> group sparse edges by (layer,target,source)
      -> form H-dimensional pair codes
      -> separate route magnitude from code direction
      -> freeze answer-level mechanism features

[3] frozen feature artifact
      -> open token labels
      -> answer label = any hallucinated response token
      -> univariate tests + source-grouped logistic readability probes
```

The supervised logistic probes are mechanism-readability diagnostics. They are
not an unsupervised hallucination detector.

## Features and controls

Mass/topology features test existing hypotheses:

```text
prompt versus generated-response mass
source effective number and top-1 share
mean causal lag and broad-but-shallow routing
self and native unresolved mass
```

Pair-code features are computed under four geometries:

```text
identity                 raw Euclidean head-code control
operator_raw             full W_O W_V Gram geometry
operator_normalized      unit-norm head operators; tests operator correlation
operator_permuted        head/operator binding permutation control
```

For each geometry the code reports prompt/history operator dispersion,
prompt-history operator distance, same-layer switching across generated tokens,
late response-code stability, response lock-in, and an early-confusion to
late-collapse statistic.

The central comparisons are:

```text
mass + operator_normalized  > mass only
mass + operator_normalized  > mass + raw head code
mass + operator_normalized  > mass + operator_permuted
```

Without these gains, the attention head vector may be only another correlated
routing statistic rather than a useful functional code.

## Run QA

The model path must identify the same frozen LLM that produced the attention
cache.

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
conda run -n research bash \
experiments/attention_operator_validation/run_qa.sh
```

The first run extracts the operator geometry. Later runs print
`Reuse cached operator geometry` and do not reload the LLM.

Optional factorized basis export:

```bash
MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
SAVE_BASIS=1 \
conda run -n research bash \
experiments/attention_operator_validation/run_qa.sh
```

Small smoke test:

```bash
MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
LIMIT=16 BOOTSTRAP=50 CV_FOLDS=3 \
OUT=experiments/attention_operator_validation/outputs/smoke \
conda run -n research bash \
experiments/attention_operator_validation/run_qa.sh
```

Reuse an existing operator cache while changing feature imputation:

```bash
MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
START_STAGE=2 IMPUTATION=midpoint \
OUT=experiments/attention_operator_validation/outputs/qa/midpoint \
conda run -n research bash \
experiments/attention_operator_validation/run_qa.sh
```

Supported censored-head treatments are `zero`, `floor`, `midpoint`, and
`excess`. A mechanism result should be considered robust only when its direction
survives reasonable censoring treatments and controls for observed-head
coverage and native unresolved mass.

## Current data boundary

The existing cache contains response-query rows, retained typed edges, exact
diagonal mass and row-level unresolved mass. It does not contain prompt-query
rows or layer hidden states. Consequently this experiment can validate
pair-specific attention codes and operator geometry, but cannot yet compute the
actual input-conditioned message

\[
W^{(l)}(z_{q,k}^{(l)})\,\widetilde x_k^{(l)}.
\]

That stronger test requires a small recache with pre-attention normalized hidden
states, or fixed low-dimensional sketches of `W_O W_V x`.
