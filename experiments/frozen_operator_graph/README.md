# Frozen Hypernetwork Operator Graph

This package constructs response-token graphs without hallucination labels and
without training a second GNN to imitate the Transformer's message passing.
The graph is extracted from the frozen causal language model's actual forward
computation.

## Method contract

For decoder layer `l`, query token `t`, source token `s`, and query head `h`,
the exact post-softmax route is

\[
a^{(l)}_{h,t,s}.
\]

For grouped-query attention, query head `h` reads value head `kv(h)`. The
per-source, per-head context is

\[
u^{(l)}_{t,s,h}
=
 a^{(l)}_{h,t,s}V^{(l)}_{s,kv(h)}.
\]

Concatenating heads and applying the checkpoint's actual output projection gives

\[
m^{(l)}_{s\to t}
=
W_O^{(l)}\operatorname{vec}
\left(u^{(l)}_{t,s,1},\ldots,u^{(l)}_{t,s,H}\right).
\]

The layer attention update is reconstructed exactly:

\[
\Delta h^{(l)}_{t,\mathrm{attn}}
=
\sum_s m^{(l)}_{s\to t}+b_O^{(l)}.
\]

The package checks both

\[
A^{(l)}V^{(l)}=\text{o\_proj input}
\]

and

\[
W_O^{(l)}(A^{(l)}V^{(l)})+b_O^{(l)}
=
\text{captured attention output}
\]

before accepting a graph.

## What is a node and what is an edge?

The persisted graph is a layer-typed multiplex token graph:

- token coordinates remain the original absolute prompt+response positions;
- every response token is a downstream node for evaluation;
- a layer-specific edge connects causal source `s` to response target `t`;
- `edge_layer` identifies the Transformer layer;
- `edge_attention_code[e]` is the complete cross-head vector
  `[a_1,...,a_H]`, not a head average;
- edge features contain route mass, role-normalized mass, exact pre-output
  value energy, lag, head-code entropy/effective number/top-1 share, and the
  checkpoint-induced operator-code norm.

Sources are partitioned exactly into three disjoint roles:

1. `prompt`: source lies before `response_start`;
2. `history`: earlier generated response source;
3. `self`: the diagonal source equal to the current response target.

Future sources receive no role and can never become graph edges.

## Full graph and exact quotient mode

The default is

```text
route_mass_retention = 1.0
value_energy_retention = 1.0
```

which exposes every causal token pair, including exact cross-head attention
codes. No edge selection is performed.

For very large runs, both retentions may be set below one. This does **not**
zero-fill or discard the remaining data. Selection is performed separately
inside each `(layer, target, source_role)` block using the union of:

- the minimum deterministic source prefix preserving the requested route mass;
- the minimum deterministic source prefix preserving the requested pre-output
  value energy.

Every unexposed source is still consumed in an exact role-specific quotient
remainder. The remainder conserves:

- per-head context before `W_O`;
- total route mass;
- `W_O`-projected residual message;
- source count, lag moments, head-code statistics, and message alignments.

The artifact records the maximum conservation errors. Any violation above the
configured tolerance is a hard failure.

## Response-token representation

No learned graph encoder is used. The deterministic node representation is the
concatenation of:

```text
final_hidden       [R, D]
route_features     [R, L, H, 27]
layer_features     [R, L, F_layer]
temporal_features  [R, 34]
```

`route_features` preserve the full layer-head axes and report, for each
prompt/history/self role:

```text
mass
source entropy
source effective number
source top-1 share
lag mean and variance
value-message energy
aggregate value-context norm
message coherence
```

`layer_features` use the actual residual computation and include:

```text
pre-attention and pre-MLP normalized-state norms
normalization alignment with residual states
attention / MLP / layer-output norms
prompt-history, prompt-MLP, history-MLP, attention-MLP alignment
prompt route fraction and prompt residual-message fraction
QK-route versus value-message grounding mismatch
layer update magnitude and state/update alignment
role-wise route, energy, message, coherence, and operator-code statistics
complete role-wise mean unit cross-head operator code
```

`temporal_features` add response-position dynamics, layer slopes, late-layer
summaries, token-to-token hidden/operator stability, grounding loss, MLP versus
prompt-message dominance, and a response lock-in index. They are deterministic
mechanism coordinates, not a supervised classifier.

The flattened `node_embedding` contains every channel above. There is no PCA,
random projection, learned adapter, hidden-state average, layer average, or head
average in the stored representation.

## Label firewall

Construction opens the canonical `research_dataset` with
`retain_embedded_labels=False`. It never calls `labels()`,
`prepare_evaluation_labels()`, or `response_labels()`.

Every artifact and split manifest records:

```text
labels_consumed_by_construction = false
fallbacks_used = []
```

Labels may be opened later by the user's independent evaluation code. The
provided `OperatorGraphDataset` intentionally has no label interface.

## Data fidelity and hard failures

The package has no hidden-state surrogate, no mean-attention fallback, no
fabricated prompt-query rows, and no zero-filled censored attention path.
Instead, it teacher-forces the exact cached token sequence through the exact
checkpoint using eager attention and captures:

```text
full response-query attention probabilities
all source value states
actual o_proj input
actual attention output
residual input and post-attention residual
pre-attention and pre-MLP normalized states
MLP update and layer output
final normalized hidden state
```

Before graph construction, the fresh dense attention is bound to the existing
sparse cache by checking:

- every retained endpoint;
- every exact diagonal endpoint;
- every omitted causal off-diagonal entry against the declared censoring floor.

A checkpoint mismatch, dtype/mask mismatch, head-order mismatch, unsupported
architecture, missing hook, noncausal edge, nonfinite tensor, or failed message
identity stops the run.

## Package layout

```text
config.py       deterministic construction configuration
schema.py       in-memory and persisted tensor contracts
basis.py        frozen W_O blocks and W_OV operator geometry
capture.py      exact Llama/GQA teacher-forced signal capture
binding.py      dense replay <-> sparse cache numerical binding
graph.py        exact messages, role decomposition, and quotient graph
encoding.py     label-free node and trajectory encoding
artifacts.py    atomic storage, hashes, manifest, provenance
dataset.py      lazy verified graph-artifact reader
pipeline.py     end-to-end split construction with label firewall
run.py          CLI
run_qa.sh       one-command QA launcher
tests/          formula, capture, binding, artifact, and firewall tests
```

## Install into the repository

From the extracted delivery directory:

```bash
bash install_into_graph_repo.sh \
  /share/home/tm902089733300000/a903202310/lys/research/graph
```

The installer copies only the new experiment directory and workflow. It refuses
to overwrite an existing `experiments/frozen_operator_graph` directory unless
`OVERWRITE=1` is explicitly supplied.

## Run tests

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
conda run -n research pytest -q \
  experiments/frozen_operator_graph/tests
```

## One-command QA run

The launcher already contains the project paths used on the server:

```text
repository  /share/home/tm902089733300000/a903202310/lys/research/graph
model       /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
raw source  /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl
formal data /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876/test
```

Run the complete exact graph with no required environment variables:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
conda run -n research bash experiments/frozen_operator_graph/run_qa.sh
```

The script reads the exact replay dtype and runtime provenance from the formal
cache manifest. It refuses a different checkpoint, PyTorch/Transformers build,
attention implementation, or non-empty output directory. The raw
`response.jsonl` is hashed into artifact provenance; it is not reparsed for
graph features because doing so could change the formal cache's token alignment.

Small smoke test:

```bash
LIMIT=2 OUT=/tmp/frozen_operator_graph_smoke \
conda run -n research bash experiments/frozen_operator_graph/run_qa.sh
```

Intentional replacement of an existing output:

```bash
OVERWRITE=1 conda run -n research bash \
  experiments/frozen_operator_graph/run_qa.sh
```

## Read artifacts in the existing evaluator

```python
from experiments.frozen_operator_graph import OperatorGraphDataset

graphs = OperatorGraphDataset(
    "experiments/frozen_operator_graph/outputs/qa_full",
    verify_hashes=True,
)

artifact = graphs[graphs.sample_ids[0]]
node_representation = artifact.node_embedding       # [response_token, feature]
edge_index = artifact.edge_index                    # [2, exposed_edge]
edge_layer = artifact.edge_layer                    # [exposed_edge]
edge_code = artifact.edge_attention_code            # [exposed_edge, head]
route_tensor = artifact.route_features              # [R, L, H, 27]
layer_tensor = artifact.layer_features              # [R, L, F_layer]
```

The evaluator can open its own token labels only after these bytes have been
frozen and hash-verified.
