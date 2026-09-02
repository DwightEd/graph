# Registered information-route graph

This experiment performs label-free token-level hallucination detection from
the route graph of one teacher-forced forward pass. It does not assume that a
single pattern such as concentrated attention or response-history dominance is
always an error.

The method keeps four additive residual origins:

```text
E  external evidence
P  question, instruction, system, and other prompt context
R  prior response-token embeddings
M  endogenous state introduced by native MLP computation
```

Under the RMS scales and attention gates observed in the native forward pass,
the four registered `A V W_O` writes reconstruct every attention residual
write. The native MLP write enters `M`; at later layers it can travel through
real token-to-token attention edges. The four final registers reconstruct the
native final hidden state after its observed RMS gate.

This is an exact additive ledger of the observed computation, not a
counterfactual causal decomposition. In particular, `M` is not automatically
parameter knowledge and neither response dominance nor route contraction is
automatically hallucination.

## Pipeline

The formal computation is:

```text
sample and prompt roles
  -> teacher-forced native capture
  -> E/P/R/M registered AVWO messages
  -> compact head/layer graph frame
  -> full-tensor product metric
  -> multimodal conditional transition energy
  -> label-free calibration
  -> evaluation
```

The graph frame contains:

```text
node_embedding      [token, 4, hidden]
residual_gram       [token, layer+1, 4, 4]
head_write_gram     [token, layer, head, 4, 4]
route_topology      [token, layer, head, 4, 7]
mlp_relation        [token, layer, 5]
margin_contribution [token, 4]
```

`route_topology` is computed from every causal source endpoint. Its final axis
is:

```text
log route mass
log effective source count
top-one source share
prompt-source fraction
response-history fraction
predictor-self fraction
head-to-head route consensus
```

No head or layer is averaged before the registered messages and Gram tensors
are built. No PCA, random projection, learned adapter, GNN, autoencoder,
supervised regressor, or HMM produces the primary state. Sparse edges are used
neither inferred from the compact artifact nor used to change its score.

## Detection

The detector compares the complete graph-frame tensors with a product metric.
Normal dynamics are represented by actual transitions

```text
(graph[t-2], graph[t-1]) -> graph[t]
```

rather than one mean state. For each task, response-position decile, and
prompt-length quartile, eight diverse observed windows are retained as
prototypes. Their next graph states remain separate. Conditional energy
is the negative log product-kernel compatibility of the current next state
with that context-weighted multimodal set.

The first two answer events are captured for inspection but are not assigned a
primary score because a complete two-frame context does not yet exist.

Prototype, calibration, and evaluated sources are disjoint. Hallucination
labels are opened only after scores have been frozen in `evaluate.py`.

The historical QA functional-route-collapse result (`AUROC` about `0.7337`) is
reported unchanged as a control. It is not blended into the new score.

## Files

The layout borrows the research-object separation of
[`belindal/state-tracking`](https://github.com/belindal/state-tracking): data,
model observation, representation, analysis, and evaluation have distinct
owners. It does not copy that project's probe objective or implementation.

- `data.py` reconstructs the exact cached sequence and assigns evidence versus
  other-prompt origins without labels.
- `capture.py` runs the frozen observer with explicit query/prediction
  coordinates and bounded-memory KV chunks.
- `messages.py` defines exact native attention-message geometry and matching
  GQA/`W_O` helpers.
- `registers.py` propagates additive origins through the observed native gates
  and closes the residual ledger in FP32 derived arithmetic.
- `graph.py` forms the compact graph frame from all dense endpoints.
- `metric.py` defines the fixed six-block product distance without learned
  projection or weights.
- `detector.py` selects actual transition prototypes and performs label-free
  conditional calibration.
- `controls.py` contains only the locked historical route-collapse equations.
- `evaluate.py` is the only module that reads hallucination annotations.
- `visualize.py` renders a selected sample from the same registered graph.
- `run.py` only orders the stages.

The exact equations and claim boundaries are in `METHOD.md`.

## Run

The foreground-only entry remains:

```bash
bash experiments/evidence_route_state/run_all.sh
```

The shell script contains only:

```bash
python -m experiments.evidence_route_state.run all
```

A small `--limit 2` run verifies execution and tensor invariants only. Two is
the minimum because reference and calibration sources remain disjoint even in
a smoke run. It cannot establish detection effectiveness because its
prototype strata, calibration tails, and source-bootstrap intervals are
necessarily undersampled.

## Resource envelope

For a 4095-token Llama-3.1-8B replay, the FP32 four-register value cache adds
about 2.0 GiB of GPU memory and the native BF16 KV cache about 0.5 GiB, in
addition to the model. Capture streams response queries in chunks and never
materializes `[query, head, source, hidden]` messages.

The complete 509k-token corpus is expected to occupy roughly 75 GiB before
NPZ compression with FP32 Gram matrices; compression depends on the observed
states. Fitting loads one task and one physical split, approximately 16 GiB
for the largest QA side before array overhead. These costs are the price of
retaining every ordered layer/head/origin coordinate without a learned or
random projection.

## Required evidence

The method is kept only if held-out paired tests show that:

- registered functional routes outperform raw attention;
- real endpoints outperform rewired endpoints;
- ordered layers outperform shuffled layers;
- two-frame conditioning outperforms independent-token matching;
- MLP/readout geometry contributes beyond confidence;
- QA remains competitive with the locked route-collapse result;
- correct narrow-focus Summary and Data2txt tokens are not systematically
  treated as anomalies.

Failure is recorded as a stop result. Labels are not used to reverse a score,
select favorable heads, tune a combiner, or redefine the mechanism.
