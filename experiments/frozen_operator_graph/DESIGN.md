# Design Plan and Scientific Invariants

## Research question

Can a response token be represented from the frozen LLM's real internal message
transport—rather than from a separately trained GNN—so that an independent
label-free or post-hoc evaluator can test grounding loss, response self-reliance,
operator switching, message conflict, and stable response lock-in?

## Non-negotiable constraints

1. Hallucination/correctness labels are never inputs to graph construction.
2. No trainable graph message function or node-update function is introduced.
3. No mean-head attention graph replaces the original multi-head code.
4. No hidden-state surrogate replaces actual `V` and `W_O` transport.
5. Missing prompt-query rows are never fabricated from the sparse cache.
6. A fresh replay must numerically bind to the exact cached checkpoint/sample.
7. Every causal source is consumed. In quotient mode, omitted identities are
   represented by an exact role remainder rather than zero-filled or discarded.
8. Every persisted feature has a stable name and a checked tensor contract.
9. Every sample must pass message and route conservation before it is saved.
10. Evaluation labels may only be opened after artifact bytes are frozen.
11. The originating raw response JSONL is hash-bound as provenance but is not reparsed into graph features.

## Graph definition

### Nodes

The downstream evaluation nodes are response tokens. Their content state is the
checkpoint's actual final normalized hidden state. Their mechanism state retains
all layer/head route channels plus all layer residual-dynamics channels.

### Multiplex edges

For every layer and causal token pair `(source, response_target)`, one edge
stores the complete cross-head attention code. A scalar head average is never
used as the edge identity.

### Exact edge message

The source/head context is `attention * value_state`. The checkpoint `o_proj`
blocks transform the concatenated head context into the residual-stream message.
The code verifies that summing these messages reproduces the captured attention
update.

### Source roles

Prompt, previous response, and diagonal self are mutually exclusive causal
roles. This role partition supports grounding and self-reliance analysis without
using labels or hand-selected evidence spans.

## Construction stages

### Stage A — exact frozen capture

Teacher-force the cached token sequence through the checkpoint with eager
attention. Capture attention probabilities, values, `o_proj` input, attention
output, residual states, normalization outputs, MLP updates, layer outputs, and
final hidden states.

### Stage B — cache binding

Compare all retained sparse endpoints and all exact diagonal entries. Verify
that every omitted causal off-diagonal entry is at or below the declared cache
floor. Abort on mismatch.

### Stage C — operator graph

Split every route and vector message into prompt/history/self roles. Persist all
edges by default. Optional quotient mode chooses explicit identities by both
route mass and value energy, then conserves every remaining source in a role
remainder.

### Stage D — node encoding

Concatenate raw final hidden state, the complete `[layer, head]` mechanism tensor,
layer residual features, and deterministic temporal dynamics. No learned or
linear dimensionality reduction is applied.

### Stage E — freeze artifacts

Atomically save per-sample files, hashes, a split index, feature-contract hash,
construction configuration, checkpoint identity, and label/fallback audit.

## Evaluation handoff

The package deliberately does not define the user's anomaly score. It exposes:

- exact graph topology and cross-head edge codes;
- exact role-remainder statistics when quotient mode is selected;
- response-token node representations;
- unfused layer/head mechanism tensors;
- conservation and provenance audits.

The existing evaluator can compare graph quality, token separability, mechanism
signals, clustering, or unsupervised anomaly models after opening labels in a
separate evaluation phase.

## Required controls for the later evaluator

1. Full operator graph versus attention-mass-only graph.
2. Exact `A V W_O` messages versus hidden-state-weighted attention surrogate.
3. Original head/operator binding versus a layer-wise head permutation control.
4. Prompt/history/self decomposition versus role-shuffled controls.
5. Full graph versus exact quotient at several retention levels.
6. Position- and response-length-conditioned evaluation.
7. Source-grouped splits and label access only after artifact hashing.

## Acceptance criteria

A construction run is accepted only when:

- cache binding is verified;
- no label interface was consumed;
- `A V` reconstructs the captured `o_proj` input within tolerance;
- `W_O(A V)+b` reconstructs the captured attention output within tolerance;
- prompt/history/self contexts sum to the total context;
- explicit plus remainder contexts and route mass are conserved;
- all feature tensors are finite and match the feature-name contract;
- all selected samples complete and a final manifest is written.
