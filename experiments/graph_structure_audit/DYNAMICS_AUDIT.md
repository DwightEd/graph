# Cross-Origin Routing Dynamics Audit

The QA30 recovery result suggests a specific competing pattern:

- inter-token edge routing is harder to recover for hallucinated tokens;
- token-diagonal routing is slightly easier to recover;
- layer order matters more than exact endpoint identity in the current model.

The previous audit reconstructs masked values inside one multiplex graph. This
follow-up instead treats transformer depth as graph dynamics and predicts layer
`l + 1` from layer `l`.

## Hypothesis

The audit tests a **fracture--lock-in** hypothesis without assuming its direction:

1. correct generation follows a reusable cross-layer transport law;
2. hallucination may show an interaction-routing innovation that is not explained
   by the previous layer;
3. the self/diagonal state may nevertheless become more predictable;
4. response-origin messages may become more sufficient than prompt-origin
   messages after the transition.

Attention alone cannot identify the MLP's causal contribution. The next-layer
prediction residual is only a proxy for the unobserved block transformation that
includes residual, attention output, and MLP output.

## Model

For every token-pair edge and layer, the model reads the complete head vector.
Prompt and response edges use separate message functions. Each relation is
aggregated with learned weights and both first and second moments, so the model
can represent agreement and dispersion rather than only a sum.

A routing transition cell updates each token from three candidates:

```text
self state
prompt-origin message
response-origin message
```

Its softmax gates are saved for every token and layer. The updated graph state
predicts the next layer's edge weights, retained-support mask, and token
diagonal.

This follows a dynamics-first principle: interactions are useful only when they
improve prediction of the next state. Relation-specific transport is used in
place of homogeneous neighbor averaging.

## Frozen scores

The score artifact contains:

```text
edge_transition
prompt_edge_transition
response_edge_transition
diagonal_transition
support_transition
edge_state_gap
edge_state_decoupling
origin_gap
origin_fracture
message_gain
prompt_gain
response_gain
closure
layer_order_gain
head_identity_gain
endpoint_gain
```

`edge_state_gap` is the raw edge-minus-diagonal error. Because the two errors
have different scales, `edge_state_decoupling` first robustly standardizes them
within task and causal-position buckets, then subtracts them.

The artifact also stores full maps:

```text
[token, layer_transition, head]
```

for edge, prompt-edge, response-edge, diagonal, and support errors, plus
prompt/response/self transition gates.

## Required audits

1. **Transition audit**: which layer/head bands have different next-layer
   prediction errors?
2. **Origin audit**: is the effect concentrated on prompt edges, response edges,
   or their difference?
3. **Decoupling audit**: does edge fracture coexist with diagonal lock-in?
4. **Sufficiency audit**: how much do prompt-only and response-only messages
   improve next-layer prediction?
5. **Structure audit**: do message passing, exact endpoints, layer order, and
   head identity improve transition prediction?
6. **Trajectory audit**: does a transition spike precede a later response-gated
   regime around hallucination onset?

No best layer or score direction is selected with test labels. The evaluator
reports every layer/head, fixed early/middle/late bands, same-response matched
effects, and source-level bootstrap intervals.

## Run

```bash
ROOT=/path/to/attention_cache \
OUT=experiments/graph_structure_audit/outputs/dynamics_smoke \
TRAIN_LIMIT=30 TEST_LIMIT=30 EPOCHS=2 SCORE_ROUNDS=2 DEVICE=cpu \
  bash experiments/graph_structure_audit/run_dynamics.sh
```

The current masked-recovery run remains a baseline. A method claim is justified
only if the new next-layer dynamics audit shows stable relation- and depth-
specific structure beyond the aggregate recovery difference.
