# Grounded Anchor Flow

This experiment tests one mechanism instead of adding another stack of scalar
attention features:

> A correct continuation may pass through influential response anchors, but
> those anchors remain reachable from the prompt/evidence.  A hallucinated
> continuation may instead be carried by a response-seeded anchor backbone that
> still has high downstream influence after prompt-seeded paths weaken.

The method combines two useful ideas without copying their attention-only
implementations.  The preplan-and-anchor work motivates separating long-range
incoming gathering from later downstream influence.  FlowTracer motivates
conditioning a causal DAG on reaching the current target and measuring the
transit nodes on all target-reaching paths.  Here both operations are performed
on the frozen model's real functional messages rather than on head-averaged raw
attention.

## 1. Exact generation edges

For response token `y_t`, the causal predictor is `q_t = P - 1 + t`.  At layer
`l`, query head `h`, and source `s <= q_t`, the observer already records

\[
m_{t,l,h,s}
=
W^O_{l,h}\left(A^{l,h}_{q_t,s}V^{l,g(h)}_s\right)
\]

and its signed first-order support for the target log probability

\[
\phi_{t,l,h,s}
=
\left\langle
\frac{\partial\log p(y_t)}{\partial o^l_{q_t}},
 m_{t,l,h,s}
\right\rangle.
\]

All source, layer, and head atoms are retained in the original functional graph.
For global flow, they are reduced only over layer and head while source and
response-target identities stay intact:

\[
C^+_{s,t}=\sum_{l,h}[\phi_{t,l,h,s}]_+.
\]

The same all-source table stores veto, raw attention, and exact residual-message
norm.  Thus the identical graph algorithm can be run on:

1. positive functional `AVWO` support — the method;
2. raw attention — information-selection control;
3. residual-message norm — information-transport control.

## 2. Target-conditioned all-path flow

Each response target column is normalized:

\[
W_{s,t}=\frac{C^+_{s,t}}{\sum_{u<t}C^+_{u,t}}.
\]

Let `B` be the response-to-response block of `W`.  It is strictly upper
triangular, so every response path is summed exactly by

\[
H=(I-B)^{-1}=I+B+B^2+\cdots.
\]

`H[i,t]` is the total product weight of all response paths from response token
`i` to target `t`.  Prompt-to-target paths are the direct prompt block followed
by `H`; no full token-by-token matrix inverse is constructed.

For targets that have prior response tokens, a fixed source prior assigns half
of its mass uniformly to the prompt and half uniformly to prior response
positions.  Conditioning that prior on reaching the target gives the
`response_seeded_path_share`.  This is a global path quantity; the ordinary
one-step `direct_response_share` is reported separately.

## 3. Anchor mediation

For source group `g`, response transit token `v`, and target `t`, all paths that
pass through `v` factorize as

\[
O_g(v\mid t)
\propto
H_g(v)H(v,t).
\]

A prior response token can also be sampled as a zero-hop starting point.  That
start contribution is subtracted before anchor mediation is measured.  The
remaining occupancy therefore represents genuine transit through `v`, not the
trivial fact that `v` is itself a response token.

The single primary score is

\[
\boxed{
\operatorname{RSAF}_t
=
\frac{\sum_{v<t}O_R(v\mid t)}
{\sum_{g\in\{E,Q,R\}}\sum_{v<t}O_g(v\mid t)}
}
\]

and is stored as `functional_response_seeded_anchor_flow`.  A high value means
that the response anchors mediating the current target are reached mainly from
prior response seeds rather than prompt seeds.  It is a candidate
self-confirming-flow signal, not a claim of semantic falsity or causal
necessity.

The full functional artifact additionally retains:

- `source_path_posterior`: evidence, other-prompt, and response path origins;
- `anchor_occupancy`: which earlier response tokens mediate each target;
- `anchor_group_occupancy`: the source group of each anchor route;
- `gather_distance`: clipped distance of incoming functional gathering;
- `future_anchor_influence`: later target-conditioned occupancy through a token;
- `dominant_anchor` and `anchor_concentration` for visualization.

`gather_distance` and `future_anchor_influence` test the gather–anchor rhythm.
They are not multiplied into the detector.

## 4. Evaluation contract

Hallucination labels are opened only after every graph and score has been saved.
QA, Summary, and Data2txt are evaluated separately with token AUROC, average
precision, and source-cluster bootstrap intervals.

The primary functional RSAF is compared on the same valid tokens against:

- attention-derived anchor flow;
- message-norm-derived anchor flow;
- response-seeded all-path share without requiring anchor mediation;
- direct response dependence;
- response position, response length, and target surprisal.

A label-free response-position adjustment is reported because ordinary
autoregressive generation naturally becomes more response-dependent over time.

The current experiment is an observed functional-flow audit.  A later
confirmatory phase must remove or keep the selected anchor backbone in a real
model re-forward, with downstream gates recomputed, before claiming causal
necessity or sufficiency.

## 5. Files

```text
flow.py      target-normalized path closure and anchor occupancy
pipeline.py  run the same graph operator on functional and control capacities
evaluate.py  freeze scores, then load labels and compare capacities
run.py       one foreground end-to-end entry
run_all.sh   one-command launcher
tests/       path, relay, anchor, control, and evaluation invariants
```

The exact functional graph remains in the sibling
`attention_mechanism_audit` package.  This package adds only the global graph
operator and its evaluation; it does not introduce a learned GNN, autoencoder,
or feature combiner.

## Run

```bash
bash experiments/grounded_anchor_flow/run_all.sh
```

Smoke run:

```bash
bash experiments/grounded_anchor_flow/run_all.sh --limit 2 --bootstrap 0
```
