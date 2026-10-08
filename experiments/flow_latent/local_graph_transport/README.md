# Full-coordinate local graph transport primitives

## 2026-10-08 measured reader validation

Collector and source-self-supervised token readers have now been implemented
and run. [Actual results and commands](../../../docs/SOURCE_TRANSFER_VALIDATION_20261008.md)
separate source-program performance from natural QA hallucination detection.
Three-seed first postcandidate program AUROC is 0.996755, but the frozen natural
QA ensemble is 0.677372, below the previous fixed source/route score 0.779725.
No new default detector has been selected. Candidate-conditioned compatibility
and source influence alone are not certified factual support.

The following primitive-only description records the earlier implementation
scope; its statements about absent collectors/fits are historical.

This folder implements **mechanism measurements, not a hallucination detector**.
It contains independent Torch functions and CPU science checks. It has no model
collector, natural-label reader, training, density estimator, risk score, or
reported natural AUROC.

`messages.py` accepts one layer's two attention worlds at identical token
positions. With `A` shaped `[receiver, physical_head, sender]` and `V` shaped
`[sender, physical_head, head_dim]`, it keeps both full-coordinate edge terms:

```text
content = (A_plus + A_minus)/2 * (V_plus - V_minus)
routing = (A_plus - A_minus) * (V_plus + V_minus)/2
content + routing = A_plus V_plus - A_minus V_minus  # at each edge
```

This is a finite two-world identity. It is not a Jacobian, semantic support
measure, or independent causal attribution of Q/K and V. Signed terms are
retained. GQA values must already be expanded to physical query-head identities.
The functions expect floating tensors and `W_O` in the same computation dtype
and device. For the existing FP32 observer, pass `layer.self_attn.o_proj.weight.float()`.
Keeping the complete 128-coordinate head factors plus the native `W_O` loses no
coordinates; there is no PCA, low-rank truncation, or four-coordinate projection.

`project_edges` applies the native bias-free output matrix to every edge.
`aggregate_attention_write` sums native sender messages and returns a different
4096-coordinate write for each receiver. `partition_senders` creates disjoint
source, strict-past local answer, strict-past remote answer, self, and other
prompt groups; all future senders are excluded. The self key has its own group,
including at a prompt-end receiver. `aggregate_partitions` preserves exact group
message sums without renormalizing the local neighborhood or averaging a span.

`state_conservation_error` checks
`delta_after - delta_before - delta_attention - delta_mlp`. The error is zero
for any correctly captured native update, including factual or erroneous
continuations. Using it as a hallucination score would be invalid.

For a future observer interface, input is `prompt + answer[:-1]`, and answer
token `t` is predicted at absolute position `P+t-1`. Use existing full-coordinate
hooks in `ordered_source_transport/observe.py` and
`token_backtrace/grounded_projection.py`, plus the same-position mask in
`provenance_joint_state/measure.py:block_source_reads`. Capture both worlds'
residual/attention/MLP and QKV. Reconstructing blocked-world attention must apply
the source-block mask before softmax; the existing `observed_attention` function
currently applies causal masking only. Source accessibility is different from
semantic truth. No complete native/blocked capture is claimed here.

Run CPU checks from the graph repository:

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest experiments/flow_latent/local_graph_transport/test_messages.py -q
```

The checks cover per-edge finite-difference reconstruction, separate routing and
content effects, native output projection, a nonidentity `W_O` counterexample to
scalar attention transport of hidden differences, complete causal partitions,
remote-boundary preservation, distinct receiver outputs, and the diagnostic-only
meaning of state conservation. No GPU forward or fit is required.

Verified on 2026-10-08 with the command above: **13 passed in 9.09s**, exit0.
These are CPU algebra/structure tests; native8B capture and detector effectiveness
were not tested.

## Optional scalar-field smoothing

`smooth.py` accepts an **external token-specific scalar unary** `s_t` whose
semantic direction must already be justified. It cannot turn source dependence,
an arbitrary norm, or wrongly oriented unaries into truth scores. No punctuation
span grouping or span score broadcasting is implemented.

The strictly convex objective is

```text
J(r) = 0.5 * sum_t (r_t - s_t)^2
       + lambda * sum_(j<t) w_jt * Huber_delta(r_t - r_j)
Huber_delta(d) = 0.5*d^2                 when abs(d) <= delta
                 delta*(abs(d)-delta/2) otherwise
```

The caller supplies nonnegative raw weights already converted to dimensionless
units using an independent reference scale. This module does not estimate that
scale. For raw incident weighted degree `d_t`, use
`w_jt = raw_jt / max(1, d_j, d_t)`. The unit floor prevents weak links from being
amplified: a single raw weight0.001 remains0.001. Every normalized incident
degree is at most1. This is a continuity scale budget; it does not construct
source support or factual truth. Relative weights and topology affect the
solution; a uniform rescaling can be canceled when the degree budget saturates.

The node-wise stationary equation is
`r_t-s_t + lambda*sum_j w_jt*clip(r_t-r_j,-delta,delta)=0`.
It implies the interpretable displacement cap `abs(r_t-s_t) <= lambda*delta`.
The Huber term bounds the pull across a large jump. It still expresses an
assumption that neighbors' scalar scores often agree; that assumption may fail
at a true token-level error boundary. It is not a causal error-propagation rule
and is not the exact HMM inference used by CORTEX.

`smooth_field` uses `index_add_` endpoint gradients, no dense incidence matrix.
The unary term provides strong convexity1. The normalized degree budget gives
gradient Lipschitz bound `1+2*lambda`; the fixed step is `1/(1+2*lambda)`.
Defaults `lambda=.5`, `delta=1`, tolerance`1e-8` and max_iterations`1000` are
fixed for CPU float64 software examples, not selected as natural-data optima.
The result includes the solution, objective, maximum gradient, convergence flag,
iteration count, maximum actual unary change, and `lambda*delta` bound. Failed
convergence raises an explicit error.

`smooth_field` is offline: future unaries/edges may revise earlier positions.
`filter_prefix_field` instead optimizes each prefix separately, includes only
edges with receiver`<=t`, and recomputes the raw-degree budget inside that
prefix. It emits only the then-current `r_t`. Unknown future edge weights cannot
rescale past scores; previously emitted alarms are never retrospectively changed.
For the toy unaries `[0,1,10]` on a unit-weight chain, offline output is
approximately `[.20833,1.04167,9.75]`, whereas prefix output is `[0,.75,9.75]`.

Run the additional CPU checks:

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest experiments/flow_latent/local_graph_transport/test_smooth.py -q
```

Verified on2026-10-08: **11 passed in3.73s**, exit0, no warnings. Tests cover the
zero-penalty/isolated cases, the degree budget and weak-link floor, independent
autograd stationarity, the displacement cap, denoising without equalizing all
receivers, preserving large jumps, topology/relative-weight dependence, future
leakage controls, and explicit failed-convergence behavior. No GPU forward,
fit, new natural-label read, or natural detection metric was produced.
