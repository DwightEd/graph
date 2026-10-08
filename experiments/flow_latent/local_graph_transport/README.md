# Full-coordinate local graph transport primitives

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
