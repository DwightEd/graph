# Token maintenance and source control

This pilot measures which source positions influence each output choice and whether a source event affects later tokens in the maintained fragment. It is an unsupervised measurement procedure, not a validated hallucination detector. Labels only enter the separate evaluation and explicitly post-hoc position diagnostics.

## Measurements

`measure.py` reads the existing 32 layers × 32 heads × answer tokens × source keys. For the actual-token versus strongest alternative logit margin F, the cached post-softmax message gate derivative is g. The derivative with respect to an attention **logit** is

    eta[j] = g[j] - A[j] * sum_k g[k].

This follows the softmax Jacobian and includes competition between source keys. The row sums to zero; a positive value means increasing that source's attention logit favors the actual token over this rival. It does not mean the source fact is true or applicable. All heads and signs are retained. Source lexical groups are automatic, position-specific, and not manual evidence/entity annotations. The two automatically selected receivers per answer are the strongest single lexical source-key sensitivity in high/low reading-continuity halves; group selection then occurs in that head, not a claimed global maximum over all source groups and heads.

`memory.py` combines the existing per-head reading maintenance with signed source-choice profiles. For every head, it maintains a full `[positive/negative, prompt-key]` distribution. With the previous reading age n and continuation c, let n_new = 1+c*n. The update is

    M_new = ((n_new - 1) * M_old + normalize([eta_positive, eta_negative])) / n_new.

Address affinity sums the signs before computing the Bhattacharyya overlap with the previous memory; signed affinity preserves them. The joint readout is continuation × address affinity. Signs are never canceled in the memory. The output candidate changes with each token, so stable signed profiles are not a proof of stable factual meaning. This implementation stores compact full-head diagnostics and dominant memory addresses; the full memory exists during sequential execution and can be reproduced from immutable A/g caches.

`native.py` changes one receiver's selected source-group attention logits by ±epsilon, normalizes the attention, and recomputes the **entire original token sequence**. All downstream attention Q/K/V, residual updates, MLP and normalization run natively; no original detached past KV is reused. Generated tokens are clamped, so this is state-mediated propagation and does not include a changed discrete sampled answer. The finite response matrix is

    Psi(source event s, target t) = (F_t(+epsilon at s) - F_t(-epsilon at s)) / (2 epsilon).

We save signed margin/logp responses and normalized final-hidden changes for every later target. This is a directional sensitivity matrix for the probed events, not the complete causal graph. Softmax logit dose .05 multiplies source-versus-other attention odds by exp(.05); different groups need not move equal attention mass. Actual attention masses are saved. Cross-layer derivatives are not a conservation ledger or independent causal contributions.

## Run

From the graph repository, reuse the existing local model and caches. One command runs all stages: `python -m experiments.span_source_control.run --output outputs/span_source_control_new` (use the research environment below). The equivalent stage commands are:

```bash
export PYTHONPATH=.:teaching/state_audit/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
research_python=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
control_output=outputs/span_source_control_new
"$research_python" -m unittest experiments.span_source_control.test_control -v
"$research_python" -m experiments.span_source_control.measure --output "$control_output"
"$research_python" -m experiments.span_source_control.finite --output "$control_output"
"$research_python" -m experiments.span_source_control.memory --output "$control_output"
"$research_python" -m experiments.span_source_control.evaluate --output "$control_output"
"$research_python" -m experiments.span_source_control.diagnostics --output "$control_output"
"$research_python" -m experiments.span_source_control.report --output "$control_output"
```

`diagnostics` adds ten exposed paired-case positions and two smaller-dose checks; its positions are **not label-free selections**. They are separate from the primary 24 automatically selected events and descriptive rankings. A/G caches come from `outputs/message_js_20260928_v1` (RAGTruth), `outputs/constraint_uptake_20260929_dense` (GSM), via `anchored_flow.edges.layer_arrays`; continuity/age come from `outputs/span_maintenance_20260929_v1`. See its manifest for original source locations. No new labels are broadcast from GSM steps to token truth. Read the report before interpreting signs or ranking numbers.

## References

- [ALTI-Logit / Explaining How Transformers Use Context to Build Predictions](https://aclanthology.org/2023.acl-long.301/): output-conditioned context contributions and the role of MLPs.
- [DecompX](https://aclanthology.org/2023.acl-long.149/): preserve source-decomposed vector messages through Transformer components. This pilot's native interventions are not DecompX's decomposition rules.
- [Information Flow Routes](https://aclanthology.org/2024.emnlp-main.965/): message paths motivate tracing where influence travels; routing importance alone does not establish factual applicability.

Results and limitations: [RESULTS_ZH.md](RESULTS_ZH.md). Detailed local token audit: `outputs/span_source_control_20260929_v1/TOKEN_AUDIT.html`.
