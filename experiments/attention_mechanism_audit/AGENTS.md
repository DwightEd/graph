# Attention mechanism audit rules

Read and follow `../grounded_route/iclr/ENGINEERING_RULES.md` before changing
this directory. The rules below are mandatory for this experiment.

## Formal method

- Maintain one implementation: the **Evidence-Adoption Incidence Graph**. Do
  not add `v2`, `new`, fallback detectors, feature registries, a second graph
  encoder, or another meaning for the primary score.
- A target response token `t` is evaluated at its causal predictor
  `q_t = response_start - 1 + t`. Preserve source token, target token, layer,
  and query-head identity until the model's real `W_O` merge.
- The source-head incidence ledger is the dynamic magnitude

      e[l,h,t,s] = A[l,h,q_t,s] ||W_O[l,h] V[l,g(h),s]||_2

  with the matching layer/head output block and explicit GQA mapping. Do not
  replace this with attention alone, checkpoint geometry, or a head average.
  This unsigned scalar locates message-path incidence but does not establish
  support or opposition. Preserve direction and cancellation in exact role net
  writes: sum `A V` within each head, project through its matching `W_O` block,
  then merge heads by their real vector sum in the shared residual stream.
- Partition sources into exactly four disjoint roles: direct evidence, other
  prompt, strict response history, and predictor self. Predictor self is never
  silently included in strict history.
- Compute separate head-resolved adaptive covers for evidence and strict
  response history. Each cover is the smallest sources carrying at least `0.8`
  of that `(layer, target, head, role)` incidence magnitude. Store the required
  cover size, as many leading source indices and magnitudes as the compact slot
  limit permits, and all remaining mass. Other prompt and predictor self remain
  in the dense role statistics but do not receive sparse covers. Unstored mass
  is not zero.
- Route contraction is an onset-local explanatory control. Preserve effective
  head-source support contraction and total route-mass contraction as separate
  audits; do not add or multiply them. Measure cover size, anchors, and route
  rank without averaging heads first. Do not claim that every hallucinated span
  must remain globally locked to a small prompt set.
- A contraction value is defined only when a current route and a prior route
  are comparable. Exclude undefined rows from fitting and calibration, assign
  the fixed neutral percentile `0.5` when a common evaluation pool requires a
  score, and report comparable token/source counts.

## Causal endpoint and pathway audit

- Keep exactly four model branches: `full`, `no_evidence`, `no_history`, and
  `no_evidence_history`. Evidence and strict-history writes are deleted at the
  aligned response predictor queries throughout replay, so their effects
  propagate through later layers and response KV states. Predictor self is
  fixed in all branches.
- The fixed raw endpoint is

      unsupported_history_takeover
        = (no_evidence - no_evidence_history) - (full - no_evidence)

  where each term is the target-token log probability. High values mean that
  strict history remains supportive after direct evidence-token attention-value
  writes are cut at response predictor queries, while those direct writes
  contribute little in the full computation. This cut does not erase evidence:
  other prompt, predictor residual/self, MLP, and evidence already propagated
  into prompt or response states remain. This is a real frozen-model
  intervention endpoint, not an attention rollout or a learned feature sum.
  Report the raw quantity as a mechanism audit; do not confuse it with the
  final detector scale.
- The primary detector named `unsupported_history_takeover` is the out-of-fold
  percentile of that raw endpoint after source-disjoint nuisance regression.
  Fit only relative position, squared relative position, and separate
  prompt/evidence/response lengths on fit sources; use distinct calibration
  sources for the empirical CDF and evaluate on held-out sources. Do not put
  confidence or target margin in this regression. Keep confidence as an
  independent control.
- Only response indices `t >= 2` have strict history separate from predictor
  self. Mark earlier tokens `detection_valid=False`, exclude them from fitting,
  calibration, and detection metrics, and report `evaluated_tokens` explicitly.
- At every layer and response predictor, retain the 2x2 factorial effects of
  evidence, history, and their interaction for attention output, MLP output,
  and residual state. For any aligned quantity `x`, use

      evidence    = 0.5 * ((x_full - x_noE) + (x_noH - x_noEH))
      history     = 0.5 * ((x_full - x_noH) + (x_noE - x_noEH))
      interaction = x_full - x_noE - x_noH + x_noEH

  These are finite differences under the named deletions, not an additive
  decomposition of all information in the hidden state.
- Treat the serialized `RouteGraph` as layer-local head-source incidence, not
  cross-layer ancestry. Never connect equal-numbered heads across layers or use
  the graph alone to decide whether a history source is evidence-grounded.
  MLP and branch pathway quantities remain separate finite-difference audits.
- MLP diagnostics test whether an evidence/history branch difference is
  amplified, cancelled, or rotated between attention output and layer output.
  Never call an MLP norm or the `no_evidence_history` margin parametric
  knowledge.
- A pure parametric-attention bias claim requires controlled prior pairs with
  matched context and known model priors or counterfactual facts. RAGTruth alone
  identifies evidence bypass/history takeover, not pure parametric bias.

## Data, evaluation, and files

- Keep labels sealed through collection, graph construction, score
  construction, nuisance fitting, and calibration. Open them only for final
  AUROC/AP, source-cluster bootstrap intervals, and named post-hoc audits.
- Require every nuisance fit to have enough rows and full column rank. Report
  sklearn average precision as `AP`, not `AUPRC`, and report the samples and
  sources that actually enter detection.
- Compare observer, dtype, capture spec, and source identities before pooling
  shards. Journal full token/evidence-mask digests and reject changed or
  size-corrupted artifacts before resume or label reconnection.
- Pool physical train/test cache directories within each task, then split by
  `source_id`. Report QA, Summary, and Data2txt separately. Control prompt,
  evidence, response length, and response position explicitly.
- `capture.py` owns frozen-model incidences, the four interventions, and
  attention/MLP/residual pathway traces. `collect.py` only traverses samples,
  checks identity/alignment at the input boundary, serializes the current
  schema, and resumes completed samples. `graph.py` constructs the incidence
  graph. `detect.py` implements the one primary endpoint and controls.
  `evaluate.py` opens labels and produces reports; its `audit` section is
  post-hoc explanation, not another detector.
- A changed capture schema requires fresh recapture; never silently adapt an
  earlier prompt-carrier artifact. Keep one output tree and one foreground
  entry, `python -m experiments.attention_mechanism_audit.run all`.
- Test predictor/target alignment, role partitioning, per-head incidence and
  real `W_O` merge, 0.8 cover plus remainder, the four branch equations,
  attention/MLP/residual factorial effects, source-disjoint evaluation, and
  label sealing. Synthetic correctness does not establish empirical value.
