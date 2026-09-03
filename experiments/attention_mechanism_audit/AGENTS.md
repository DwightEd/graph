# Attention mechanism audit rules

Read and follow `../grounded_route/iclr/ENGINEERING_RULES.md` before changing
this directory. The rules below define the current experiment.

## Method

- Maintain one implementation of the **dual-register attention mechanism
  audit with shortcut-route validation**. Do not restore the retired incidence-
  graph framing or make `unsupported_history_takeover` or a new shortcut
  statistic the fixed primary detector before full-data evaluation.
- Let `P = response_start`. Response target `t` is evaluated at its causal
  predictor `q_t = P - 1 + t`. Preserve target, source, layer, and query-head
  identity throughout capture. Predictor self is distinct from strict response
  history.
- Replay exactly four aligned branches: `F` (`full`), `noE` (`no_evidence`),
  `noH` (`no_history`), and `noEH` (`no_evidence_history`). At response
  predictor queries, `noE` deletes direct-evidence attention writes, `noH`
  deletes strict-history attention writes, and `noEH` deletes both. The
  deletions propagate through later layers and response KV states; they do not
  erase evidence already present elsewhere in the computation.
- Form two branch finite-difference registers for every aligned hidden-state
  quantity:

      evidence-adoption register:  P_reg = F - noE
      autonomous-history register: R_reg = noE - noEH

  The symbols `P_reg` and `R_reg` are register names; `P = response_start` in
  the predictor equation above.
- At every decoder layer, capture the exact residual identity

      output = input + attention_write + mlp_write

  and apply the same finite differences to all four terms. Store register
  norms, the attention-plus-MLP step, MLP alignment, interaction norms, and the
  numerical closure error. Do not hide the MLP inside an attention route.
- For each response token and register, construct the cross-layer Gram matrix
  of the finite-difference steps. The label-free candidate
  `provenance_takeover` is the raw log spectral quotient of the leading Gram
  eigenvalues for `R_reg` over `P_reg`. It mixes step energy and cross-layer
  alignment and remains a graph candidate, not the primary baseline or a
  complete ancestry reconstruction.

## Residual-message routes

- Compute branch-difference attention routes with the actual post-intervention
  attention coefficients, values, GQA mapping, and matching query-head block
  of `W_O`. Do not replace them with attention weights, head averages, or
  checkpoint-only geometry.
- Globally rank head-source edges within each `(layer, target, register)` by
  absolute signed contribution to the complete register attention write. Keep
  the adaptive cover subject to `top_k`; every saved edge retains its source,
  head, nonnegative `delta(A V W_O)` magnitude, and signed contribution.
- Decompose signed branch-difference contributions into intervention `root`,
  non-root `carrier = mean(A) delta(V)`, and non-root
  `gate = delta(A) mean(V)`. Their explicit edges and unresolved tails must
  reconstruct the complete signed contribution. This midpoint split is an
  exact algebraic identity but not a unique causal decomposition; `gate`
  includes Q, K, and softmax-competition changes.
- Partition sources into exactly four disjoint roles: `evidence`,
  `other_prompt`, `response_history`, and `predictor_self`. Preserve dense
  head/register/role totals.
- Preserve the omitted sum of per-edge norms and the omitted signed
  contribution as row tails. The magnitude tail is not the norm of the omitted
  net vector. A tail has no known source or head endpoint and must never be
  expanded into invented edges. Connect explicit routes to their aligned
  attention-stage nodes, but keep the sparse view distinct from a complete
  cross-layer ancestry graph.
- Capture, for every `(layer,target)`, and fail if the exact midpoint relay
  decomposition does not close. Store the residual-space Gram of the full
  strict-history write, direct-evidence write, evidence-conditioned history
  carrier and gate writes, autonomous-history write, and an adjacent-endpoint
  rewiring control. Preserve the matching per-head Gram. These are the raw
  audit objects; scalar completion scores are derived later and never replace
  them.
- The route-completion hypothesis is specific: a supported response relay has
  a full history write explained by direct evidence plus evidence-conditioned
  carrier/gate writes. The fixed shortcut candidate is route incompleteness
  multiplied by the positive signed contribution of autonomous history to the
  full-history direction; do not residualize two algebraically identical
  remainder vectors. Compare observed and adjacent-rewired candidates on their
  common fixed token set and source-bootstrap their AUROC/AP difference.
- Keep the established full-prompt collapse measurements only as a historical
  QA audit. They are not the current detector and must not be generalized from
  the earlier QA result to Summary or Data2txt.

## Scores, labels, and claims

- Keep the following fixed raw log-probability controls:

      evidence_bypass              = noE - F
      symmetric_route_capture      = noE - noH
      unsupported_history_takeover = 2 * noE - F - noEH
      confidence                   = -F

  `unsupported_history_takeover` is a control, not the primary method.
  Keep `evidence_bypass` as the locked primary raw baseline; a new graph
  statistic remains a candidate until independent full-data evaluation.
- Do not fit nuisance regressions, cross-fit a detector, calibrate an empirical
  CDF, percentile-transform scores, or flip directions after reading labels.
  Raw equations and their validity masks are frozen before evaluation.
- Strict autonomous history is unavailable for the first two response targets.
  Mark history-dependent scores invalid there and report the tokens and sources
  that actually enter evaluation.
- Keep hallucination labels sealed during capture, graph construction, and
  score construction. Freeze every score-specific validity mask and use their
  intersection for the printed comparison. Open labels only in final task-specific evaluation and
  explicitly named post-hoc audits. Report QA, Summary, and Data2txt separately
  with token-micro AUROC, sklearn average precision (`AP`), and source-cluster
  bootstrap intervals.
- AVWO magnitude tracing is prior art in the line from Kobayashi et al. through
  ALTI and IFR. Limit any novelty claim to the branch-defined distinction
  between evidence-descended and autonomous-history state, plus explicit MLP
  finite-difference registers.
- Do not claim complete causal flow, full token ancestry, or identification of
  parametric knowledge. The four replay branches identify effects only under
  their named attention-write deletions.

## Files and execution

- `capture.py` owns the frozen-model replay, exact layer traces, Gram state, and
  signed residual-message routes. `collect.py` owns traversal, alignment,
  serialization, and resume. `graph.py` exposes the sparse dual-register view.
  `detect.py` computes only fixed raw scores. `evaluate.py` is the label-opening
  boundary.
- Schema 10 requires a fresh capture under `shortcut_route_state_v10/train/`
  and `shortcut_route_state_v10/test/`. Do not adapt old artifacts in place and
  do not delete them; historical output directories remain preserved. Write v10
  task reports under `shortcut_route_v10/{qa,summary,data2txt}/` instead of
  replacing earlier reports.
- Use the single foreground entry:

      bash experiments/attention_mechanism_audit/run_all.sh

  It runs `python -m experiments.attention_mechanism_audit.run all` once and
  evaluates the three tasks separately. Do not launch hidden or per-task
  background jobs.
- Test predictor alignment, branch removals, role partitioning, layer closure,
  register Gram construction, shortcut-route Gram geometry, endpoint rewiring,
  global sparse selection and exact tails, raw score equations, validity masks,
  and label sealing. Synthetic correctness is not empirical validation.
