# Attention mechanism detection rules

Read and follow `../grounded_route/iclr/ENGINEERING_RULES.md` before changing
this directory. The rules below are mandatory for this experiment.

- Put the mechanism algorithm first. Keep one formal implementation; delete a
  superseded path instead of adding `new`, `v2`, fallback, approximate, or
  degraded variants.
- Preserve the exact frozen-model state: dynamic attention, value, matching
  layer/head `W_O` writes, and causal message deletion. Never silently replace
  it with attention-only weights, static checkpoint geometry, or scalar feature
  factories.
- Preserve the layer, query-head, source-role, and response-token axes until
  the formal low-rank projection. The source roles are exactly evidence, other
  prompt, response history, and predictor self. Do not merge history with the
  causal diagonal.
- Store role edge energy, exact per-head role write norm, per-head normalized
  source entropy, and role-wise across-head coherence. The causal branches are
  full, no-evidence, no-history, and no-evidence-history, with evidence and
  strict-history deletions applied symmetrically at response predictor queries;
  predictor self remains fixed in every branch.
- Treat the no-evidence-history branch as remaining-context support, not pure
  parametric knowledge. Preserve the full factorial evidence, response, and
  interaction effects rather than a single asymmetric difference. Condition
  its absolute target margin on full-branch confidence using fit sources before
  it enters the primary state.
- The formal primary detector is cross-fitted dynamic mechanism innovation.
  Remove position and response-length nuisance trends on fit sources, learn
  unlabeled low-rank layer/head structure, fit the response-state transition,
  and calibrate innovation on separate unlabeled sources. Static-state and
  confidence scores are the default controls, not alternative primary
  implementations. Any later head-collapse or channel-ablation diagnostic must
  remain a prespecified control and must not select the primary result.
- Assign folds by source identity. A source and all of its tokens must never
  cross fit, calibration, and held-out roles. Weight sources equally while
  fitting. Evaluate the token detector as token-micro AUROC/AP with a
  source-cluster bootstrap; aggregate post-hoc mechanism contrasts with equal
  source weight.
- Treat physical `train` and `test` cache directories only as storage shards.
  Pool them within each task before source-level fold assignment. Report QA,
  Summary, and Data2txt separately and never mix tasks into one headline.
- Keep labels sealed during collection, normalization, nuisance regression,
  representation learning, transition fitting, score construction, and
  calibration. Open labels only after all out-of-fold scores are frozen, for
  AUROC, AUPRC, matched scientific diagnostics, and bootstrap intervals.
- `collect.py` owns traversal, identity checks, serialization, and resume only.
  Reject mismatched schema, model, dtype, top-k, task/source identity, or token
  alignment. Do not reject a scientifically identical cache merely because an
  absolute filesystem path changed.
- Generate task-level population reports after both physical shards are
  pooled. Do not automatically emit per-sample reports or figures. A sample
  figure is an explicit on-demand operation selected by sample ID.
- Keep one one-click `run_all.sh` containing only the single foreground Python
  entry. Orchestration belongs in Python so exceptions retain their traceback
  and stop the run.
- Test exact message/write equations, role partitioning, symmetric deletion,
  token alignment, source-disjoint cross-fitting, label sealing, position
  controls, pooled evaluation, and the CLI contract. Avoid tests of incidental
  error wording or absolute paths.
- When another agent is explicitly asked to work here, repeat the exact file
  scope and these scientific constraints; do not assume it inherits this file.
