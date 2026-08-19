# Causal Token Routing Basin Baseline

This experiment is a token-level, prefix-causal mechanism baseline. It replaces
the old artificial RR/PCA residual, but current pilot results do **not** show
that it is an effective hallucination detector.

## Method

For each retained off-diagonal attention edge, the observable weight is

```text
excess = max(attention - attention_floor, 0)
```

No sub-floor value is reconstructed. For response query `q`, two multiplex
operators are formed before compression:

```text
B_id[q, (layer, head, absolute_source)] = excess
B_rel[q, (layer, head, role, prompt_position/RR_lag)] = excess
```

The exact column identities are mapped through a deterministic signed
CountSketch. This approximates row inner products while keeping memory
independent of prompt length. Separate rolling Gram matrices
are computed for all routes, prompt routes, previous-response routes, and the
relative route operator. Their eigenvalues are squared singular values of
these rectangular operators; no eigenvalue of a triangular causal adjacency
matrix is presented as a graph spectrum.

Each token row contains 17 causal features:

- prompt/previous-response mass, effective-source fraction, and top-1 share;
- recent previous-response mass;
- repeated exact prompt anchor and causal run length;
- effective rank and dominant-mode share for the combined, prompt-only,
  response-only, and relative multiplex operators;
- relative-route velocity.

`TokenRoutingDetector.fit()` uses three disjoint source-group partitions:

1. fit: nuisance regression, robust residual scale, and VAR(1) dynamics;
2. component calibration: empirical tails for state, transition, and
   preregistered concentration/graph directions;
3. final calibration: conformal calibration of the component maximum.

The four outputs are `state_novelty`, `transition_surprise`,
`basin_commitment`, and `smoothed_commitment`. The last is explicitly an EMA,
not evidence of a dynamical-system attractor. Training is label-free, so the
fit set is an unlabeled mixture under a majority-normal assumption, not a
known-clean normal set.

The cache alignment is `post_token_query_at_same_position`: score `s_t` is
available after token `t` is emitted. Evaluation reports both contemporaneous
`s_t -> y_t` detection and `s_t -> y_(t+1,t+2,t+4)` forecasting. Only the latter
is a pre-emission warning test for future tokens.

## One-command run

PowerShell:

```powershell
cd D:\projects\python_projects\research\graph
powershell -NoProfile -ExecutionPolicy Bypass `
  -File experiments\token_routing_basin\run.ps1 `
  -Root D:\projects\python_projects\research\data\RAGTruth\llama31_8b `
  -Python "D:\Apps\Program Files\anaconda3\python.exe" `
  -Device cpu
```

For a smoke test add `-Limit 16`. `Limit` counts samples, not tokens. The
script resolves the repository root, checks both manifests, runs focused tests,
then runs fit, score, and evaluation. Add `-SkipTests` only when the focused
tests have already passed.

Git Bash/Linux:

```bash
ROOT=/path/to/RAGTruth/llama31_8b DEVICE=cpu LIMIT=16 \
  bash /path/to/graph/experiments/token_routing_basin/run.sh
```

CUDA is optional. The CountSketch version is intended to run locally on CPU.
The output contains:

```text
reference.npz          fitted/calibrated unlabeled reference
test_scores.npz        frozen token scores, features, source audit, provenance
evaluation/report.json current-token, forecast, onset, coverage, and group metrics
logs/                  one log per stage
```

## Verified status

The current code passes synthetic tests for label isolation, future-prefix
invariance, floor censoring, invalid-gap handling, artifact round-trip, and
head/source wiring sensitivity. On the local 64-sample development run:

- fit extraction: about 34 seconds; score extraction: about 42 seconds;
- 12,324/12,324 token rows valid;
- AUROC 0.473, AUPRC 0.055 at prevalence 0.062;
- horizon 1/2/4 AUROC: 0.475/0.473/0.468;
- recall 0.031 with 62.3 false alerts per 1,000 normal tokens;
- the strongest individual feature was `prompt_top1_share` (AUROC 0.620,
  AUPRC 0.096), while the best calibrated component was transition surprise
  at AUROC 0.503.

This is a negative pilot, not evidence for a unified unlabeled basin score. A useful next
step is a source-disjoint supervised causal sequence model over the frozen
multiplex states, evaluated against controls-only, the old PCA baseline, and
this unlabeled baseline. Any paper claim also requires a fresh source/task/model
holdout because the current RAGTruth test labels informed method design.

Closest references: [Lookback Lens (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.84/),
[Multi-View Attention Features](https://arxiv.org/abs/2504.04335),
[LapEigvals (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1239/), and
[TOHA (ACL 2026)](https://aclanthology.org/2026.acl-long.704/).
