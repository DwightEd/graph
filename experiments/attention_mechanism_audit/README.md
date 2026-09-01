# Teacher-forced mechanism-state detector

This experiment runs one frozen Llama-3.1-8B observer over every cached
RAGTruth response. `train` and `test` are physical cache shards, not supervised
splits. They are pooled within each task; QA, Summary, and Data2txt are fitted,
scored, and reported separately.

The current method deepens the earlier attention-drift, dispersion, and bias
audit. That audit compared a small set of all-token means and confidence
intervals after labels were opened. Those contrasts remain useful scientific
diagnostics, but they discard which layer and head changed, whether heads
cancelled one another, and whether the change was an unexpected transition for
that response position. The detector therefore preserves the mechanism state
first and models its dynamics before labels are read.

## Mechanism state

For response predictor `t`, visible source `s`, layer `l`, and query head `h`,
the functional edge energy is

```text
e[l,t,h,s] = A[l,h,q_t,s] ||W_O[l,h] V[l,g(h),s]||_2
```

`q_t` predicts response token `t`, and `g(h)` maps a query head to its GQA KV
head. The implementation uses the sample's dynamic attention and value tensors
and the matching layer/head block of `W_O`; it is not an attention-only proxy.

Sources are separated into four roles:

| role | meaning |
|---|---|
| `evidence` | passages, source article, or structured record |
| `other_prompt` | system, question, instruction, and template context |
| `response_history` | earlier response tokens visible to the predictor |
| `predictor_self` | the causal-diagonal predictor token |

The saved state retains, for every layer, response token, and head:

- role edge energy;
- pure attention mass for the same four roles;
- the norm of each role's exact per-head `W_O` write;
- normalized source entropy;
- across-head write coherence for every role;
- the strongest functional message sources.

Keeping `predictor_self` separate prevents the causal diagonal from being
mistaken for response-history drift. Keeping the head axis exposes dispersed
heads, role disagreement, and cancellation that all-layer scalar averages
cannot represent.

## Symmetric causal branches

Four teacher-forced branches are run at response predictor queries only:

| branch | attention-value writes deleted after softmax |
|---|---|
| `full` | none |
| `no_evidence` | evidence writes |
| `no_history` | strict response-history writes; predictor self is retained |
| `no_evidence_history` | both groups |

Deleted mass is not renormalized. The altered state continues through the real
output projection, residual path, MLP, later layers, and LM head. The four
log-probabilities produce symmetric factorial evidence, history, and
interaction channels. The no-evidence-history branch is also retained as a
diagnostic of remaining-context support, not described as pure parametric
knowledge: prompt context, predictor self, residual state, and MLP computation
still remain.

## Cross-fitted dynamic detector

The detector consumes the complete layer/head/role state rather than choosing
weights for a short feature list. Sources are assigned deterministically to
cross-fitting folds, so every reported token is scored by a model that did not
fit or calibrate on that source. Within every task and fold it:

1. removes response-position and response-length nuisance trends using only
   fit sources;
2. learns low-rank layer/head structure from the unlabeled mechanism tensor;
3. models the transition from the preceding response state;
4. scores the current innovation, including static state and confidence
   controls;
5. calibrates the innovation within position and length cells using separate
   unlabeled calibration sources.

The remaining-context target margin is conditioned on full-branch log
probability and margin using fit sources. It therefore measures preference left
after route deletion beyond ordinary token confidence; it is not claimed to be
pure parametric knowledge.

This makes attention drift a change in the multivariate mechanism trajectory,
not merely `response_share - evidence_share`. Dispersion is represented per
head together with head disagreement and write coherence. Bias is represented
by the symmetric causal factorial channels, rather than by a single branch
margin with an overly strong “parameter knowledge” interpretation.

Hallucination labels remain sealed during collection, nuisance removal,
representation learning, transition fitting, score construction, and
calibration. They are opened only after all out-of-fold token scores have been
frozen, to compute post-hoc AUROC, AUPRC, matched mechanism contrasts, and
source-level bootstrap intervals.

Detection AUROC and AUPRC use the token-micro estimand because the target is a
token detector; their intervals resample whole sources so within-response token
dependence is not treated as independence. The explanatory matched contrasts
use a different, explicitly source-equal estimand.

## Detailed post-hoc audit

The earlier mean/CI audit is retained as an interpretation layer rather than
used to fit the detector. For every token it now reports the attention, edge,
and exact-write shares of all four source roles; attention-to-edge and
edge-to-write gains; non-self evidence/history route balance and velocity;
per-head source dispersion and cross-head role disagreement; within-head and
across-head write coherence; early, late, and early-to-late layer changes; and
the symmetric evidence, history, and interaction effects. Hallucinated versus
correct contrasts are matched within response by absolute and relative
position, weighted equally by source, and accompanied by source-level bootstrap
intervals. A focused onset difference-in-differences audit checks whether the
mechanism changes locally when a hallucinated span begins.

These label-opened contrasts explain a frozen score; they never choose its
features, signs, ranks, nuisance model, transition, or calibration.

## Code boundaries

- `capture.py` computes the dynamic mechanism tensor and causal branches.
- `collect.py` only traverses caches, records identities, serializes samples,
  and resumes collection. It contains no detector or label logic.
- `detect.py` performs label-free source-level cross-fitting and freezes the
  joint-state detector scores.
- `evaluate.py` opens labels only after score freezing, then writes task reports
  and detailed post-hoc audits.
- `run.py` is the single foreground command-line entry.

Schema 5 writes to a new `mechanism_state/` directory. Version-4 `traces/`
remain untouched and cannot be silently reused for the richer tensor.

## Run all data

From the repository root, run:

```bash
bash experiments/attention_mechanism_audit/run_all.sh
```

The shell file contains only:

```bash
python -m experiments.attention_mechanism_audit.run all
```

The default output root is
`experiments/attention_mechanism_audit/outputs/Meta-Llama-3.1-8B-Instruct/`.
`mechanism_state/train/` and `mechanism_state/test/` are shared resumable
collections. Each of `qa/`, `summary/`, and `data2txt/` contains its own
`report.json`, `token_scores.npz`, fitted cross-fold metadata, and population
`figures/`. The command runs in the foreground and stops on the first error.

Optional path overrides are listed by:

```bash
python -m experiments.attention_mechanism_audit.run all --help
```

## Plot one sample

Render a saved sample by ID without replaying the model:

```bash
python -m experiments.attention_mechanism_audit.run plot-sample \
  --input /path/to/output/mechanism_state/train \
  --input /path/to/output/mechanism_state/test \
  --sample-id SAMPLE_ID \
  --output sample.png
```

The detector describes how the frozen observer processes teacher-forced tokens.
If a different model generated the cached answer, it does not reconstruct that
generator's internal mechanism.

Run focused tests with:

```bash
pytest -q experiments/attention_mechanism_audit/tests
```
