# Experiment history and decisions

This document preserves the main experiments that preceded HoloRoute. Historical code was removed during repository consolidation; the numbers below are retained so rejected hypotheses are not silently repeated.

The results were produced at different stages, scopes, tasks, and sample counts. They should not be compared as one leaderboard. Labels were used for post-hoc evaluation unless otherwise stated.

## 1. Static attention and graph statistics

A token-level audit of simple attention-derived quantities produced the following AUROC values:

| Signal | AUROC |
|---|---:|
| `history_edge_fraction` | 0.6418 |
| `history_mass_fraction` | 0.5865 |
| `retained_concentration` | 0.5850 |
| `top1_share` | 0.5842 |
| `mean_edge_strength` | 0.5337 |
| `in_degree` | 0.4631 |
| `prompt_mass_fraction` | 0.4135 |
| `normalized_entropy` | 0.4086 |
| direct Lookback anomaly | 0.3899 |

**Decision.** Attention concentration and response-history structure contain weak-to-moderate signal, but no single static statistic supports the final method.

## 2. Dirichlet suitability audit

Attention role/provenance compositions were fitted with Dirichlet and logistic-normal models. The weighted Dirichlet-minus-logistic-normal validation log likelihood was consistently negative:

| Representation | Pseudocount range | Dirichlet - logistic-normal, nats |
|---|---:|---:|
| provenance | `1e-6` to `1e-3` | -5.28 to -4.94 |
| role | `1e-6` to `1e-3` | -0.69 to -0.91 |

Positive off-diagonal covariance occurred for roughly 24-25% of entries, which a standard Dirichlet cannot express naturally.

**Decision.** A Dirichlet likelihood is not an adequate core model for these attention compositions. Logistic-normal/simplex geometry remains a possible representation tool, not a detector by itself.

## 3. Source-reuse contrast and CaSH-style discrimination

The source-history discriminator learned to separate real source histories from artificial rewires, but the task saturated and did not transfer to hallucination detection. In one smoke test:

- 732 of 751 token contrast scores were exactly `-1.0`;
- only 14 distinct float32 contrast scores remained;
- positive probabilities were mostly 1.0;
- negative probabilities were near `1e-10`.

Post-hoc token results on 751 Data2txt tokens with 26 positives were near random:

| Score | AUROC | AUPRC |
|---|---:|---:|
| current negative margin | 0.5083 | 0.0335 |
| birth negative margin | 0.5080 | 0.0335 |
| dynamic negative margin | 0.4919 | 0.0321 |
| dynamic shuffled NLL | 0.5376 | 0.0377 |

**Decision.** Artificial real-vs-rewired discrimination is too easy and does not establish a hallucination mechanism. This line was stopped.

## 4. Multiplex graph recovery audit

A 30-sample run contained 6,866 response tokens and 415 hallucinated tokens. Matched recovery effects were:

| Quantity | Hallucination - correct | Interpretation |
|---|---:|---|
| recovery | +0.00280 | correct tokens slightly more recoverable |
| edge recovery | +0.00312 | correct tokens slightly more recoverable |
| diagonal recovery | -0.000212 | opposite direction |

Structural controls had very small effects:

| Control | Mean gain |
|---|---:|
| message aggregation | `2.75e-05` |
| layer order | `2.31e-04` |
| head identity | `1.68e-05` |
| exact endpoint | `4.33e-06` |
| layer-head joint structure | `-9.19e-04` |
| full channel structure | `-1.51e-03` |

**Decision.** The audit showed that some structure affects reconstruction, but effect sizes were too small and inconsistent to motivate recovery error as the detector.

## 5. Typed relation and lineage audits

A later relation audit reported:

| Relation score | AUROC | AUPRC |
|---|---:|---:|
| origin transition gap | 0.5450 | 0.0962 |
| multihop response base | 0.5263 | 0.0958 |
| inherited response base | 0.5326 | 0.0950 |
| direct role | 0.5290 | 0.0914 |
| endpoint concentration | 0.4813 | 0.0772 |
| lineage margin | 0.4515 | 0.0702 |

**Decision.** Prompt/response lineage summaries alone were weak and did not justify a hand-built grounding state.

## 6. Causal walk and variable-order audits

The strongest score in the initial causal walk audit was first-order route prediction error:

| Score | AUROC | AUPRC |
|---|---:|---:|
| `order1_error` | 0.6256 | 0.1133 |
| `response_base_mass` | 0.5881 | 0.0957 |
| `direct_role` | 0.5841 | 0.1031 |
| `order2_error` | 0.5419 | 0.0927 |
| `order2_path_gain` | 0.5224 | 0.0753 |
| `order3_path_gain` | 0.5203 | 0.0782 |
| `lock_in` | 0.5106 | 0.0756 |
| `recoupling_failure` | 0.4997 | 0.0757 |
| `anchor_js_mean` | 0.2475 | 0.0469 |

**Decision.** One-step route predictability had signal, but higher-order path gain, lock-in, recoupling, and anchor divergence did not behave as hypothesized.

## 7. Full typed-path De Bruijn detector

The full fixed-order detector initially appeared stronger:

| Score | AUROC | AUPRC |
|---|---:|---:|
| full typed-path score | 0.6603 | 0.1328 |
| rupture alone | 0.6629 | 0.1402 |
| lock-in alone | 0.5759 | 0.0889 |
| channel-score mean | 0.6178 | 0.1051 |
| absolute position | 0.6171 | 0.1129 |
| relative position | 0.6066 | 0.0949 |

However:

- Spearman(full score, absolute position) = 0.928;
- Spearman(rupture, absolute position) = 0.974.

**Decision.** The apparent gain was dominated by cumulative token position. CUSUM/rupture, prefix accumulation, and manually multiplied lock-in scores were removed from the active research line.

## 8. Attention transport and holonomy audit

The audit that directly motivated HoloRoute measured held-out structural prediction gain without hallucination labels:

| Structural control | Held-out gain | Decision |
|---|---:|---|
| depth transport | +0.05663 | core relation |
| query-set context | +0.02642 | core higher-order relation |
| relay transport | +0.00590 | weak auxiliary relation |
| exact-path rewire | +0.00569 | weak but nonzero topology evidence |
| diamond coverage | 99.56% | holonomy is measurable |

**Decision.** The active architecture is built around depth continuation and query source coalitions. Relay and holonomy remain conditional modules that must show incremental detection value.

## 9. Current HoloRoute status

HoloRoute is implemented as a neural, attention-only, unsupervised event-graph model. It has not yet been established as SOTA and no final full QA benchmark is recorded here.

Mandatory comparisons before making a graph contribution claim:

1. HoloRoute versus Flat-1024 under the same source split, masking, parameter budget, calibration, and evaluation;
2. event-only -> depth -> depth+query -> +relay -> +holonomy incremental chain;
3. real relay paths versus matched middle-token rewiring;
4. score-position Spearman and position-only baselines;
5. source-cluster bootstrap confidence intervals;
6. QA, Summary, and Data2txt results on multiple seeds.

## Historical branches represented by this record

The following development branches were used during the research process:

```text
agent/dirichlet-suitability-audit
agent/source-reuse-contrast
agent/graph-structure-audit
agent/routing-dynamics-audit
agent/causal-walk-audit
agent/causal-typed-path-debruijn-qa
agent/attention-holonomy-audit
agent/holoroute-method
agent/holoroute-readable-refactor
agent-grounding-refinement
```

Notable historical commits include:

```text
825cbec  causal typed-path De Bruijn detector
c577ae2  typed route grammar integration
f5eec43  attention holonomy audit
e0457a0  Flat-1024 all-layer control
ce54c1f  HoloRoute readable runner state before consolidation
```

The active code is now maintained only on `main`; this document is the durable record of rejected and retained ideas.
