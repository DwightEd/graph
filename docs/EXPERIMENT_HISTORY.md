# Experiment history and decisions

This document preserves the main experiments that preceded the current
held-out typed endpoint-recovery experiment. Historical code was removed during
repository consolidation; the numbers below are retained so rejected
hypotheses are not silently repeated.

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

## 9. Historical HoloRoute status

HoloRoute was implemented as a neural, attention-only event-graph model. It
did not establish a final full QA benchmark and is no longer the active
experiment. Its masked reconstruction result remains a baseline rather than
evidence that hallucinations are structurally anomalous.

## 10. P-Cut closure falsification

P-Cut compared the same token under full, no-prompt and no-response-closed
graph views. The preregistered claim was that a high closure score indicates a
token that can run on response history without prompt evidence.

The frozen full-QA evaluation contained 30,470 tokens:

| Quantity | AUROC | AUPRC |
|---|---:|---:|
| raw closure | 0.4209 | 0.0734 |
| conditionally calibrated closure | 0.4210 | 0.0730 |
| absolute position | 0.6171 | 0.1129 |
| relative position | 0.6066 | 0.0949 |

**Decision.** The preregistered direction failed. Test-label direction
flipping and renaming are forbidden. Route cuts and closure are not part of
the current implementation.

## 11. GroundedRoute and directed route hypergraph

GroundedRoute consolidated the graph contract around token nodes, typed
`(source,target,layer,head,weight)` edges, separate diagonal mass and explicit
native unresolved mass. The directed route hypergraph then made each
`(target,layer,head)` row an explicit source-to-hyperedge-to-target computation
and added deterministic ordered P/R/U provenance, incidence/head corruption and
an ordered exact-endpoint layout target.

The averaged first-order GCN remains the strong graph baseline. On the same
149-sample QA scale, it learned 64D frozen node embeddings with label-free
endpoint prediction and obtained:

| GCN reader | AUROC | AUPRC |
|---|---:|---:|
| PCA-kNN | 0.6982 | 0.1617 |
| Isolation Forest | 0.6362 | 0.1411 |
| Autoencoder | 0.5649 | 0.0935 |
| Linear supervised readability probe | 0.7865 | 0.2999 |
| MLP supervised readability probe | 0.7785 | 0.2760 |

These GCN numbers are a representation baseline, not evidence that its exact
topology is causal: the GCN includes role/position features and averages heads
and layers before propagation.

## 12. Ordered endpoint-layout negative result

The directed hypergraph run composed attention transitions in Transformer layer
order and jointly trained local clean-row reconstruction, ordered P/R/U flow,
and ordered exact-endpoint layout reconstruction. Its frozen configuration was:

```text
rows_per_graph                  256
layout_rows_per_graph            32
layout_rows_per_batch             64
layout_min_mass               0.0001
layout_max_elements          8000000
layout_max_work_elements   250000000
layout_order                  ordered
incidence_dropout                0.15
head_dropout                     0.05
flow_weight                       0.5
layout_weight                    0.25
variance_weight                  0.05
epochs                              8
learning_rate                   0.001
weight_decay                   0.0001
seed                         20260827
```

The best validation loss was `1.9461063737791728` at epoch 5. The total-loss
trajectory was:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 2.188632 | 2.079341 |
| 2 | 2.064419 | 2.006295 |
| 3 | 2.020466 | 2.013727 |
| 4 | 1.995902 | 1.975025 |
| 5 | 1.982798 | **1.946106** |
| 6 | 1.971470 | 1.951295 |
| 7 | 1.964670 | 1.947767 |
| 8 | 1.962858 | 1.955408 |

At the selected epoch, validation components were row `1.386462`, flow
`0.583809`, layout `0.917132`, layout-sink `0.492043`, layout-self `0.000049`,
layout-external `0.425040`, variance `0.769134`, and both layout self/external
coverage values were `0.050505`. Thus optimization progressed, but most selected
rows contributed little self-conditional supervision and the flow loss changed
very little across training.

The frozen representation evaluation used 149 samples, 30,619 tokens, 2,307
positive tokens and prevalence `0.07534537378751756`. The exported embedding was
64D. PCA used a `(32, 64)` basis and `(20,000, 32)` calibration reference,
reported `collapsed=false`, used 20 neighbors, and had whitening scales from
`0.0129843` to `5.7181654`.

The preserved output directory is:

```text
/share/home/tm902089733300000/a903202310/lys/research/graph/experiments/directed_route_hypergraph/outputs/qa/ordered_layout_real_lr32_fw0.5_lw0.25_rw1.0_seed20260827
```

The exact training command and training commit were not captured with the run
and are therefore recorded as unknown rather than reconstructed from memory.
The representation evaluator was invoked through
`experiments/grounded_route/evaluation/run.sh` with 5 folds, 20 epochs, 1,000
bootstrap replicates and seeds `20260825 20260826 20260827`.

| Unsupervised reader | AUROC | AUPRC | AUPRC lift |
|---|---:|---:|---:|
| Autoencoder | 0.540130 | 0.081410 | 1.080 |
| Deep SVDD | 0.526438 | 0.078837 | 1.046 |
| Isolation Forest | 0.550779 | 0.082963 | 1.101 |
| LOF | 0.518836 | 0.076342 | 1.013 |
| PCA-kNN | 0.548162 | 0.083510 | 1.108 |

| Position/readability diagnostic | AUROC | AUPRC |
|---|---:|---:|
| Absolute position | 0.617076 | 0.112859 |
| Relative position | 0.606555 | 0.094919 |
| Linear probe on directed-hypergraph embedding | 0.597574 | 0.096996 |
| Linear probe on position | 0.606064 | 0.101359 |
| MLP probe on directed-hypergraph embedding | 0.552147 | 0.079619 |
| MLP probe on position | 0.568792 | 0.092920 |

For the readers available in both reports, the direct comparison is:

| Reader | Ordered-layout hypergraph | First-order GCN | AUROC / AUPRC gap |
|---|---:|---:|---:|
| PCA-kNN | 0.5482 / 0.0835 | 0.6982 / 0.1617 | -0.1500 / -0.0782 |
| Isolation Forest | 0.5508 / 0.0830 | 0.6362 / 0.1411 | -0.0854 / -0.0581 |
| Autoencoder | 0.5401 / 0.0814 | 0.5649 / 0.0935 | -0.0248 / -0.0121 |
| Linear probe | 0.5976 / 0.0970 | 0.7865 / 0.2999 | -0.1889 / -0.2029 |
| MLP probe | 0.5521 / 0.0796 | 0.7785 / 0.2760 | -0.2264 / -0.1964 |

**Decision.** This is a negative representation result. Training loss cannot
be used as a hallucination score, every unsupervised reader is close to chance,
position is stronger, and the supervised readability ceiling is far below the
GCN ceiling. The run does not isolate layer order because several nested
objectives were enabled together. It therefore rejects the current joint
clean-support objective, not all possible uses of layer order or exact typed
graphs.

## 13. New active line: leakage-free typed endpoint recovery

The current implementation reuses GroundedRoute's existing
`matched_negative_edges` sampler. For every sampled positive
`(source,target,layer,head)` edge, the student graph is forced to hide that exact
edge. The final node latent must rank the clean positive source above causal
non-edges matched on source role and logarithmic lag within the same typed row.
This removes the previous possibility that a scored positive support remains
visible in the student input.

Artificially hidden retained mass is no longer added to native `unresolved`.
The public graph continues to represent only cache censoring through
`unresolved`; a separate student-only `masked_mass` channel represents known
training holdouts. Clean flow/layout teachers, when enabled as ablations, always
read the uncorrupted graph.

The deterministic 64D latent is the mandatory fair baseline. A variational
posterior is optional and explicitly configured. With hidden dimension 64,
`mean` export remains 64D and `mean_logvar` export is 128D. Training may sample
from the posterior, but evaluation exports deterministically. Posterior
variance measures dispersion under this corruption model and Gaussian
bottleneck; it is not factual uncertainty or a hallucination score.

**Decision.** Run endpoint-only deterministic recovery first, compare it with
the existing GCN under the same readers, then test VAE and ordered auxiliaries
as separate deltas. If deterministic recovery cannot improve on the failed
ordered-layout representation, do not use a VAE to conceal the objective
failure. If real endpoints do not beat role/lag-matched rewires, remove the
exact-topology mechanism claim.

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
