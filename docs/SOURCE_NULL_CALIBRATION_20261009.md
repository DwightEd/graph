# Zero-effect-anchored source readout, controlled experiment

Status: all five literal CPU score/evaluation commands completed exit0. Predeclared primary on complete exposed QA DEV reaches AUROC .803398309, compared with old .772523119 and prior likelihood .773897757. Source-cluster AUROC increment95CI versus old is [.027357717,.034597403]. This achieves the numerical DEV ranking target, but the complete expansion gate fails: same5414FP TP2136->1671 and normal-answer alarms438->511. Default unchanged, no new test/other-task expansion. This is a limited coordinate correction, not a factual binding detector; independent numerical audit and complete token diagnosis are finished. Audit verdict WARN retains historical exposure, observer and runtime/snapshot limitations; the numerical reconstruction passes.

The observed defect is that the same source-deletion effect of zero receives different conditional ranks. Let Delta be absent-minus-present actual-token log probability, with the same native local/full view. The previous source-conditioned empirical CDF has q = F_b(Delta), where b is the frozen native-logp bucket. On the complete source-equal FIT reference, q0 = F_b(0) ranges from .335129 to .961166 locally and from .307905 to .971673 in the full-prefix view. Thus a high rank can describe an ordinary near-zero effect on a confident token. It does not necessarily describe factual inconsistency.

The fixed primary replaces each source coordinate by

\[
Z = \tfrac12 + \frac{q-q_0}{2\max(q_0,1-q_0)}.
\]

Zero maps exactly to .5, values stay in [0,1], and the slope with respect to q is in [.5,1]. Within a bin the order is retained; different bins can reorder. CDF atoms and gaps remain. The formula introduces no tolerance, entropy error unary, annotated prompt entities, max-attention source selector, or span average. The midpoint is a statistical coordinate, not a probability of hallucination. Zero actual-token effect can arise from saturation or cancellation; it does not establish absent reading of evidence.

The unchanged unary is .375(Z_local + Z_full) + .25 rank(raw_route). The unchanged graph uses the original layer15 local-attention channel, mean across32 heads, lag8, incident-degree normalization, lambda .5 and delta1 Huber solve. This round does not claim to retain fullhead information or measure a new causal path. It tests whether a better coordinate makes the existing graph operate on a less biased unary.

These bounded fields stay in the quadratic interval of the delta1 penalty. The graph therefore solves the original Laplacian smoothing system, rather than discovering new boundaries. Each resulting coordinate combines nearby unaries; an already low error signal can remain low or lose a peak. If both source effects are nonpositive, both anchored coordinates are at most .5 and the unary is at most .625, irrespective of the route rank. This is a concrete limitation of the fixed signed-source blend, not evidence that such tokens are factually correct. Any threshold above that bound cannot recover them from their unary alone.

The eight predeclared fields are old_unary, old_native, likelihood_native, null_unary, primary null_native, balanced_native, global_null_native and shuffled_null_native. The balanced control rescales each side of q0 separately; its rare positive tail can amplify disturbances by17.6511 on the real FIT bins, so it is not the primary. Global and within-source shuffled references each have their own consistent zero anchor. Old three fields must reproduce their frozen complete-population scores within1e-8. No control can become primary after evaluation.

Implementation:

- `experiments/flow_latent/local_graph_transport/source_null_calibration.py`: bounded primary, balanced control, paired conditional coordinates.
- `run_source_null_calibration.py`: exact source/route/graph reuse; copies references byte-identically, scores complete answers, binds consumed scalar bytes and exact capture fields, freezes FIT/pilot/DEV before official-label evaluation.
- `test_source_null_calibration.py`:12 actual checks covering skewed zero ranks, endpoints, source weights, CDF atoms/gaps, bounded slope/order and original graph identities.

Frozen populations: FIT4026 answers/671 sources/612083 tokens; historical pilot8 answers/1139 tokens; QA DEV1008 answers/147981 tokens. All eight fields on all three populations freeze before labels. FIT95 is an unknown-label source-equal mixture quantile, not a normal5%FPR operating point. Equal-FP/normal-answer budgets at evaluation are explicitly label-assisted oracle diagnostics. The experiment has0 new model/GPU forwards and0 natural-label fits; historical observer and development exposure limits remain.

Run from graph with the existing research Python:

```bash
env PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_source_null_calibration --stage score --phase fit
```

Repeat the same command with phase pilot, then dev. Only after all three freeze files exist and verify, use stage evaluate with phase pilot, then dev. The fresh output directory is `outputs/source_null_calibration_20261009`; completed phases refuse overwrite. Canonical full commands, actual execution witness, fixed plan, reference diagnostics, results, every FP/FN and independent audit are under `/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_null_calibration_20261009/`.

The probability-scale residual literature supports CDF-based residual coordinates, not this physical-null anchor or factual detection: Shepherd, Li and Liu2016, [Probability-scale residuals for continuous, discrete, and censored data](https://pmc.ncbi.nlm.nih.gov/articles/PMC5364820/). This source-null transformation is our limited hypothesis. Even if it improves ranking, it adds no subject/relationship/range information. Errors supported by a source word but bound to the wrong fact can remain undetectable. A failed fixed primary will not trigger test/other-task expansion or replace the default.
