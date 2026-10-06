# Graph: mechanism-grounded token detection

Status: tested measurement/optimization primitives plus an experimental independent-text relation parser and sparse key-address pilot. Automatic semantic proposals, independent reference pools, stratified 8B fidelity and new natural-data detection results are **not yet implemented/validated as a complete pipeline**. The measured-event CLI must not be presented as an end-to-end text detector.

## Module responsibilities

| Layer | Existing location | Responsibility |
|---|---|---|
| Native model and dataset adapters | `teaching/state_audit/src/state_audit` | Native sites, tokenization/offsets, replay, interventions and storage |
| Full-answer root attribution | `experiments/token_backtrace/trace.py`, `readout.py` | Each original target independently, complete prompt/history; explanatory triangular lineage |
| Internal message effects | `experiments/token_backtrace/messages.py` | Native pre-WO head tensors, residual-normalized O-only selection, aligned donor differences, directional VJPs and finite patches |
| Token graph readout | `experiments/token_backtrace/global_graph.py` | Source/scope eligibility, bounded capacities, exact min-cut, token min-marginals, separate direct alarm and coverage |
| Explicit disk stages | `experiments/token_backtrace/pipeline.py` | Read measured events, write every original token; no labels, observer generation or hidden fitting |
| Existing automatic source experiments | `experiments/automatic_evidence` | Source addresses, position-preserving key masks, source/donor captures; lexical relation proposals remain historical diagnostics |
| Retained comparisons | `experiments/unsupervised_graph/scalar.py`, `fixed.py`; `experiments/source_relation` | Exact historical fixed baseline; source-relation JS and message MMD |
| Native history checks | `experiments/choice_feedback`; `experiments/gsm8k_states` | Sequential KV and first-error/state diagnostics, separate from detection claims |
| Evaluation | `token_backtrace/span_metrics.py`, retained evaluator utilities | Read annotations only after scoring; character union and span coverage |

Small files in older experiment packages remain only when the dependency inventory shows actual consumers. In particular, `decision_risk_flow/run.py` now contains only model loading; its supervised CLI is removed. Existing untracked `routing_likelihood` imports remain valid. No extra algorithm package or duplicate model adapter is introduced.

## Generation mechanism and the quantities measured

Use zero-based answer indices. A prompt of length P predicts y_t at native position **P+t−1**. The post-token carrier for y_q sits at **P+q** and may influence only targets t>q. All original prompt/history tokens remain in native replay; the final carrier has no later answer target. Storing the full answer does not permit future tokens to influence earlier predictions.

For physical head h in layer l at receiver q, capture

```
m[l,h,q] = sum_i A[l,h,q,i] V[l,h,i]      # before W_O
```

This is the entire head output, not a single attention edge. Native RMSNorm, residual connections, attention routing and MLPs remain in the autograd graph. O-only head selection uses `||W_O^h m|| / (||residual_before|| + 1e-8)`; ties are ordered by physical layer/head. Attention concentration, hidden rank and repeated heads do not define correctness.

For each aligned world P∈{repair R, equivalent E, unrelated U}, define the frozen full-vocabulary direction

```
v = log p_P − log p_O
a = p_O * (v − sum(p_O * v))
F = a^T logits_O
eta[P,donor,q,h,t] = (d F_t / d m[q,h])^T (m_donor[q,h] − m_O[q,h])
```

`a = J_softmax(logits_O) v`; its components sum to zero. The dot product is the first-order Taylor term of F under a whole-head patch. It is not the finite intervention effect. `finite_effect` measures the actual changed native output under the same frozen direction. It currently replays the complete input, so its cost is a full forward, not an accelerated suffix run. Both eager and SDPA small-model tests check derivatives and causal positions. Real 8B error bounds remain unknown.

All three worlds define their **own** direction. For each P, the positive excess subtracts the strongest nonnegative effect of the other donors under that direction. The selected statistics preserve physical head identity:

- node: maximum excess over that node's selected heads and future targets;
- directed edge: maximum excess over heads for q→t;
- adjacent pair: maximum, over a common physical head and common future target, of the smaller endpoint excess.

The E/U self-aligned statistics are empirical geometry references; semantic roles are not exchangeable, so they do not provide p-values or a false-positive guarantee. Unaligned/missing observations remain explicit, distinct from a measured zero. Alignment requires identical original token identities in a contiguous suffix and equal position displacement across donor worlds.

`C = B + L C` is a triangular explanatory lineage from normalized total embedding-root responses. It is useful for source ancestry; its edges are not native local Jacobians and its row mass is not factual support. The original signed raw root responses stay in their trace cache.

## Sparse physical source addresses (2026-10-06)

`messages.native_edge_trace` additionally captures explicit `(layer, head, query,
key, source_id)` coordinates. For each edge, the observed patch direction is
`m_e = A[query,key] V[key]` before W_O. `sparse_edge_vjp` returns the signed
amplitude-deletion derivative `-grad(log p(y_t)) dot m_e` for every independent
target. Native head-output tensors retain autograd through downstream routing,
RMS, residuals and MLPs. Detached Q/K/V captures define the fixed deletion
direction; this does not differentiate an independent reconstructed attention
program. `finite_edge_effect` uses the existing native edge-gate implementation
and complete replay for the same amplitude deletion without renormalization.
Addresses, target indices, attention, valid flags and signed derivatives remain
separate. A sampled address table is not a complete all-edge graph.

`logic_benchmark --stage typed-capture/typed-evaluate` pilots independent source
and answer relation extraction. Literal quotes are mechanically grounded; every
alternative contributes to conservative consensus. Potentially related records
with mismatched parses stay unknown, and lexical exclusions are saved. Parser
failure, unsupported content, quotation and unresolved scope are explicit. All
natural original-token offsets stay in the denominator. This candidate parser
is not a completed graph detector: matched edits, typed uptake versus parser-only
and rewired controls, independent calibration and natural detection still need
validation. See the shared TG tracker before expanding runs.

## Why the graph solver is exact, and what it does not prove

For an independently source-qualified relation e, use

```
E_e(z) = sum_t (c + r_t − s_t − u_t) z_t
       + sum_(q<t) w_qt z_q (1−z_t)
       + sum_t b_t |z_t−z_(t−1)|,          c = 0.5.
```

The semantic source contrast and its qualification are upstream inputs. Prior generated steps stay in the model conditions, but do not automatically become external factual authority. This limits first-error coverage in GSM when the relevant constraint must itself be derived.

Each pair term is submodular: `w≥0` has energies (0,0,w,0) for assignments (00,01,10,11), and Potts has (0,b,b,0). Equivalently `E00+E11 ≤ E01+E10`. With z=1 on the source side, q→t capacity w charges precisely zq=1,zt=0. Two opposite arcs of capacity b implement Potts. For unary d z, use node→sink capacity max(d,0), source→node max(−d,0), and constant min(d,0). Thus a minimum s–t cut minimizes the original binary energy exactly. Source reachability in the final residual graph gives the inclusion-minimal source set among tied minima without perturbing the energy.

One unconstrained optimum plus one opposite-label constrained cut per token gives

```
rho_t = min_(z_t=0) E − min_(z_t=1) E.
```

The solver returns the original energies and constrained solutions, so min-marginals are auditable. An exact solver does not establish that the chosen energy distinguishes hallucinations.

Graph conflict is seeded once at the earliest edit. Each subsequent node needs its own measured carrier evidence u; a single seed cannot pay an arbitrary span's token cost. Current expression scope gates membership: a normal intervening sentence may carry historical information without belonging to the erroneous span. Outgoing capacity is capped at 2c and adjacent continuity at .75c; independently supported transitions stop continuity. Direct alarms remain separate, so graph regularization cannot erase an already positive direct score. Distinct events keep distinct runs.

## Scoring contract and missing work

`pipeline.score_document` accepts schema `measured_token_graph_v1` with original token IDs/character offsets, a frozen proposal-completion flag, source ID, independently referenced nonnegative direct/graph thresholds, and per-event fields documented by `global_graph.score_events`. These are **already measured/calibrated** fields, not hand-annotated gold spans or arbitrary hidden-state norms. The CLI does not generate or certify them.

`reference_scale` requires a matched pool with ≥32 independent sources at c=.5, deduplicates record identities, source-weights right tails with ties, and retains unresolved pools as NaN. This count only permits numerical resolution; it is not a statistical sample-size guarantee. Matching, source disjointness, selected-statistic repetition and sham-derived noise floors remain the calibration caller's responsibility. The default floor 1e-8 is only a lower bound.

Next integration work in the shared GG tracker: frozen semantic observer proposals and scopes; closed-branch external-source contrasts; O/R/E/U construction; matched independent A/B pools; raw-to-calibrated event assembly; 12-stratum finite fidelity; complete natural-data comparison against direct-only/u-only/scope-only and rewired graph controls. Until these pass, no new detection AUROC/F1 is claimed.

## Validation and development

Run `python -m pytest -q` from the repository root. The test configuration covers retained root tests, experiment-local tests, and native adapter tests. `requirements.txt` includes the runtime needed to collect these tests. As requested, the numerical core uses the documented native coordinates and arrays directly, without repeated type/shape/range guards. Callers supply integer head coordinates, aligned suffix maps and boolean measurement flags; scope separately uses -1 for unknown. Causal positions, head identity, finite derivatives and cut optimality are checked in numerical tests. Hook cleanup remains in `finally`. Raw datasets, checkpoints, caches, results and existing user archives are not cleanup targets.

The existing scalar comparisons can be run end to end with `python main.py token baseline --stage run-test --output OUTPUT`. This fits unlabeled training references, freezes all 2,700 official test answer scores, then reads official annotations for token and character-span evaluation. It reuses cached native scalar measurements and does not run the unfinished message-graph detector.

Theory/measurement sources previously reviewed in the shared handoff: [AtP*](https://arxiv.org/abs/2403.00745), [CAGE](https://arxiv.org/abs/2512.15663), [Information Flow](https://proceedings.mlr.press/v306/xu26c.html), [CausalGaze](https://aclanthology.org/2026.findings-acl.1943/), [FlowTracer](https://proceedings.mlr.press/v306/dong26f.html). Their attribution machinery is not claimed as an unsupervised token detector or as this project's new mathematics.
