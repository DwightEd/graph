# Architecture Round 1 Method Review

Date: 2026-09-13

## Review Mode

Codex MCP is unavailable. This file is a real GPT-5.5 xhigh collaboration-agent fallback review using the `research-refine` method-review rubric, not an external MCP verdict and not an experiment-integrity audit. No GPU was used. No code, frozen RAGTruth run, experiment output, state file, or manifest was changed. The review input was:

- `refine-logs/architecture_round0_20260913.md`
- `refine-logs/grounding_architecture_lit_20260913.md`
- `refine-logs/attribution_architecture_lit_20260913.md`
- `refine-logs/unsupervised_detector_lit_20260913.md`

## Bottom Line

The proposal has the right top-level split: normative evidence applicability must be separated from actual model adoption. That split preserves the problem anchor better than the older route-only or entropy-only ideas. But the current architecture is still not implementation-ready as a focused paper method. It defines many plausible pieces, yet the exact target being optimized, the evidence-set identifiability assumptions, the span transition model, and the causal group acceptance criteria remain under-specified. The plan also quietly makes the semantic weak-supervision branch the real hallucination detector, while the intervention branch mostly provides explanation and route-derived taxonomy. That is acceptable only if the paper says so explicitly.

The most important fix is to simplify. Do not add more experiments or more modules. Make the first complete method a one-trainable-branch evidence assignment and segmentation model, plus an algorithmic finite-intervention causal verifier for route and lookback. Train an intervention student only after the verifier has produced enough measured group effects to justify distillation. Full QKV/MLP graph learning over all tokens/layers is not a credible first version on one 4090.

Verdict: **REVISE**

Overall score: **6.2 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8.0 | The proposal still attacks the actual anchor: source applicability, automatic lookback, error continuation/correction, and continuous spans without training on natural RAGTruth hallucination labels. The main drift risk is that semantic weak supervision becomes teacher-verifier distillation rather than measurement of source-grounded applicability. |
| Method Specificity | 5.9 | Many components are named, but key interfaces remain symbolic: what exactly is an evidence unit, what labels `membership(c,e)` when multiple evidence sets exist, what state transition defines a continuous span, and what finite-intervention criteria make a lookback group accepted. |
| Contribution Quality | 5.6 | The proposal currently contains at least four contributions: semantic evidence assignment, semi-Markov span decoding, intervention-effect graph distillation, and causal group search. The dominant contribution should be the evidence-vs-adoption comparator; the trainable route graph and full QKV/MLP model create contribution sprawl. |
| Frontier Leverage | 7.1 | Frozen LLM replay, finite interventions, source-derived weak supervision, and signed output-direction attribution are appropriate foundation-model-era primitives. The weak spot is not old-fashioned modeling, but using extra trainable graph machinery before the causal measurement object is pinned down. |
| Feasibility | 4.7 | One-4090 feasibility is not convincing. Saving residual, post-attention, MLP, Q/K/V, source/history factors, and doing many real group interventions across full responses will exceed the stated 6-16 GPU-hour first-pass budget unless the method is restricted to on-demand target-span VJPs and measured group replay. |
| Validation Focus | 7.0 | The three validation blocks are focused and avoid trivial "relations affect output" tests. However, the method itself is not yet frozen enough for those validations to be decisive. |
| Venue Readiness | 5.2 | A top-venue paper shape exists, but not with the current module count and ambiguity. Reviewers will ask whether the detector is just a weak semantic verifier with a post-hoc attribution story. |

Weighted calculation: `8.0*.15 + 5.9*.25 + 5.6*.25 + 7.1*.15 + 4.7*.10 + 7.0*.05 + 5.2*.05 = 6.22`.

## Grounding Facts Used

These are literature/code facts from the local notes, not new claims from this review:

- MiniCheck provides useful synthetic support/necessary-evidence construction, but its labels are model-generated and only partially reliable under the paper's small manual audit. Its inference path chunks documents and takes maximum support probability, so it does not solve multi-evidence provenance or generator-internal adoption.
- GraphCheck encodes document and claim graphs into a small number of soft tokens for a frozen LLM. It does not output local claim-source alignment, null evidence, or generator route adoption. Its reported setup uses large models and multi-A100 training, so it is not evidence that the current full graph design is single-4090 feasible.
- ALTI-style rollout is nonnegative and normalized after local transformations; it cannot represent contradiction or signed adoption. DecompX/autoregressive token decomposition preserve output-direction components better, but reconstruction under fixed routing is not the same as finite causal intervention.
- EAP-IG, Circuit Tracing, and QK attribution support the need for signed output-direction effects and group verification, but they also warn that independent edge scores and frozen-attention OV graphs can miss QK-mediated or nonlinear joint effects.
- IRIS and RAUQ are useful baselines or design hints for uncertainty/history propagation, but neither solves source applicability, evidence ownership, route-derived hallucination, correction, or token/span localization.

Everything below is this reviewer's inference about the current proposal.

## Critical Blockers

### 1. The actual detection branch is the semantic branch, not the route branch

The proposed route-derived hallucination rule says: unsupported span plus positive adoption from wrong evidence. In that rule, "unsupported" and "wrong evidence" are both supplied by the semantic evidence assignment branch. The intervention branch measures what pushed the model to emit the span, but it does not know whether the span is false or unsupported. Its target `F_c = mean log p(y_t | prompt, y_<t)` is a generation-support target, not a factuality target.

This is not fatal, but it must be stated as the method's core structure:

- The semantic branch detects unsupported/contradicted/NULL spans.
- The causal branch verifies adoption, lookback, continuation, and correction for those spans.
- The final hallucination label is evidence-first; route evidence refines the subtype and influence interval.

Concrete fix: rename the dominant method around the comparator, for example "route-verified evidence assignment", and delete any claim that the native route graph by itself detects hallucination. The method can still test whether route signals improve boundaries or continuation labels, but the paper should not pretend that `Delta F_c` is a truth signal.

Priority: **CRITICAL**

### 2. Weak supervision risks drifting into teacher-verifier distillation

The training protocol allows Qwen-generated provenance and Llama verification. That is reasonable weak supervision, but it is no longer "no semantic supervision." The proposal mostly acknowledges this, yet the architecture still depends on the weak semantic head for the primary unsupported posterior. If the teacher proposes the wrong evidence or learns artifacts of generated correct/incorrect texts, the detector can become a verifier-distillation system rather than a source-grounded measurement model.

Concrete fix: split the semantic labels into three stored strata and never mix their claims:

1. `constructive_provenance`: source-derived statements with known source spans from the data-construction process.
2. `verified_semantic_weak`: teacher-proposed evidence/labels accepted only after an independent verifier check, with confidence and abstention retained.
3. `perturbation_boundary`: response spans whose edited character ranges are known, without automatically treating every edited continuation as hallucinated.

Train with confidence-weighted losses, but report the detector as trained with source-derived and model-verified weak semantic supervision. The strict no-semantic-weak-supervision version should be a degraded control, not the main promised capability.

Priority: **CRITICAL**

### 3. Multi-evidence membership is not identifiable as written

`membership(c,e)` is non-exclusive, NULL is allowed, and support can require multiple facts. But the proposal does not define the label when several alternative evidence sets can support the same claim, when deleting one evidence unit leaves another valid support path, or when two source units jointly imply a claim but neither is individually necessary. A binary membership loss will either over-label all plausible evidence or under-label alternatives.

Concrete fix: make the evidence set latent and evaluate minimal support sets, not independent evidence membership. For each claim span `c`, define a candidate family `E_c = {G_1...G_m, NULL, ABSTAIN}` where each `G_k` is a set of source units. The semantic model outputs:

```text
p_app(G_k | c, D)
p_state(c) in {supported, contradicted, insufficient, abstain}
p_member(e | c) = sum_{G_k contains e} p_app(G_k | c, D)
```

Training should use marginal likelihood over all verified support sets:

```text
L_set(c) = -log sum_{G in VerifiedSupport(c)} p_app(G | c, D)
```

For NULL, require a verified absence condition over the candidate family, not just deletion artifacts. If verified alternatives are incomplete, train partial labels as constraints: positive evidence units must receive enough marginal mass, known invalid units must receive low mass, and the unobserved rest is masked rather than labeled negative.

Priority: **CRITICAL**

### 4. Automatic span decoding can collapse distinct facts into one span

The semi-Markov layer prefers continuation when adjacent spans share evidence. That will merge unrelated clauses supported by the same source passage, and it will also merge an unsupported error followed by a supported correction if the evidence distribution remains similar. The current "longer than 64 words can merge adjacent blocks" rule protects compute, but it does not define a reliable factual span boundary.

Concrete fix: define span state transitions explicitly at token or word-boundary positions. A minimal transition model is:

```text
z_i = (support_state_i, evidence_set_id_i, route_state_i, relation_to_prev_i)
support_state in {supported, contradicted, insufficient, abstain}
route_state in {no_verified_route, nonapplicable_positive, applicable_positive, history_positive, history_negative}
relation_to_prev in {same_claim, new_claim, elaboration, correction, topic_shift}
```

The boundary score between positions `i` and `i+1` should add a boundary when any of these changes with high posterior:

```text
B_i = w_s * 1[support_state changes]
    + w_e * evidence_set_distance(E_i, E_{i+1})
    + w_r * 1[route_state changes]
    + w_rel * p(new_claim/correction/topic_shift)
    - w_cont * p(same_claim/elaboration)
```

Use Viterbi or forward-backward over this state space. Do not let "same evidence set" alone suppress a boundary. For corrections, the support state and text relation must override evidence-set similarity.

Priority: **CRITICAL**

### 5. Causal group selection lacks acceptance criteria

"Beam search finds a minimal group whose erroneous attribution path exceeds a pre-registered effect threshold" is still symbolic. It does not specify the gate, the endpoint, sufficiency, necessity, specificity, sign stability, or how to handle multiple equivalent groups. Without these, automatic lookback can select attention-heavy groups that change likelihood for stylistic or formatting tokens rather than the factual constraint.

Concrete fix: define each causal group result as a measured object with four required checks:

```text
Target:
  F_c = mean_{t in content_tokens(c)} log p(y_t | prompt, y_<t)

Gate:
  gate(G, lambda) attenuates named source/history/MLP messages in G,
  with lambda in {1.0, 0.5, 0.0}; all runs replay the same token history.

Deletion effect:
  Delta_del(G,c) = F_c(base) - F_c(gate(G,0.0))

Sufficiency effect:
  Delta_keep(G,c) = F_c(base) - F_c(keep_only(G, same_role_candidates))

Specificity:
  Delta_del(G,c) must exceed max effect on unrelated control spans by margin gamma.

Acceptance:
  verified_positive(G,c) iff Delta_del(G,c) >= tau_eff,
  Delta_keep(G,c) <= eps_keep,
  sign(Delta at lambda=0.5) = sign(Delta at lambda=0.0),
  and specificity margin >= gamma.
```

Minimality means no proper subgroup in the searched candidate family satisfies the same checks. If two non-nested groups pass within tolerance, return both and mark non-unique. If deletion passes but sufficiency fails, call the group contributory, not sufficient. If no group passes under budget, return unresolved.

Priority: **CRITICAL**

### 6. The QKV/MLP full graph is not a credible first-pass trainable object on one 4090

The proposal asks to replay Llama3.1 and save residual, post-attention, MLP, Q/K/V, source/history connection factors, full token graph nodes, layer indices, and group interventions. At 32 layers, long RAG contexts, and full attention rows, dense attention alone can become tens of GB per example at 4k tokens if stored naively. Backpropagating full-chain VJPs for many spans and masks multiplies this. The stated 6-16 GPU-hour budget is not believable unless most of this is on-demand and target-restricted.

Concrete fix: the first implementable method should not train a full native-graph head. Use:

- one trainable semantic evidence/segmentation model;
- on-demand observer replay for selected risk spans;
- VJP or finite-gate measurements only for content tokens in those spans;
- chunked attention-row extraction for source/history candidates, not stored full `T x T x L x H`;
- V-message and pre-softmax route gates first;
- MLP/residual gates only as named aggregate groups;
- QK decomposition as diagnostic after a verified group is found, not as part of the primary detector.

Only after this measured verifier works should a student head be trained with:

```text
L_route = Huber(predicted_Delta(G,c,mask), measured_Delta(G,c,mask))
```

The route student must be explicitly optional acceleration. It should not be required for the first scientific claim.

Priority: **CRITICAL**

## Other Important Issues

### A. `support/contradict/insufficient(c,Source)` is ambiguous

The proposal alternates between evidence unit, source document, source set, and full-source classification. Define a single hierarchy:

```text
EvidenceUnit e: sentence/clause/span with source_id, char range, token range, role/context.
EvidenceSet G: finite set of EvidenceUnits.
ClaimSpan c: response char/token range.
SupportState(c): state under the best/marginal evidence sets across all sources.
```

Contradiction should be tied to a relevant evidence set that asserts an incompatible value/relation, not to generic failure of support. Insufficient should include NULL/absence when no verified support set exists.

Priority: **IMPORTANT**

### B. "Route-derived hallucination" needs an exact rule

Use a deterministic rule after posteriors and measured effects are available:

```text
unsupported(c) iff p_state(insufficient or contradicted | c) >= tau_u
route_adopted_wrong(c) iff exists group G:
  verified_positive(G,c)
  and semantic_relation(G,c) in {nonapplicable, contradictory, unsupported_history}

route_derived_hallucination(c) iff unsupported(c) and route_adopted_wrong(c)
```

For history:

```text
error_continuation(h,c) iff unsupported(h) and unsupported(c)
  and relation(h,c) in {same_claim, elaboration}
  and verified_positive(history_group(h), c)

correction(h,c) iff unsupported(h)
  and relation(h,c) in {negation, correction, replacement}
  and measured history effect on the old wrong value is negative
```

If current `c` is supported by source evidence, do not inherit the hallucination label from `h`. The proposal says this in prose, but it needs to be a machine-level transition rule.

Priority: **IMPORTANT**

### C. The model should distinguish fact interval from influence interval by construction

Fact interval:

```text
contiguous spans where p_state(insufficient/contradicted) >= tau_u
and the decoded claim identity/relation remains the same
```

Influence interval:

```text
positions/spans for which a verified group tied to the earlier unsupported claim
has |Delta_del(G,c)| >= tau_eff under fixed-history replay
```

These intervals can overlap, but neither contains the other by definition. The current proposal states the distinction, but the decoding contract should return two separately computed intervals with separate uncertainty.

Priority: **IMPORTANT**

### D. The semantic model input may leak response artifacts into source applicability

The semantic view says source token representations read only source/task, while response text gets a frozen "reading" forward pass. That is good, but the pair model can still learn artifacts from synthetic erroneous responses. Add two controls at the method level:

- source-only perturbation labels cannot include response style/template features;
- response encoder dropout/paraphrase consistency should force the same evidence assignment under surface rewrites of the same claim.

This is not an extra experiment block; it is part of the training objective:

```text
L_consistency = KL(p_app(. | c, D), p_app(. | paraphrase(c), D))
              + KL(p_state(. | c, D), p_state(. | paraphrase(c), D))
```

Only apply it to verified paraphrases, and keep abstain when equivalence is uncertain.

Priority: **IMPORTANT**

### E. The "learned graph" claim is not yet justified

If the semantic branch is a cross-encoder over claims and source units, it is already a strong baseline. A graph layer is justified only if typed edges solve a specific ambiguity: ownership, stage/time, condition scope, or multi-evidence conjunction. Do not make "graph" the contribution unless the graph is the smallest mechanism needed for those errors.

Concrete fix: make the primary architecture "typed evidence-set assignment" and allow the encoder to be implemented as either cross-attention or a typed graph transformer. The claim should be about preserving claim-to-evidence set structure and comparing it to measured adoption, not about inventing a GNN.

Priority: **IMPORTANT**

## Recommended Simplified Method

This is the smallest complete method I would revise toward.

### Interfaces

```text
ClaimSpan
  id
  response_char_start, response_char_end
  response_token_ids
  content_token_ids
  online_or_offline

EvidenceUnit
  id
  source_id
  char_start, char_end
  token_ids
  parent_sentence_id
  local_context_ids
  extraction_confidence

EvidenceAssignment
  p_state(c) over {supported, contradicted, insufficient, abstain}
  p_set(G | c, D) over candidate evidence sets plus NULL
  p_member(e | c, D)
  abstain_reason

SignedRouteMeasurement
  target F_c
  group G
  gate_type in {source_value, source_route, history_value, history_route, mlp_aggregate, residual_aggregate}
  Delta_del, Delta_keep, control_effect, sign_stability
  verified_status in {sufficient, contributory, negative, unresolved}

SpanState
  support_state
  evidence_set_id
  route_state
  relation_to_previous
```

### Trainable Components

Use one trainable component in the first paper version:

```text
EvidenceSegmenter(candidates, source_units) -> EvidenceAssignment + SpanState emissions
```

It can be a small cross-graph transformer if needed, but it should be described by its input/output contract, not by generic graph novelty. The intervention verifier is algorithmic measurement, not a second trainable model. A route student is optional acceleration after measured traces exist.

### Losses

```text
L_state = CE(p_state(c), weak_state_label, weight=label_confidence)

L_set = -log sum_{G in verified_support_sets(c)} p_set(G | c, D)
        for supported claims with one or more verified support sets

L_null = -log p_set(NULL | c, D)
         only when absence has been verified over the candidate evidence family

L_member = partial BCE on known-positive and known-invalid evidence units;
           unknown alternatives are masked

L_segment = negative log likelihood of weak/known character boundaries;
            marginalize unknown boundaries with forward-backward

L_consistency = source-order equivariance + verified paraphrase consistency

L_route_student = optional Huber(predicted_Delta, measured_Delta)
                  trained only on measured intervention records,
                  with no gradient into EvidenceSegmenter
```

No loss should use natural RAGTruth hallucination labels for training, checkpoint selection, thresholding, or early stopping.

### Decoding And Transition

Decode spans with emissions from `p_state`, `p_set`, and relation features. Put boundaries where support state, evidence set, relation, or verified route state changes. Allow long spans only when both claim identity and evidence relation remain stable. A supported correction following an unsupported span starts a new fact interval even if it references the same evidence.

### Causal Group Criteria

A lookback group should be returned only as a measured finite-intervention result:

```text
accepted_sufficient(G,c) iff
  Delta_del(G,c) >= tau_eff
  and Delta_keep(G,c) <= eps_keep
  and target_specificity(G,c) >= gamma
  and sign_stable(G,c)
  and no searched proper subgroup satisfies the same criteria
```

Return multiple groups if non-unique. Return unresolved if the budget ends before these checks pass. This directly satisfies the user's request for automatic lookback without pretending that a single attention point is necessary or unique.

### Final Decision Rule

```text
unsupported_span(c) =
  p_state(insufficient or contradicted | c) >= tau_u

route_derived(c) =
  unsupported_span(c)
  and exists accepted_sufficient_or_contributory(G,c)
  and semantic_relation(G,c) in {nonapplicable, contradictory, unsupported_history}

error_continuation(h,c) =
  unsupported_span(h)
  and unsupported_span(c)
  and relation(h,c) in {same_claim, elaboration}
  and accepted_positive_history_group(h,c)

correction(h,c) =
  unsupported_span(h)
  and relation(h,c) in {negation, correction, replacement}
  and source_supported(c)
```

Thresholds `tau_u`, `tau_eff`, `eps_keep`, and `gamma` must be selected only on source-derived dev and measured route-dev records, not on natural hallucination labels.

## Simplification Opportunities

1. Delete the trainable intervention graph head from the first complete method. Keep measured finite interventions and optional local gradients. Add the student only as a later acceleration module.
2. Do not put QK, V, residual, and MLP into one all-purpose graph learner. First version should use V/route gates plus aggregate MLP/residual controls; QK decomposition is diagnostic for verified groups.
3. Make semi-Markov span decoding a thin structured layer over evidence and route-state emissions, not another major contribution.

## Modernization Opportunities

1. Use finite group interventions and output-direction VJPs as the modern primitive; that is more natural here than another GNN classifier.
2. Treat LLM-generated provenance as weak data construction with confidence and abstention, not as an oracle. This keeps the foundation-model leverage honest.
3. If distillation is needed, distill measured causal effects into a student after measurement succeeds. Do not train a route predictor before the measured target distribution exists.

## Drift Warning

No full drift yet. The proposal still aims at the anchored problem. The dangerous future drift is semantic-verifier distillation plus attribution decoration: if the unsupported-span scores come from a weak semantic teacher and the route branch only explains already-flagged spans, the paper must claim route-verified evidence assignment, not unsupervised native-graph hallucination detection.

## Required Action Items Before The Next Architecture Round

1. State that semantic evidence assignment is the detector and route measurement is the adoption/lookback/correction verifier, unless the route branch truly changes the detection decision.
2. Replace independent `membership(c,e)` supervision with latent evidence-set supervision and partial-label constraints.
3. Write exact span transition states and boundary scoring, including correction and topic-shift resets.
4. Define finite-intervention group acceptance: deletion, sufficiency, specificity, sign stability, minimality, non-uniqueness, and unresolved status.
5. Remove the trainable intervention graph head from the first method or demote it to optional acceleration after measured traces exist.
6. Rewrite the single contribution as: evidence applicability versus measured adoption for route-derived unsupported spans and influence intervals.
7. Re-estimate compute under an on-demand replay contract. The current full QKV/MLP graph training plan should not be presented as one-4090 feasible.

Final verdict: **REVISE**. The architecture has a viable core, but READY requires a narrower method with machine-level targets and measured causal criteria. The next revision should be a complete interface-and-loss spec for the simplified method, not a larger experiment menu.
