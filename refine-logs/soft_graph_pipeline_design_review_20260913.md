# soft graph pipeline engineering/design review — 2026-09-13

Scope: bounded static review of `route_graph/soft_graph_energy.py`, `route_graph/soft_graph_structure.py`, `route_graph/soft_graph_phases.py`, with `event_matcher.py`, `source_event_graph.py`, `target_dependence.py`, and existing soft graph/source-event tests read only as supporting context. I did not modify implementation or run GPU. I did not independently rerun the reported 31 CPU tests.

Status: Critical 0, Required 2, Optional 4. Energy is no longer the main blocker; the remaining blockers are in the control-pool contract and null/unknown source provenance coverage.

## Confirmed fixes / non-blocking checks

1. `soft_graph_energy.positive_dependence` now fails closed unless the native edge has exact sham/prefix checks, two fixed controls, distinct control IDs, selected ID outside controls, raw-origin mediated scope, donor and recipient origin sham checks, artifact verification, and repeat validity. Negative or non-specific effects do not lower risk or count as corrective evidence.

2. Unknown mass is neutral for risk. `infer_claim` uses `Lglobal = log((C+N+eps)/(S+eps))` and shrinks scores toward 0.5 through confidence; there is no `-lambda_unknown * pU` term. Local `I` is folded into local unknown, so a locally unrelated candidate cannot manufacture absence evidence.

3. History propagation is now directionally constrained. Only an argmax relation label with probability at least 0.8 can become known, only `reuse` propagates, and correction/quote/new-topic/unknown do not flip a previous risk. The extra multiplication by the previous claim confidence prevents an almost-all-unknown prior with a large raw logit from driving downstream risk.

4. The phase interfaces preserve the intended prediction/certificate separation. `GLOBAL_PROMPT` is finite SCNU over full source; `LOCAL_PROMPT` is finite SCIU over a highlighted occurrence; `HISTORY_PROMPT` is finite R/C/Q/T/U. None of these prompts asks for a free-form answer. Native phase measures all claims under the input limit and uses at most two source edges plus one history edge per claim.

5. I did not find future-response leakage in the explicit payloads. Structure/semantic phases pass only `row["response"][:claim["span"][0]]` as earlier response, and history candidates are drawn from prepared claims with `span[1] <= current span[0]`. Response feature capture is over the full causal input, but under a causal decoder hidden states at an earlier token should not attend to later response tokens; this should still be stated as an assumption of the frozen model replay.

## Required

### R1. Native control pools are still structurally underqualified, so a positive native edge can be over-specific by construction.

Location: `soft_graph_structure.matched_controls` selects controls by disjoint keys, token-count ratio, and state-norm distance only. `soft_graph_phases.semantic_phase` then filters source controls only by local `I >= 0.8` and history controls only by `new_topic >= 0.8`.

Why it matters: this is not enough for the `positive_dependence` gate to mean “specific to the candidate source/history occurrence.” For source edges, a selected literal field or parsed role can be compared against raw-token controls or a different source kind with similar norm/length. For history edges, a selected factual event can be compared against a structurally different earlier fragment. The semantic unrelated/new-topic filter removes related controls, but it does not enforce matched nuisance structure. That can make `delta - max(control_deltas)` large because the controls are easy negatives, not because the selected occurrence is specifically causal.

Minimum fix before using `full_graph` native lift as evidence: freeze structurally matched controls before C/D with same domain and disjoint keys, plus compatible source kind and role class. Concrete sufficient rules:

- selected parsed source role: controls must also be parsed source roles with the same role name or a predeclared compatible role class, comparable key count, comparable state norm, and different event membership;
- selected literal field: controls must also be literal scalar fields with same value type and preferably same terminal field name or same record schema position; raw tokens should not be controls for literal fields unless the selected candidate itself is raw-token fallback;
- selected raw-token fallback: controls may be raw-token fallback with comparable key count/norm, but the resulting edge should be marked raw-token scope, not event/field specificity;
- history controls: controls must be previous claim/event spans of the same extraction class and comparable token count/norm, then C may filter them as new-topic;
- if two such controls are unavailable after finite semantic filtering, the native term must remain `controls_valid=False` / positive weight 0, not fall back to weaker controls.

The current code already has the right ordering principle: B freezes the pool before C, and D does not refill controls after effects. The missing part is the structural eligibility predicate used to form that frozen pool.

### R2. `null_literal` source candidates are produced by the matcher but dropped before finite verification and output marginals.

Location: `event_matcher.propose_event` marks source literal `None` as `status="null_literal"`; `soft_graph_structure.assignment_marginals` skips every candidate whose status is not exactly `matched`.

Why it matters: Data2txt `None` / unknown fields are explicitly part of the source graph, and earlier design constraints said they should remain unknown rather than become absence or disappear. At the moment they can appear in `role_candidate_pools`, but they never become `source_terms`, never receive local SCIU verification, and never show up as candidate provenance in the soft graph score. Full-source SCNU may still handle the claim globally, but the graph component loses the applicable-unknown source occurrence and may overemphasize a nearby observed literal instead.

Minimum fix: keep `null_literal` in the marginalized candidate output as an applicable source-provenance candidate whose local relation is forced or verified to U and whose native risk weight remains 0 unless a later explicit rule justifies otherwise. It should contribute to coverage/confidence reporting as “source field found but epistemically unknown,” not to S/C risk. If this is intentionally excluded from the first full36, the output must count `null_literal_dropped` as an uncovered denominator, otherwise Data2txt coverage will be overstated.

## Optional / should-not-block engineering notes

1. `assignment_marginals` hardcodes temperature `.2` instead of using `PROTOCOL["assignment_temperature"]`. This is not a current correctness bug because the constants match, but changing the protocol later will silently not change the marginalizer.

2. `positive_dependence` checks distinct `control_ids`, but it only checks the length of `origin_measurement_ids`. If malformed data could enter merge, duplicate origin measurement IDs would still pass. This is lower risk in the current execution path because `_measure_edge` constructs them from oracle records, but a distinctness check would make the energy gate more self-contained.

3. `response_claims` avoids splitting parsed event anchors at punctuation, which addresses the main “event not split” concern. If pointer extraction misses an event, fallback leaves still come from text-unit/punctuation boundaries and are encoded as `condition`-only partial events. That is acceptable as a coverage fallback, but downstream output should keep `extraction_kind="unparsed_leaf"` visible and should not compare fallback coverage to parsed event coverage as if they had equal event identity.

4. The top-k verifier budget is four total source terms per claim, not four candidates per response role. If the intended contract is truly per-role four, the current implementation under-covers multi-role events. If the intended contract is four role edges total per claim, the code matches it; make the wording in the run manifest unambiguous.

## Freeze recommendation

Energy can be frozen for the next batch, but the full soft graph native variant should not be interpreted as a valid control-adjusted native improvement until R1 is fixed or explicitly reported as `control_structural_unresolved`. R2 affects Data2txt unknown/null provenance coverage; it is less likely to create false high native risk, but it can make the graph appear to have searched/represented applicable source evidence when the null-literal candidate was actually dropped.

## Regression review after R1/R2 fixes — 2026-09-13

Scope: bounded follow-up review of the two prior Required items plus adjacent energy/phase wiring. I inspected `soft_graph_structure.py`, `soft_graph_phases.py`, `soft_graph_energy.py`, `event_matcher.py`, and the related tests. I also ran a short inline regression check for the exact failure modes and the relevant CPU tests.

Verification run:

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=graph /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest graph/tests/test_soft_graph_energy.py graph/tests/test_event_matcher.py graph/tests/test_soft_graph_pipeline.py -q
31 passed in 13.76s
```

Additional inline checks passed for: parsed-role controls rejecting wrong role/raw-token controls; literal controls requiring same value type and normalized schema with different record and non-None control; `null_literal` surviving `assignment_marginals`; duplicate `origin_measurement_ids` failing closed in `positive_dependence`.

### R1 status: closed

`matched_controls` now freezes structural eligibility before C/D. The current predicate enforces same `kind`; parsed roles additionally require same role and disjoint event membership; literal fields require same `value_type`, normalized field-path schema, different record membership, and non-None controls; raw fallback only matches raw; history matches the same `extraction_kind`; length and state-norm ratios are both bounded in `[0.5, 2]`. C only filters this frozen pool with finite semantic probabilities, and D gets no replacement controls after native effects. If two eligible controls are unavailable, `_measure_edge` remains position-only and `positive_dependence` returns zero. This addresses the previous false-specificity blocker.

### R2 status: closed

`assignment_marginals` now retains both `matched` and `null_literal` candidates, so literal `None` provenance is no longer dropped before local verification/output. `semantic_phase` still calls the finite local verifier and records its raw distribution, then forces `q_local={S:0,C:0,U:1}` with `epistemic_override="literal_None_is_unknown_not_false_or_not_stated"`. That preserves source provenance while preventing null literals from becoming support, contradiction, or absence evidence. This addresses the previous Data2txt unknown-field coverage blocker.

### Energy/phase follow-up status

The energy gate now also rejects duplicate `origin_measurement_ids`, and `assignment_marginals` takes the protocol temperature instead of a hard-coded constant. `extraction_kind` is carried into the inference output, so downstream merge/eval can separate parsed pointer events from unparsed fallback leaves.

No remaining Critical or Required blockers found in this bounded review.

Optional note: `native_phase` still chooses source edges by the pre-semantic matcher order, so a high-marginal `null_literal` or local-U term can consume one of the two source native slots even though it cannot contribute risk after the U override. This is coverage/performance risk rather than a validity blocker because the energy contribution remains zero. If full36 shows many skipped relation-known source edges, select native source jobs after C by `pi * pool_quality * (q_local[S] + q_local[C])`, while still using only the fixed B pool and without backfilling controls.
