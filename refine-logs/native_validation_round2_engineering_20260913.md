# Native validation engineering re-review, round 2

Date: 2026-09-13  
Scope: `route_graph/causal_groups.py`, `route_graph/audit_pools.py`, `route_graph/audit_certificates.py`, `route_graph/audit_interactions.py`, `route_graph/audit_phase_native.py`, `route_graph/audit_output.py`; `route_graph/audit_protocol.py` was checked because it is frozen into run settings and must match the implementation contract.  
Mode: independent engineering/code review only. No GPU was used and no implementation file was modified.

## Verdict

Critical: 0  
Required: 2  
Optional: 4

The code is now runnable on CPU tests and the earlier high-risk implementation gaps are mostly closed: common-prefix gates are enforced, B/D base replay is checked, B proposal remains native/candidate-blind, origin controls now use fixed raw input windows, N skips paired-role and MLP branches, MLP specificity uses joint sham and status downgrading, and the continuous-error path now requires mapped previous answer-slot origin.

I do not see a remaining false-positive blocker for the strengthened continuous-error criterion. I do see two remaining blockers before freezing a natural 36-response run if the output is intended to make auditable internal-certificate and protocol-trace claims.

## Required findings

### Required 1: N `internal_certificate` can still be true with only a position/template certificate and no origin certificate

Files:

- `route_graph/audit_output.py:22-27`
- `route_graph/audit_output.py:113-117`
- `route_graph/audit_output.py:219-220`
- `route_graph/audit_phase_native.py:277-291`

For N claims, merge filters native positions by `template.passed` and `primary_id`, but it does not require `template.origin_passed` before setting `internal_certificate = bool(positions)` or before counting `risk_claims_with_internal_certificate`.

This produces a remaining coverage-level false positive if the phrase "N primary position/origin" means the N certificate must include both the repeated withholding-template position effect and the input-origin mediation check. A synthetic merge check reproduced the behavior:

- `template.passed=True`
- `template.origin_passed=False`
- selected position passed
- selected origin failed

The merged claim still had `native_certificate_ids=['p1']` and `risk_claims_with_internal_certificate=1`, while `mechanisms=[]`.

The mechanism output is conservative, because N does not resolve a mechanism without mediated origin. The blocker is the coverage denominator and certificate label: a reviewer can read `risk_claims_with_internal_certificate` as a full N internal certificate even when only the position side survived.

Minimal fix: split the output fields into `position_certificate` and `origin_certificate`, or require `template.origin_passed` for N positions to count toward `internal_certificate` / `native_certificate_ids`. If the intended definition is position-only, rename the coverage field so it does not imply origin identity.

### Required 2: frozen `PROTOCOL["validation_order"]` still describes a single branch that is false for N claims

Files:

- `route_graph/audit_protocol.py:28-36`
- `route_graph/audit_phase_native.py:189-208`
- `route_graph/audit_phase_native.py:212-300`
- `route_graph/audit_phase_native.py:325-327`

The implementation now correctly branches:

- non-N claims can run paired role and MLP interaction;
- N claims set paired role and MLP to `N_scope_unresolved_not_measured`;
- N then performs the primary content template check and layer-band check.

The frozen protocol still advertises a single validation order:

`position -> origin -> paired_role -> MLP_interaction -> N_template -> layer_bands`

This is no longer the actual audit contract for N claims. Since the protocol is serialized with the run settings, a full-batch artifact would claim an execution order that the code does not follow. That is a traceability blocker even though the implementation behavior is mostly right.

Minimal fix: make the protocol branch-specific, for example:

- `non_N_validation_order = ["position", "origin", "paired_role", "MLP_interaction", "layer_bands"]`
- `N_validation_order = ["position", "origin", "N_template", "layer_bands"]`

The existing `branch_policy` field in validation output is useful but not sufficient by itself, because the top-level frozen protocol remains contradictory.

## Confirmed fixes / non-blockers

### Continuous-error false-positive guard is now materially stronger

Files:

- `route_graph/audit_semantics.py:84-135`
- `route_graph/audit_phase_native.py:133-153`
- `route_graph/audit_output.py:43-80`

The C-phase relation check now binds `previous_question_id` to one of the blinded prior slot questions and requires `current_slot_derives_from_previous_slot`. D then maps the history origin to `previous_answer_span` only when `slot_link_valid` is true; merge only emits `error_history_continuation` when the mediated origin selection is `mapped_previous_answer_slot`.

I checked the critical same-sentence case with a synthetic merge: the current claim pointed to a supported prior slot in the same previous sentence while another prior slot in that sentence was unsupported. The merge produced only `conditional_history_dependency`; it did not emit `error_history_continuation`. This addresses the previously identified false-positive path where a correct fact in the same sentence could be treated as propagation from a different wrong slot.

### Origin control windows are the right small-scope repair

Files:

- `route_graph/audit_pools.py:120-142`
- `route_graph/audit_pools.py:172-222`
- `route_graph/audit_phase_native.py:154-160`
- `route_graph/audit_semantics.py:13-76`

The latest snapshot adds fixed raw-origin control pools after native search. The design is scoped correctly:

- B search still proposes candidates without semantic labels;
- raw-origin controls are generated from the selected native group and frozen prior answer spans after search;
- C sees only fixed text views and no native effect/rank information;
- D selects unrelated origin controls from that frozen pool;
- `text_node_ids` still refer only to source text nodes, so the N source-unit coverage check is not polluted by raw-origin control views.

This repair is necessary because mapped previous answer spans can be one to three tokens, while whole-text source controls often have no length-matched unrelated alternatives. The new pool does not solve every coverage case, but it turns the earlier structural zero-coverage problem into an explicit per-origin unresolved outcome.

### Common-prefix / event-level forward accounting remains coherent

Files:

- `route_graph/causal_groups.py:135-206`
- `route_graph/causal_groups.py:208-247`
- `route_graph/causal_groups.py:249-291`
- `route_graph/audit_phase_native.py:93-104`

The main finite event effect remains a complete B/A continuation log-odds difference under shared-prefix interventions. Gate queries and keys are checked against the shared-prefix domain; B/D baseline replay refuses cached effects if token log-probabilities differ; mediated donor capture uses branch-matched complete replays and full-shape sham checks. The implementation no longer treats B search calls as free; proposal and validation forward calls are added in the returned total.

## Optional / cleanup findings

### Optional 1: hard-coded budget constants should be centralized

Files:

- `route_graph/audit_phase_native.py:33-34`
- `route_graph/audit_phase_native.py:96`
- `route_graph/audit_phase_native.py:238-245`
- `route_graph/audit_phase_native.py:308`
- `route_graph/audit_protocol.py:9-16`

The production values currently match the revised plan (`320` per claim, `64` screening forwards, `40` N template, `42` layer bands), but they are duplicated in code and protocol. This can drift again. Prefer reading these from `PROTOCOL` or from a single native-budget config object.

### Optional 2: `CausalOracle` default budget is stale

File:

- `route_graph/causal_groups.py:67-69`

`CausalOracle` still defaults to `budget=256`, while production now uses 320 through explicit calls. This is not a current production bug, but the default is misleading and could affect future tests or ad hoc scripts.

### Optional 3: `root_order` is a declared order, not the actually enqueued roots

File:

- `route_graph/causal_groups.py:483-489`

The returned `root_order` always lists source/history content and broad route roots, even when a root was not actually enqueued due to empty domains or causal-position guards. This is acceptable as a schema hint, but if used as an audit denominator it should be replaced by an `actual_root_order`.

### Optional 4: add regression tests for the two edge cases reviewed here

No repo tests were added in this review. The current full suite passes, but there is no explicit checked-in test for:

- N template position passed while origin failed must not be counted as a full internal certificate, if that is the intended definition;
- current slot linked to a supported prior question in the same previous sentence must not produce `error_history_continuation` just because another slot in that sentence was unsupported.

The second behavior passed in a synthetic in-memory merge check. The first behavior exposed Required finding 1.

## Verification

Command run from `/share/home/tm902089733300000/a903202310/lys/research/graph`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests -q
```

Result:

```text
87 passed in 93.98s (0:01:33)
```

Additional in-memory merge checks:

- N position-only certificate with failed origin increments `risk_claims_with_internal_certificate`: reproduced.
- Same previous sentence with unsupported slot A, supported slot B, and current relation mapped to slot B does not emit `error_history_continuation`: reproduced.

