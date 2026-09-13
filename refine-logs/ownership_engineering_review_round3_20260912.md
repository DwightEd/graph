# Ownership engineering review, round 3 — 2026-09-12

Final bounded follow-up by the separate engineering-review agent using
`/root/.agents/skills/code-review-and-quality/SKILL.md`. Prior review records are
preserved. Scope: round-2 reporting fixes, `analyze_ownership_factorial.py`, and
read-only numeric checks against completed factorial/multilayer raw artifacts.
No implementation or raw output was edited; this reviewer did not run GPU work.

## Verdict and issue closure

**Engineering review: approve the reviewed fixes and analysis implementation.**
No Required issue remains in the reviewed code. This verdict does not constitute
independent scientific approval; `REVIEW_UNAVAILABLE` remains in force.

- Round 1 hook cleanup: fixed and independently verified in round 2.
- Round 1 B1 full-vocabulary JS: implemented in the new analyzer. Both source
  contrasts at fixed history and both history contrasts at fixed source are
  computed from persisted full-vocabulary logits, separately for grill and onion.
  This reviewer independently checked those calculations. Publishing the derived
  analysis artifact remains the parent's execution task.
- Round 2 per-query ties: `native_max_tie_counts` now compares each row with its
  own maximum and records all 132 counts. Confirmed both in code and in all 98
  completed multilayer condition records.
- Round 2 environment provenance suggestion: the driver now freshly records
  torch/transformers versions, GPU name, and model-file metadata. The completed
  run records torch 2.8.0+cu126, transformers 4.57.1, and RTX 4090.
- During this final review, the analyzer's conditional contribution had been
  labeled as the change produced by removal despite computing full minus
  leave-one-out. The parent changed the field to `full_minus_leave_one_out` and
  the legend to “Full minus leave-one-out.” The signed quantity is now explicit
  and consistent with its formula.

## Independent B1 numeric and index checks

Verified all **18** files listed in the original factorial manifest. Loaded all
four raw world arrays and independently computed margins, factorial contrasts,
and JS using NumPy float64, alongside the analyzer's shared float32 metric.

Grill uses saved-logit row 6, absolute query 633, and saved target token 975 (14).
Onion uses row 19, query 665, and target token 717 (12). The analyzer obtains each
fixed observed target from `tokens[settings['queries'][row] + 1]`; this is the
correct next-token index. It consistently measures the same observed target
across the artificial worlds rather than changing the evaluation target with the
intervention.

The raw margin matrices, indexed `[source, history]`, are:

- Grill: `[[7.125, 7.125], [-6.0, -6.0]]`.
- Onion: `[[-10.375, -9.0], [7.875, 8.25]]`.

They exactly reproduce the saved main effects and interaction: grill source
−13.125, history 0, interaction 0; onion source +17.75, history +0.875,
interaction −1. No favorable condition is substituted for a factorial effect.

Independent full-vocabulary JS, in nats:

| Contrast | Grill | Onion |
| --- | ---: | ---: |
| Source change at H0 | 0.680743237 | 0.690194367 |
| Source change at H1 | 0.680743237 | 0.689271356 |
| History change at S0 | 0 | 0.000133039 |
| History change at S1 | 0 | 0.000572705 |

The maximum absolute difference between the shared float32 implementation and
the independent float64 computation was **9.45e-8**. Grill history logits are
exactly identical; the float32 JS implementation can nevertheless return roughly
1e-8 because of reduction roundoff. This is numerical noise, not a nonzero
history effect. Fixed-token log-probability effects also agreed within the
expected float32 precision.

## Completed multilayer raw-artifact checks

Verified all **116** files listed in the new run's manifest, then checked all
**98 conditions / 196 endpoint rows** against the saved raw endpoint vocabulary
logits:

- Endpoint margin, margin change, native argmax ID, and tie count exactly equal
  their recorded values. Endpoint argmax/tie entries also equal the corresponding
  t99/t131 entries in the 132-query vectors.
- Every condition has both 132-element native argmax and tie-count vectors.
- Recomputed endpoint JS and fixed-token log-probability changes using the same
  next-token indexing as the analyzer. Maximum JS difference was **5.96e-8**;
  fixed-token log-probability differences were **0**. Entropy differed by at most
  **1.19e-7**, consistent with single-row versus full-trajectory float32 reduction.
- All four shams' raw endpoint logits exactly equal the baseline endpoint logits.
  The recorded full-query sham errors are zero.
- All recorded earlier-query errors are zero, or null where no earlier answer
  query exists. The raw patch artifacts contain endpoints only, so this reviewer
  did not independently reconstruct every earlier query from raw logits.
- Both source worlds' recorded reproduction errors against the first run are 0.

The analyzer uses the declared window end minus one to index multilayer
trajectory fields, correctly selecting t99 and t131. Its individual-query table
pairs each `only_*` and `leave_*` condition with the corresponding full-window,
all-layer X intervention. `full_minus_leave_one_out` is a conditional contribution
contrast, not an assumed additive decomposition or a unique-node label.

## Remaining scope limits

The analyzer validates input manifests before producing new output, verifies
world margins against raw logits, preserves old run directories, and records
analyzer/metric-code hashes and input-manifest hashes for derived artifacts.
Its plots and tables describe the frozen interventions; their existence does
not supply scientific validation. Numeric influence, argmax changes, and
conditional query contributions remain observations on these fixed traces, not
factual repairs, detection performance, or generalization evidence.
