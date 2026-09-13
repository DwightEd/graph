# Ownership engineering review, round 4 — 2026-09-12

Separate engineering-review agent, continuing
`/root/.agents/skills/code-review-and-quality/SKILL.md`. Earlier reviews are
preserved. Scope: `OWNERSHIP_SOURCE_LAYER_PLAN_20260912.md`,
`ownership_source_layer.py`, its launcher, and
`analyze_ownership_source_layer.py`. Supplementary checks read the completed raw
artifacts on CPU. No implementation or raw-output edits and no GPU work were
performed by this reviewer.

## Verdict

**Engineering review: approve. No unresolved Required issue.** Independent
scientific review remains unavailable; this is not scientific approval.

One Required issue was found and fixed during review: the analyzer obtained
`prompt_length` from a live original trace without checking it against the saved
input hash. Run-manifest validation alone did not authenticate that external
dependency. The analyzer now compares `sha256(original)` with the recorded input
hash before loading the trace, preventing a changed original file from silently
altering fixed-token evaluation indices. The fix was inspected in the final code.

## Real inputs and independent source edits

Independently loaded and decoded all four original traces. Their prompt lengths
are 535. The sample metadata identifies source **14375** with seeds **0, 1, 2, 3**
for `00012` through `00015`; these are four responses to one source.

| Trace | Grill prediction / query / token | Second prediction / query / token |
| --- | --- | --- |
| 00012 | t99 / 633 / 14 | t131 / 665 / 12 |
| 00013 | t100 / 634 / 14 | t126 / 660 / ` over` |
| 00014 | t98 / 632 / 14 | t61 / 595 / 20 |
| 00015 | t109 / 643 / 14 | t142 / 676 / 10 |

The decoded second windows match the declared onion number, continued onion
cooking, simmering for 15–20 minutes, and alternative grilling for 5–10 minutes.
The primary grill windows all contain the observed 10–14-minute instruction.

A changes only absolute prompt position 76 from token 717 (12) to 975 (14).
B changes only position 341 from 975 to 717. H changes only the current trace's
observed grill-upper-bound token at `535 + history_step` from 14 to 12. Checked
all **32 persisted token arrays** against their original traces: every changed
position set exactly matches the declared active A/B/H edits. All other tokens,
including the complete remaining observed answer, are preserved. A/B together
preserve the prompt token multiset; individual A or B does not, as disclosed.

The four A=B worlds for `00012` reproduce the preceding source/history
factorial, with all saved reproduction errors 0. The code compares H0/H1 at the
grill prediction and any selected earlier control readout, using `step <=
history_step`. In particular, the t61 simmer control is correctly included even
though it is stored after the t98 grill readout in the case dictionary. Verified
raw equality for these selected future-H checks across all four A/B settings.

## Factor reuse and the 64 single-layer interventions

The prior multilayer manifest is verified before donor factors are loaded.
Independently confirmed that the current source mask produces exactly the same
ordered **427 source positions** as the prior run's saved `source_positions`.
The donor attention axis has shape `[32, 132, 427]`; donor values have shape
`[427, 8, 128]` at the inspected first/last layers. The donor query axis is indexed
by prediction step, so `weights[:, [99]]` and `weights[:, [131]]` are correct.
The actual patch queries remain one token earlier: 633 and 665. X uses current
receiving A and the reused donor V.

The driver enumerates all zero-based layers 0–31 at each endpoint, for **64**
conditions, and reads all 132 original answer predictions on every patched
forward. `full_logits[:step]` checks 99 earlier predictions for a grill patch and
131 for an onion patch. These checks are nonempty and have recorded error 0 for
every condition. Inputs are complete observed token sequences; the 132-query
readout list does not truncate the forward context.

Full-prefix checks apply to this single-layer branch. The A/B/H source-world
branch saves and checks its two declared readouts per response; it does not save
a 132-query trajectory for each world. No broader raw-prefix validation is claimed
here.

The four previously chosen layers are compared at both windows, giving eight
repeated single-layer X margins. The driver checks exact agreement. Every new
condition saves both endpoints' full-vocabulary logits, even though its summary
describes the selected target endpoint. The exhaustive spectrum prevents silently
skipping an untested dominant layer; it does not imply additive layer effects.

## Analysis and numeric checks

Verified all **139** files in the completed source/layer run manifest. Then checked
the 32 world arrays and all 64 single-layer raw outputs against their records:

- All **64 source-world endpoint** margins, native argmax IDs, and tie counts
  exactly match raw logits.
- Reconstructed all five declared numeric A/B/H margin cubes and their three
  mean main effects directly from raw logits; they exactly match the summary.
- All 64 single-layer target margins, margin changes, native argmax IDs, tie
  counts, and argmax-change flags match their raw endpoint logits and the prior
  baseline. Both window spectra have 0 native argmax flips in the saved run;
  this is an observed result, not an implementation requirement.
- Earlier-query counts are exactly 99/131 and recorded errors are 0 throughout.

The analyzer's `before`/`after` construction inserts the varied bit at the correct
A/B/H axis while retaining the other two bits. It therefore generates four
conditional comparisons per factor/window/trace: **96 endpoint contrasts** in
total. Fixed observed-token log-probabilities use `prompt + prediction_step`.
Both settings and world records preserve the driver dictionary order, so the
two-row native logit arrays align with the named readouts, including the
chronologically earlier t61 control.

Numeric 14-minus-12 margins are retained for the five declared grill/onion numeric
readouts. They are explicitly blank in exported rows for the three native control
readouts. Their full-vocabulary JS, observed-token log-probability change, and
native outcome remain available. `mean_js_nats` averages four distribution
distances and is labeled accordingly; it is not presented as a signed numeric
factorial main effect. All raw conditional comparisons are exported.

The layer analyzer uses row 0 for grill and row 1 for onion, checks native argmax
and tie counts against raw arrays, reports the full 32-layer spectrum, and labels
the maximum absolute effect as a descriptive result. It does not rename the best
layer as a predeclared method or infer that layers add linearly.

## Optional and scope limits

For future reuse, consider asserting equality of the current source-position list
with the prior run's saved list before applying saved factors. That invariant was
independently verified for this run, so no current source-axis mismatch exists.
An explicit guard would make the reused-factor contract clearer if traces or
masks change later.

Fresh output directories, input/code hashes, executed code snapshots, offline
model loading, bounded loops, and four-thread launch settings are retained. The
plan explicitly describes the work as exploratory after prior results, keeps
four same-source seeds distinct from independent sources, and preserves one
observed incorrect numeric window as the error scope. No detector is fitted;
no accuracy, unique-node truth, factual repair, cross-source generalization, or
independent scientific approval is established by these checks.
