# Ownership multilayer documented invocation: independent execution witness

Date: 2026-09-12. This witness is by the fresh execution subagent, separate from the experiment author. It records one real invocation and read-only CPU checks; it is not independent scientific approval.

## Invocation and preflight

Read `/root/.agents/skills/run-experiment/SKILL.md`, its compute environment contract, `graph/.aris/compute/local.md`, and `graph/docs/OWNERSHIP_MULTILAYER_PLAN_20260912.md`. No applicable `AGENTS.md` or `CLAUDE.md` was found in the ancestor directories or graph/reanchor trees. Canonical spec hash was independently recomputed as `8d044d57`, matching the reused environment ledger. No environment rebuild, installation, or additional CUDA witness was performed.

After the parent froze the final per-query tie-count logging update, executed the documented command verbatim once from the reanchor root:

```bash
bash scripts/run_ownership_multilayer.sh --output outputs/ownership_multilayer_20260912
```

The shell wrapper used `set -o pipefail` and `2>&1 | tee` to save stdout/stderr at `/tmp/ownership_multilayer_witness_20260912.SZZwyZ.log`. The log is outside the artifact directory. That directory did not exist before launch. GPU 0 was NVIDIA GeForce RTX 4090, 1 MiB / 24564 MiB, with no compute process immediately before launch. The only runner process observed in this namespace was PID 13944. No retries, code edits, overwritten artifacts, or additional GPU runs were performed by this witness.

Command exit code: **0**. Runner duration: **58.3940505963 seconds**, not a separately measured end-to-end shell wall duration. CUDA peak allocated: **17,463,182,848 bytes = 16.2638564 GiB**. After completion, GPU returned to 1 MiB and no compute processes. Recorded runtime versions: torch 2.8.0+cu126, transformers 4.57.1. There was no invocation/documentation failure.

## Independent read-only CPU checks

- Exactly **98 unique conditions**, with the complete predeclared matrix: 48 layer-group/window/scope/factor combinations, 4 same-world sham conditions, 40 only-one/leave-one-out conditions, and 6 endpoint swaps. Every condition's name, layer list, query selection, factor, and window matched an independently constructed expected matrix.
- Manifest **116/116** SHA256 hashes passed; its key set exactly equals the output file set excluding manifest.json itself. All **6** executed-source snapshots match their recorded code hashes and current source files. All **8** recorded input hashes match the actual input files. No verification file was written inside the artifact directory.
- Two source worlds each retain full **[132, 128256]** vocabulary logits, and all 98 interventions retain **[2, 128256]** endpoint logits; all values are finite. All endpoint candidate margins, margin changes, native argmax IDs, and exact maximum tie counts were independently recomputed from these arrays and match the JSON.
- Each of the 98 records has 132 native argmax IDs, 132 maximum tie counts, and 132 values in every trajectory field: JS, argmax-change indicator, saved-token log-probability change, and altered entropy. Tie counts are positive and numerical trajectory arrays are finite.
- The source world token arrays both have length 731. Their only differences are positions **76 and 341**, exchanging IDs **717 (12)** and **975 (14)**. The 132 query positions are consecutive **534–665**; source positions total 427. Each world's factor archive has 96 finite arrays (three factors across 32 layers); layer-zero shapes are V [427, 8, 128], A [32, 132, 427], and MLP [132, 4096].
- Baseline reproduction was independently recomputed against the prior factorial arrays over their 20 recorded queries: S0H0 **0.0**, S1H0 **0.0** maximum absolute logit errors.
- All **4 sham** records report full-trajectory maximum logit error **0.0**. Their stored endpoint arrays are independently byte-value identical to baseline, and all 132 argmax/tie values agree. Full intervention trajectories of vocabulary logits were checked at runtime by the runner, but only endpoint logits are saved for later independent numerical recomputation.
- **76 conditions** have a nonempty causal prefix containing 93–131 queries; all report runtime prefix maximum logit error **0.0**. CPU checks independently confirmed every such prefix has identical native argmax/tie values and zero saved-token log-probability change. The remaining **22** begin at t0 and correctly report null prefix error with zero earlier queries; these cannot establish a nonempty prefix invariant. Because patched full-trajectory vocabulary logits were not saved, the logged prefix maximum logit errors cannot be independently recomputed after execution.
- Maximum reported per-layer source-mass preservation error: **3.5762786865e-7**. This is a recorded runtime diagnostic, not a reconstruction of all interventions from saved factors. JS for exactly unchanged prefixes still reaches **2.8978185185e-8** from numerical calculation; it must not be read as a causal effect preceding intervention.

The standalone verification summary is `/tmp/ownership_multilayer_witness_20260912.verification.json`. CPU verification used NumPy with CUDA_VISIBLE_DEVICES empty and did not load or rerun the model.

## Observed results

All margins below are **z(14) − z(12)** at the named endpoint. S0H0 baseline is grill **+7.125** (native 14) and onion **−10.375** (native 12). S1H0 donor baseline is grill **−6.0** (native 12) and onion **+7.875** (native 14), all unique maxima.

| Endpoint | All-32-layer intervention | Final query | Original window | All history | Native result across these three scopes |
|---|---|---:|---:|---:|---|
| grill | all_x | -8.0 | -9.0 | -9.125 | 12, 12, 12; unique maxima |
| grill | all_e | 7.75 | 7.875 | 7.75 | 14, 14, 14; unique maxima |
| grill | all_xe | -3.25 | -4.25 | -4.25 | 12, 12, 12; unique maxima |
| grill | all_mlp | 1.875 | 1.375 | 1.375 | 14, 14, 14; unique maxima |
| grill | endpoint_swap | -6.375 | -7.125 | -7.625 | 12, 12, 12; unique maxima |
| onion | all_x | 6.875 | 8.125 | 8.0 | 14, 14, 14; unique maxima |
| onion | all_e | -10.5 | -10.375 | -9.5 | 12, 12, 12; unique maxima |
| onion | all_xe | 5.5 | 7.25 | 6.75 | 14, 14, 14; unique maxima |
| onion | all_mlp | -1.75 | -1.375 | -0.875 | 12, 12, 12; unique maxima |
| onion | endpoint_swap | 7.375 | 8.375 | 8.0 | 14, 14, 14; unique maxima |

The four-layer interventions have **0 native endpoint flips among 24 conditions**. All-layer final-query X alone changes grill's native maximum to 12 and onion's to 14. Across the 94 non-sham interventions, 38 change the native maximum at their selected endpoint; this is a descriptive intervention count, not detection accuracy. Endpoint swapping also changes both selected native maxima in all three scopes, while donor E alone does not. These are mechanism outcomes for known source endpoints.

The predeclared individual-query results are retained below without selecting a favorable query after seeing them:

| Window | Query t | Only this query: margin | Window excluding this query: margin |
|---|---:|---:|---:|
| grill | 93 | 7.125 | -9.0 |
| grill | 94 | 7.125 | -9.0 |
| grill | 95 | 7.125 | -8.875 |
| grill | 96 | 7.0 | -8.875 |
| grill | 97 | 6.0 | -8.5 |
| grill | 98 | 7.0 | -9.0 |
| grill | 99 | -8.0 | 5.875 |
| onion | 119 | -10.375 | 8.125 |
| onion | 120 | -10.375 | 8.125 |
| onion | 121 | -10.375 | 8.0 |
| onion | 122 | -10.375 | 8.125 |
| onion | 123 | -10.5 | 8.0 |
| onion | 124 | -10.375 | 8.125 |
| onion | 125 | -10.375 | 8.125 |
| onion | 126 | -10.25 | 8.0 |
| onion | 127 | -10.375 | 8.125 |
| onion | 128 | -10.25 | 7.875 |
| onion | 129 | -8.875 | 7.0 |
| onion | 130 | -10.375 | 8.125 |
| onion | 131 | 6.875 | -8.75 |

Within this fixed all-layer-X window intervention, the final query is sufficient to reverse the selected native maximum, and removing it from the full window prevents that reversal. This does not establish a unique earlier reasoning or screening node; only-one sufficiency and leave-one-out conditional necessity are different comparisons, and the effects are not assumed additive.

The 00013 t126 clean-control full vocabulary resolves the earlier top-k ambiguity: S0 has a two-way exact maximum tie at IDs **304 (" in")** and **927 (" over")**, with native argmax 304 by the library's tie rule. S1 has a unique maximum at **927 (" over")**. Therefore this reported native argmax change includes resolution of a baseline maximum tie. Candidate margins are −1.0625 and −1.001953125 respectively; this control endpoint is not a native number-token maximum.

## Limits

This is the post-factorial exploratory second round on one source and one actual erroneous numerical window. The full original prompt and response were held fixed; this witness did not generate candidate responses, tune the matrix, train a detector, or measure error-detection performance. Swapping 12/14 does not make the onion continuation's duration legally supported, so a numeric or native-argmax flip cannot be called error repair. The known-endpoint swap is not an unlabeled detector or proof that a complete graph is necessary. MLP output donation does not isolate parameter knowledge. These data do not provide cross-source generalization, p-values, a detection success rate, or perfect localization.

The reused environment was validated for this exact documented invocation, not for a clean installation. Weight identity uses recorded model-file size/mtime metadata rather than full weight-file hashes. The external scientific review backend remains **REVIEW_UNAVAILABLE**. This witness certifies the stated execution and artifact checks only.
