# Independent documented-command witness: ownership factorial, 2026-09-12

Fresh executor agent `/root/ownership_run_witness` read the run-experiment skill,
its shared compute-environment contract, `graph/.aris/compute/local.md`, and
`graph/docs/OWNERSHIP_FACTORIAL_PLAN_20260912.md`, then executed the documented
command **once**, without code edits, fixes, retries, output deletion, package
installation, or Git operations:

```bash
# Working directory: /share/home/tm902089733300000/a903202310/lys/research/reanchor
bash scripts/run_ownership_factorial.sh --output outputs/ownership_factorial_20260912
```

**Exit status: 0.** Output:
`reanchor/outputs/ownership_factorial_20260912`.

## Environment and execution

- Canonical environment-spec SHA256 prefix independently recomputed as
  `8d044d57`, matching the existing ledger. The environment was reused.
- Preflight: GPU 0, NVIDIA GeForce RTX 4090, 1 MiB used of 24,564 MiB.
  Output directory did not exist before launch.
- The wrapper selected the existing `research/bin/python`, four CPU threads,
  offline model loading, bf16, and eager attention. Saved settings report
  torch `2.8.0+cu126` and transformers `4.57.1`.
- GPU process observed: PID `1748509`, 17,392 MiB device memory reported by
  nvidia-smi. It exited; subsequent nvidia-smi showed 1 MiB used.
- Runner-reported elapsed time: **12.760660648345947 seconds**. This timer starts
  inside `run()` and excludes earlier imports and final output serialization;
  it is not an independently measured shell wall time.
- Invocation timestamp was `2026-09-12T14:24:38.387Z`; successful exit had been
  observed by `2026-09-12T14:25:18.881Z`. The 40.494-second observation interval
  includes inspection/polling delays and is only an upper bound on command wall
  time.
- Runner-reported peak CUDA allocated memory: **17,377,844,736 bytes
  (16.1843791008 GiB)**.

## Completion and numerical checks

The complete predeclared Cartesian product exists exactly once: four layers
`7, 15, 23, 31`, two windows, two scopes, and four factors. There are **64 local
patches** (16 X, 16 E, 16 X+E, 16 MLP), **4 same-world shams**, four S×H worlds
with both readouts, and two correct-onion source conditions. All four saved
world-logit arrays have shape `(20, 128256)` and contain only finite values.

All four recorded same-world shams have exact `max_logit_error = 0.0`,
`delta_l2 = 0.0`, and source-mass error `0.0`. These compare every vocabulary
logit at the 20 captured queries; the experiment does not save logits for every
sequence position. The patch logits themselves are not archived, so this
witness verifies the saved diagnostics and executed runtime checks rather than
independently recomputing sham equality from raw patch logits.

All 64 patch records report `earlier_query_max_logit_error = 0.0`. Of these,
48 compare at least one captured query before the intervention; the 16
whole-grill-window conditions have no earlier query in the captured set and
therefore record a zero for an empty comparison. This limits the coverage of
the check; it is not evidence about uncaptured earlier positions.

Independently comparing the saved S0H0/S0H1 and S1H0/S1H1 arrays at all seven
grill queries found zero differing elements and maximum error **0.0** in both
cases. Thus the future history edit does not change any captured grill logit.

The maximum recorded source-mass preservation error across all diagnostics is
**2.384185791015625e-07**. Token arrays independently show exactly the intended
changes: S0H1 changes position `634`; S1H0 changes positions `76, 341`; S1H1
changes positions `76, 341, 634`. The source-only swap preserves the full token
multiset and sequence length.

## Observed outputs

Margins below are `z(14) − z(12)`. Top-1 was independently recovered using
argmax over the saved full vocabulary, not inferred from top-k ordering.

| World | Grill margin | Grill top-1 | Onion margin | Onion top-1 |
|---|---:|---|---:|---|
| S0H0 | 7.125 | 14 | -10.375 | 12 |
| S0H1 | 7.125 | 14 | -9.000 | 12 |
| S1H0 | -6.000 | 12 | 7.875 | 14 |
| S1H1 | -6.000 | 12 | 8.250 | 14 |

| Readout | Source main effect | History main effect | Interaction |
|---|---:|---:|---:|
| Grill | -13.125 | 0.000 | 0.000 |
| Onion | 17.750 | 0.875 | -1.000 |

Across all 16 conditions per local factor, observed margin-change ranges are:
X `[-3.25, 1.75]`; E `[-0.25, 0.0]`; X+E `[-2.875, 1.5]`; MLP
`[-1.0, 1.5]`. None of the 64 patch records reports a native argmax change.
These are complete-factor ranges, not selected-layer claims.

The correct-onion control reports full-vocabulary JS `0.01564033329486847`
nats, saved-token log-probability change `+0.3446977138519287`, and
`argmax_changed = true`. Its displayed baseline top logits tie at `26.125`
for ` over` and ` in`; the top-k list begins with ` over` in both conditions.
Top-k ordering in a tie must not be used as native argmax evidence. Raw control
logits are not saved, so its exact argmax identities cannot be independently
recovered from this artifact.

## Documentation versus artifact

The invocation, environment reuse, condition counts, fixed layers/scopes,
fresh-directory requirement, and successful numerical runtime guards agree
with the document. No invocation or runtime failure required intervention.

The B1 plan promises complete-vocabulary JS, but `worlds.json` and
`summary.json` do not serialize these world contrasts. The saved raw logits
allow them to be recovered without another model run. As a witness supplement,
I independently computed the following using float64 softmax and
`0.5 KL(P || M) + 0.5 KL(Q || M)`, `M=(P+Q)/2`, over all 128,256 tokens:

| World contrast | Grill JS, nats | Onion JS, nats |
|---|---:|---:|
| S0H0 vs S1H0 | 0.6807432367661049 | 0.6901943669247282 |
| S0H1 vs S1H1 | 0.6807432367661049 | 0.6892713555573674 |
| S0H0 vs S0H1 | 0.0 | 0.00013303866526435343 |
| S1H0 vs S1H1 | 0.0 | 0.0005727048266754487 |

These supplemental numbers were computed after the run and were not written
back into its frozen output directory. The other reporting limits are the
unmeasured exact shell wall time, the restricted causal-check coverage, and
the top-k/argmax ambiguity described above.

## Artifact verification and scope

- Independently recomputed and verified **18/18 manifest SHA256 entries**.
- Independently verified all **3/3 recorded code hashes** and **4/4 recorded
  input hashes** against their referenced files at verification time.
- The output preserves three executed-source snapshots. Its manifest excludes
  itself, as expected.
- The input-hash list does not include the two state NPZ files used to load
  source masks; settings/control do record the resulting mask digest. Model
  identity is recorded through path, filenames, sizes, and mtimes, rather than
  cryptographic weight hashes. These are provenance limits, not execution
  failures.
- `summary.json` explicitly marks scientific review `REVIEW_UNAVAILABLE`.
  CPU tests were outside this execution witness and were not rerun here.

This is an execution and numerical-integrity witness in the reused local
environment. It does not certify a clean installation, scientific review,
ownership localization, detection accuracy, or repair of the onion answer.
