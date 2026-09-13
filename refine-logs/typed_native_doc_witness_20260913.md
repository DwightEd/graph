# Typed native v2 independent document execution witness — 2026-09-13

Executed the documented GPU wrapper exactly once. Wrapper session **7979**, wrapper PID **167072**, native child PID **167181**; both actual exit codes **0**. Population resume PID **167241** is running and has demonstrated actual growth. This is an execution/integrity witness; the scientific result is **raw native measurements without a passing selective source-path certificate**.

Read the run-experiment skill environment contract, `docs/TYPED_NATIVE_RUN_20260913.md`, and `.aris/compute/local.md` before launch. The canonical declarative environment spec hash is `03909e02`, matching the warm-reuse ledger. No environment install/rebuild, model download, code edit, parameter substitution, extra GPU command, label evaluation, or retry was performed. The graph directory has no `CLAUDE.md`; the supplied explicit local ledger and fixed absolute invocation resolved the environment. The existing environment kernel witness was reused according to the unchanged-spec rule.

Invocation, with the documented working directory supplied to the execution tool:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m experiments.interleave_typed_native --population-pid 162289 --output outputs/typed_native_v2_20260913 --journal ../reanchor/runs/typed_native_v2_20260913.json
```

Wrapper preflight child PID 167078 returned the expected prepared digest. This was the wrapper’s prescribed internal preflight, not a separately repeated CPU prepare command. The wrapper used the Linux pidfd identity check, paused original population PID 162289 at 17,180/17,790 completed with 0 failures, retained the population output lock, and saved 17,180 old manifest hashes before the native model child launched. Every session poll waited at most 20 seconds. The session was retained to its actual exit; no signal was sent to the wrapper.

Journal: `../reanchor/runs/typed_native_v2_20260913.json`; audit log: `../reanchor/runs/typed_native_v2_20260913.audit.log`; population resume log: `../reanchor/runs/typed_native_v2_20260913.population.log`. The final journal records native exit 0 and `population_resume_verified`. That wrapper status verifies matching active PID and a held output lock while loading; the independent post-resume growth observations below complete the documented recovery evidence.

The frozen train denominator is one eligible contrast from one eligible source, selected before effects from the complete 17,790-response CPU parent. Response **9273**, source **14637**, fact index **4** changes the complete event endpoint from **6 PM (original B)** to **4 PM (supported A)**, retaining preceding response text and other event characters. There was no replacement based on native outcomes. Both complete event branches retain 11 scored continuation tokens. All seven source-day endpoints remain fixed as 21 source token keys.

| Quantity | Actual result |
|---|---|
| log P(original B continuation) | -16.244619369507 |
| log P(supported A continuation) | -7.232681274414 |
| Baseline F = log P(B) − log P(A) | -9.011938095093; A-preferred |
| Position removal delta = Fbaseline − Fremoval | -12.862699508667 |
| Input-origin removal through the same A V receivers delta | -15.254831314087 |
| Position half-strength delta | -1.511796951294 |
| Origin half-strength delta | 0.232414245605 |
| Actual model forward calls | 90 = 82 recipient calls + 8 donor calls; cap 128 |
| Native runner seconds | 58.917476; not total wrapper process wall time |
| Certificates | Position false; origin false; overall false |

The observer already prefers supported A at baseline; this result is not a correction of an observer error. The negative raw deltas mean removal moves the B−A margin toward original B. Full removal yields F=3.850761413574 for position and F=6.242893218994 for origin. Strong certificate checks remain closed: the 538-member complete pre-effect control census yielded 2 frozen position controls and 0 origin controls after the two baseline forwards. Both fixed position controls fail the attention-mass check in the selected scope, leaving 0 position and 0 origin controls; no IDs were replaced. Origin half-strength also has the opposite sign, so its dose-direction check fails. Full-strength repeats agree exactly and both sham records are exact.

The selected measured group is **queries 707–710, joint layers 16–31** (zero-based coordinates), with all 21 source-day keys fixed. The root includes all 122 shared-prefix queries 589–710 and all 32 layers. Search used 62 actual forwards for 31 measured groups, including 31 joint-layer groups, 0 single-layer groups, and 4 nonadjacent query unions. The artifact preserves 60 pending groups (8 layer children, 24 query children, 28 unmeasured unions), plus the stated unenumerated subsets. Query complement delta is −0.117204666138; layer complement delta is −4.921039581299. These measurements do not establish an exhaustive search, a unique lookback point, or a per-edge effect decomposition.

Integrity checks were read-only CPU checks of saved artifacts, not another model run. All **17** native manifest artifacts matched. The **41** recipient measurement records each record two actual forwards, including the two original baseline calls; eight donor captures account for the remaining eight forward calls. All 41 prefix checks pass, and the two sham records are exact. The native result, summary, and manifest agree on 90 calls, label-free status, and frozen settings digest.

The actual original-B baseline supplied **32 layers × 143 token positions × 4096 dimensions = 18,743,296 bf16 node values**, stored as uint8 bytes with declared bf16 shapes. The 143 positions contain 122 shared-prefix queries and all 21 source A keys. NPZ layer sets, shape/byte counts, native dtype metadata, and finite bf16 bit patterns passed; no nonfinite values were present. Eight donor forward references map to **6 unique NPZ/manifest pairs**, because full-strength repeat captures reproduce existing immutable values. All 96 unique donor layer byte hashes and stored shapes passed. Complete per-layer raw node/donor hashes and all code hashes are in the machine-readable witness evidence.

| Artifact | SHA-256 |
|---|---|
| settings.json | `895966d2f2ded77df3efad7b6508d3afefa662768e3ef96356c9850fd895dde1` |
| settings object digest (runner serialization) | `36d838b6fe5b52469aaff687f1492a280bab51526069a28b32f4fb6db95acb8a` |
| results/9273_4.json | `b70cdb291505edc0fa4081e4b85df603a3b86e57baa1d195aabcd4a11e15048c` |
| nodes/9273_4.npz | `b7751d66ae990bfdffe13f2ccece3027500063d99f4549ce3e36c785bbf7d34e` |
| summary.json | `b46f20f64cdf57da671221135045b766fef9e1c4122eb3290d1db35f04264593` |
| manifest.json | `aa68821689a3beb716bed121ad946d1983f171a6c43c78853b6ecc4fa1a689d9` |
| frozen parent bridge contrasts/9273_4.json | `8b1b5f98ed773912859cd795a1400dbc29bd4346c99013d31502fa2645c696d1` |
| scheduler wrapper | `0c732f73a44c31c19d0df85ce4a1cd8ad0331afbd7e69ea1ab50c692ed5750bb` |

All **54** native live source files and **54** frozen snapshots match the frozen settings hashes. All **8** population live source files and the four frozen population settings/input/preflight artifacts match their saved hashes. The settings object digest is computed using the runner’s documented `json.dumps(sort_keys=True, ensure_ascii=False)` representation; the compact environment-spec serialization is a separate hash convention.

Population recovery was independently observed at 17,180 → 17,187 → 17256 → 17272 completed, always failed=0, with matching resume PID 167241/cmdline/cwd and a held output lock. At the final observation there were **92 newly published population manifests** relative to the paused snapshot. All **17,180** old manifest hashes and all four frozen population files were checked again after actual growth and remained unchanged. Old wrapper/native/original population PIDs were absent. Final observation time: `2026-09-13T04:07:44.682982+00:00`. Population is still running; this witness does not claim its full 17,790-response run is complete.

Full verification evidence: `refine-logs/typed_native_integrity_evidence_20260913.json`. The native manifest enumerates all raw/node/control/donor artifact file hashes; the witness evidence records independent actual hashes, per-layer raw byte hashes, the old-manifest snapshot digest, code checks, and post-resume new manifest hashes.

No execution-command divergence or wrapper error occurred. The scientific limitation is retained exactly: raw effects in the Llama 3.1 observer replay on one narrow typed-hours contrast, with no passing selective source-path certificate, no independent B-source occurrence, unresolved routing and aggregation, and no original-generator causality or unique-lookback claim. No RAGTruth label evaluation was performed.
