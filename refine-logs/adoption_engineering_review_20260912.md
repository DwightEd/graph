# Evidence adoption pilot: independent engineering review

Date: 2026-09-12. Skill: `/root/.agents/skills/code-review-and-quality/SKILL.md`.

**Verdict: approve the engineering instrument for the declared fixed-case pilot, subject to inspection of the separate real-model smoke results. No Required or Critical findings in the reviewed version. This is not scientific approval, and does not change `REVIEW_UNAVAILABLE`.**

## Scope and method

Reviewed only the new implementation, tests, protocol, driver, and launcher listed below. Read the tests and protocol before the implementation. Preserved all existing dirty and untracked files; made no implementation edits, commits, resets, or cleanup. Used no GPU. Existing surrounding research implementations and findings were outside this review.

The instrument measures signed terminal-layer attention messages against fixed output-token contrasts, and full-distribution changes under fixed-continuation key-access cuts. The protocol correctly distinguishes these measurements from factuality, semantic contradiction, or a validated detector.

## Required findings

None for the declared `00012.npz`, Llama-3.1-8B-Instruct, eager/bf16, `cuda:0` pilot. The checks below establish engineering consistency; actual 8B numerical quality and feasibility remain the separate smoke run's responsibility.

## Optional findings and scope clarifications

1. **Optional — state explicitly that finite attenuation, MLP comparison, and opposition summaries use native top-1 versus top-2 only.** `route_graph/adoption.py:125-153` uses only contrast index 0 for these summaries, while `candidate_group_head_contributions` contains every candidate contrast. The protocol at `docs/EVIDENCE_ADOPTION_PLAN_20260912.md:42-54` does not explicitly narrow those diagnostics to top-2. Either name the summary contrast in the protocol/artifacts, or extend the finite and MLP checks across candidates before interpreting source-token alternatives as equally validated. This is a documentation/coverage distinction for the present top-2 diagnostic, not evidence that the other derivatives are wrong; an independent CPU check of all seven candidate derivatives passed.

2. **Optional — qualify `surrogate_top1_matches_native` as a candidate-pool comparison.** At `route_graph/adoption.py:149`, `logits.argmax() == 0` searches selected output rows only. It establishes that the native top token remains best in that pool. It cannot establish the global fp32-surrogate argmax over uncomputed vocabulary rows. Rename it to `surrogate_candidate_top1_matches_native`, or document this field's scope. Avoid citing it as a full-vocabulary surrogate agreement check.

3. **Optional for this fixed cuda:0 invocation — derive CUDA provenance and memory statistics from `args.device`.** `src/decoding/adoption_probe.py:119` and `:242` call CUDA metadata functions without an explicit device, while the model is moved to `args.device` at `:144`. A direct run on `cuda:1` can report GPU 0's identity and memory instead of the device used; `--device cpu` also fails on CUDA metadata even though argparse accepts it. Resolve the requested device once and pass it to the metadata/statistics calls, or constrain the CLI to the supported device. This does not affect the declared default-device pilot.

4. **Optional — preserve exact model-content provenance before scaling or comparing archived runs.** The driver hashes the sample file, sample settings, source-mask bytes, core module, and driver; the output manifest hashes artifacts. Model files currently have only names, sizes, and modification times (`src/decoding/adoption_probe.py:129-133`). These are useful inventory metadata but are not content identity. A cached checkpoint manifest/content digest would make future reproduction robust to replacement of files at the same local path. No new dependency is necessary.

5. **Optional — retain the actual-model integration checks as a focused regression test when this instrument becomes reusable.** The five tests validate mathematical helpers, the native fp32 suffix, and one centered derivative. They do not themselves exercise `capture_terminal` or pass a custom mask through a real model. The independent CPU checks performed for this review cover those interfaces now, including bf16 execution; a small persistent integration test would protect against future Transformers hook/mask changes.

## Correctness and design assessment

- **V/O/GQA reconstruction:** actual `v_proj` outputs and returned final-layer attention weights are captured. GQA repetition is contiguous by KV head, matching the tested Llama implementation. Weight blocks preserve per-head signs; grouping occurs over keys before a linear projection, which commutes with summation. Reconstruction is checked against the actual attention module output, and a biasful O projection is explicitly rejected.
- **Residual and suffix:** the pre-hook on final `post_attention_layernorm` captures the residual after attention addition. The fp32 suffix applies the copied post-attention RMSNorm, final MLP, residual addition, final RMSNorm, and selected LM output rows. All model/suffix weights are frozen while the captured residual alone requests gradients. This is the declared local suffix surrogate, not an all-layer gradient.
- **Gradient and attenuation sign:** the stored contribution is the derivative of the native top-token contrast along each message. Subtracting `rho * message` and reporting baseline-minus-altered contrast has the same sign as `rho * contribution`. Per-head signed contributions are saved before group aggregation. Finite rho values characterize nonlinearity; they are not exact derivative tests at rho = 1.
- **Causal positions:** input tokens are `ids[:-1]`; response prediction t reads query `P + t - 1`. Blocking history keys `[P+119, P+133)` or `[P+85, P+99)` beginning at query `P+133` therefore first changes prediction t = 134. Prefix comparisons use `[:134]`, correctly including t = 0 through 133. Both interventions preserve all saved continuation tokens. The mask preserves the diagonal and all causal exclusions.
- **Distribution statistics:** comparisons use full-vocabulary logits, natural logarithms, the symmetric JS definition, fixed saved-token log-probability differences, and changed argmax indicators. Source-token candidate restriction is not applied to the influence metric. The sham compares ordinary native inference to an explicit causal mask.
- **Input/output safety and provenance:** inputs use `allow_pickle=False`, saved token identity is checked, the expected mask length is checked, trace paths must be filenames, an existing output directory is rejected, and JSON disallows NaN. Local-only loading and offline launcher settings avoid implicit model downloads. No secrets or external messages are introduced.
- **Architecture and readability:** the reusable mathematical instrument and diagnostic orchestration are separate, no existing capture/intervention code is modified, and no new package dependencies are introduced. File sizes are 183, 97, 95, 267, and 8 lines; no decomposition blocker is present.
- **Resource scope:** the inspected case has 731 total tokens, P = 535, and 196 response tokens. The run is finite: one native capture, a bounded query loop, and three influence forwards. Backward passes are restricted to the final fp32 suffix with batches of 16 contrasts. Capturing all layers' attention temporarily materializes approximately 1.09 GB of bf16 attention tensors at length 730, in addition to model, logits, and suffix memory. This is acceptable to assess in the separately owned smoke, but the driver has no generic sequence-length/candidate cap and should not be presented as resource-bounded for arbitrary trace replacements.
- **Scientific boundary:** the protocol expressly describes a previously inspected single case, access cuts rather than semantic erasure, fixed mediators, and the need for independent review and later held-out evaluation. None of the engineering tests validates detection, graph benefit, calibration, factual correctness, or generalization.

## Verification performed

Environment: Python 3.11.15, PyTorch 2.8.0+cu126, Transformers 4.57.1. Every Python check used the existing research conda interpreter with OMP/OpenBLAS/MKL thread counts of 4. No installation or GPU access was used.

1. `PYTHONPATH=graph:reanchor/src .../conda_envs/research/bin/python -m pytest -q graph/tests/test_adoption.py`: **5 passed in 22.47 seconds**.
2. `bash -n reanchor/scripts/run_adoption_probe.sh`: passed.
3. Independent tiny actual eager Llama CPU integration, seed 3, 2 layers, hidden size 16, 4 attention heads / 2 KV heads, 9 input tokens:
   - Captured values `(9, 8)`, attention `(4, 3, 9)`, and attention outputs `(3, 16)` for queries 3, 6, and 8.
   - fp32 message closure maximum absolute errors: `4.6566e-10`, `5.8208e-10`, `6.9849e-10`.
   - fp32 sham maximum logit error: exactly 0.
   - A keys `[2, 3]`, activation 5 cut gave exactly zero error on rows 0–4 and nonzero effects on rows 5–8.
4. The same tiny-model integration in CPU bf16:
   - Message closure maximum absolute error: `7.5367e-6`.
   - Explicit fp32 suffix versus native bf16 maximum selected-logit mismatch: `8.3336e-4`.
   - Relative L1 finite-attenuation errors: `2.3460e-4` at rho = 0.1, `2.3088e-3` at rho = 1.
   - Centered finite differences for every one of seven top-1-versus-candidate derivatives, using an actual reconstructed group message, had maximum absolute disagreement `1.1015e-6`.
   - bf16 sham maximum logit error: exactly 0; causal prefix again unchanged exactly, and every row at/after activation affected.

These are synthetic numerical/integration checks, not measurements on the research case. The real-model smoke agent owns that evidence. No broader unrelated repository test results are claimed here.

## Reviewed source identity

SHA-256 snapshots before any subsequent author changes:

| File | SHA-256 |
| --- | --- |
| `graph/route_graph/adoption.py` | `c7515df5ff57d6ddedd5a2283ea87bcdc2d3f8e0c27f3088022a79df677762bb` |
| `graph/tests/test_adoption.py` | `6dd82909fbc8f226f9707b70d65a6fcd57fa03c9502549e732b6436ec914ecb4` |
| `graph/docs/EVIDENCE_ADOPTION_PLAN_20260912.md` | `e98e3bea3f963e078d5d62e0703039a4e17c5110edb3ab1b47772327b6d36349` |
| `reanchor/src/decoding/adoption_probe.py` | `b56699d7e019408a4cdec8328cf0ab294cb84f6f45dfd221b476c734fb900873` |
| `reanchor/scripts/run_adoption_probe.sh` | `4402b853228afbe2d628f64485580c0153586059f50fb1f5491a58243c92e9da` |

## Delta review and actual smoke inspection

Follow-up review on 2026-09-12, after the author addressed Optional notes 1–3 and before completion of the full pilot. This section supplements the initial review; the earlier source hashes and findings remain as a historical record.

**Delta verdict: approve. No new Required or Critical findings. The actual smoke supports execution and numerical consistency of the original reviewed instrument on the declared case. Scientific review remains unavailable. The full seven-query pilot is not assessed in this addendum.**

### Reviewed changes

- `probe_messages` now stores finite attenuation effects for every candidate contrast at both rho values. `all_delta` has shape `[candidate contrast, group]`: the baseline vector is expanded along groups, while group-specific altered margins are transposed from `[group, candidate contrast]`. `all_predicted = rho * signed_groups` has the same axes and sign. The aggregate relative L1 diagnostic compares these arrays consistently. Existing altered logits are reused, so the change adds no MLP forwards.
- The JSON primary summary remains native top-1 versus top-2; the protocol addendum explicitly preserves this choice and limits the no-MLP comparison to that pair. This resolves the earlier ambiguity about validation scope. Individual candidate diagnostics can be reconstructed from the saved all-candidate finite effects and signed contributions.
- `surrogate_pool_top1_matches_native` now states the rank check's candidate-pool scope accurately.
- Settings record `args.device`, and both CUDA identity and peak-memory calls receive that device explicitly. This fixes incorrect device metadata for non-default CUDA devices. The driver remains a CUDA pilot, not a supported CPU command.
- The changed unit test checks the renamed field and presence/shape of the all-candidate finite array. The parent owns the delta test run; this reviewer did not rerun tests, load model weights, or use a GPU during this follow-up.
- Full checkpoint hashes and a permanent capture/mask integration regression remain optional improvements. The recorded model file size/mtime inventory is not being treated as a model-content digest.

### Actual smoke artifacts inspected

Directory: `reanchor/outputs/adoption_smoke_20260912`. Inspected `settings.json`, `summary.json`, `messages_00128.json`, `messages_00128.npz`, `influence.json`, and `manifest.json` using read-only CPU parsing.

- Recomputed all five artifact hashes in the manifest: all match.
- Recomputed both declared input-file hashes and the source-mask digest: all match.
- The smoke settings retain core hash `c7515df5...762bb` and driver hash `b56699d7...00873`, exactly matching the original review snapshots. Therefore the smoke appropriately contains the old rank-field name and does not contain the newly added all-candidate finite arrays. These are versioned artifact differences, not corruption or a failed delta run.
- Smoke recorded one query on NVIDIA GeForce RTX 4090, elapsed time `50.8871` seconds, and peak allocated GPU memory `17,738,558,976` bytes. This demonstrates feasibility for the declared one-query case on that device; it does not establish an arbitrary-input resource bound.
- Query 128 contains 53 groups and 145 candidates. Stored arrays have shapes `(53, 4096)`, `(144, 53, 32)`, `(145,)`, and `(145,)`; all values are finite. The sum over signed heads agrees with the JSON top-2 group contributions.
- Actual attention-message reconstruction error: maximum absolute `0.0104990`, relative L2 `0.00170286`. Native top-2 margin is `1.875`; fp32 surrogate margin is `1.864315`; maximum selected-logit mismatch is `0.0808830`. Candidate-pool top-1 is preserved. These discrepancies are recorded numerical approximation, so surrogate messages must continue to be described as surrogate quantities.
- Primary top-2 finite-attenuation relative L1 errors are `0.0440247` at rho = 0.1 and `0.134550` at rho = 1. This is useful agreement evidence at the measured point, with visible finite-step nonlinearity; it is not exactness or a general calibration result.
- All influence arrays contain 196 predictions and finite values. The sham maximum logit error is exactly zero. Both cuts have exactly zero maximum logit error before t = 134, zero prefix saved-token log-probability changes, and zero prefix argmax flips. Their prefix JS arrays exactly equal the sham arrays, and their first JS difference from sham occurs at t = 134, agreeing with the code's positional contract.
- **Numerical interpretation:** the exact-logit sham still has JS values up to `2.89782e-8` from float32 arithmetic. Its first strictly positive JS occurs at t = 2. Therefore `JS > 0` is not a valid onset rule; the saved exact-logit/prefix checks establish causality here. Post-134 mean JS is `0.0139997` for the selected history cut and `0.00456379` for the earlier cut, with three argmax flips each. These are single-case intervention measurements, not factual-error or influence-endpoint validation.

### Reviewed delta source identity

| File | SHA-256 |
| --- | --- |
| `graph/route_graph/adoption.py` | `9e36732cf88b0a23a5fd20005f7b7cbfe4510e0fee372f82821a7e4c0386180b` |
| `graph/tests/test_adoption.py` | `964b0c70a8d1632dd618c2d80216c23f2594cf7a89a0ae887b1e2dc5c6d8fbb3` |
| `graph/docs/EVIDENCE_ADOPTION_PLAN_20260912.md` | `b03b187458af375e4e8af29a604f31e81443b60dc225ca9394577788bda1d464` |
| `reanchor/src/decoding/adoption_probe.py` | `1315dcf2b79957d50070508c79f65aa63a74a73fae03df8734363eba7800d883` |
| `reanchor/scripts/run_adoption_probe.sh` | `4402b853228afbe2d628f64485580c0153586059f50fb1f5491a58243c92e9da` |
