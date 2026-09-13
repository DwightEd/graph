# Attributed sample graph: independent engineering review

Date: 2026-09-12. Skill: `/root/.agents/skills/code-review-and-quality/SKILL.md`, already read during this session. This is a bounded engineering review, not scientific approval.

**Verdict: approve the declared two-sample capture instrument. No Required or Critical findings in the reviewed capture path. Actual full-size captures remain the separate GPU witness's responsibility. Constraint ownership, factual correctness, discriminability, and model adoption have not been established by these checks.**

## Scope and method

Reviewed new files only: `graph/route_graph/sample_graph.py`, `graph/tests/test_sample_graph.py`, `reanchor/src/decoding/sample_graph_capture.py`, and `graph/docs/ATTRIBUTED_SAMPLE_GRAPH_20260912.md`. Read the tests and protocol before implementation. No implementation edits, GPU use, installations, git commits, resets, or deletion of existing files. The only permanent reviewer output is this report.

The latest requested representation is one graph per prompt–response sample, retaining high-dimensional node attributes and complete causal relation topology before investigating constraint ownership in real correct/wrong decision windows. This implementation stores that representation; it does not implement or claim the later ownership evaluation.

## Required findings

None for the declared eager Llama/bf16 driver and two fixed trace files. The optional validator findings below mean successful `validate_graph` execution should not be described as a complete semantic certification of every archive field.

## Optional findings

1. **Optional — enforce the raw residual addition invariant if `validate_graph` becomes an integrity gate.** `graph/route_graph/sample_graph.py:169` computes the maximum error between the next residual and post-attention plus MLP output but never rejects a large discrepancy. An independent CPU check added 100 to the final residual; validation still returned normally with error `100.000099`. For the current direct hook capture, tests establish that the actual raw state is correct and the metric exposes errors for inspection. For general archival validation, pass/record the compute dtype and compare against addition rounded in that dtype, or use an explicit dtype-aware tolerance. This avoids treating a large bf16 rounding error and a normalized/wrong-stage residual as equivalent. The tiny bf16 check showed exact equality after bf16 rounding, while unrounded fp32 addition differed by `1.08719e-4`.

2. **Optional — include output comparator fields in archive validation.** `graph/route_graph/sample_graph.py:148` checks target-aligned entropy and log normalizer, but not `top_ids` or `top_logits`. A NaN inserted into `top_logits[0, 0]` is accepted; missing/misaligned top arrays are also not covered by that loop. Validate matching `[targets, k]` shapes, finite logits, integer nonnegative IDs, and reasonable top ordering when these fields are used for native-output comparisons. This is a validator coverage gap; the current capture produces these arrays directly from native finite logits.

3. **Optional — hash the metadata inputs that supply model selection and source/response annotations.** `reanchor/src/decoding/sample_graph_capture.py:22` reads sample settings and `:23` reads `samples.jsonl`, but `input_sha256` at `:45` includes only the two NPZ files. The output does preserve the resulting source ID, seed, response text, model path, and a source-mask digest; the input and state token identities are checked. Add hashes for settings and `samples.jsonl`, and preferably record the source-state input path alongside the mask digest, to make the annotations' provenance independently traceable. Model names/sizes/mtimes remain file inventory, not checkpoint-content identity; a cached full checkpoint hash manifest remains an optional later improvement.

## Correctness and completeness

- **Complete topology:** `sample_graph.py:50` allocates `[L, H, N, N]`; `:87` captures every layer's native eager attention output without slicing away prompt queries, thresholding, averaging heads, or reducing layers. Axes are `[layer, head, receiver, sender]`, so information travels sender → receiver. Future sender positions are forbidden by validation at `:159`.
- **High-dimensional attributes:** `:46`–`:49` allocate complete residual, post-attention, actual MLP-output, and V-projection tensors. These are not candidate-margin scores or learned low-dimensional embeddings. Top-8 logits are separate output checks and do not replace node attributes or prune edges.
- **State stages:** layer pre-hooks capture raw pre-layer residuals. The `post_attention_layernorm` pre-hook captures the residual after attention addition and before the MLP. The MLP forward hook captures its actual output, not a difference of hidden states. The final decoder-layer forward hook captures raw final-layer output before `model.model.norm`. The installed Transformers version's Llama layer returns the required tensor, and the test compares its normalized value with the HF final hidden state while confirming the raw state itself differs.
- **Same-forward alignment:** all hooks run in one native forward on `ids[:-1]`. The driver reuses only source-role metadata from the old states archive and verifies exact token identity; it does not merge old hidden states with new attention. Captured fields therefore share positions, layers, and computation.
- **GQA values:** actual `v_proj` output is reshaped into `[N, KV_heads, head_dim]`, preserving the distinct KV heads. Reconstructing individual head messages requires GQA repetition and the corresponding model O-projection blocks, which are intentionally referenced rather than copied per sample. A graph edge is an attention reading weight; its vector-valued effect also depends on V/O and downstream state.
- **Prediction alignment:** for P prompt tokens and T response targets, input nodes number `N = P + T - 1`. Native logits are sliced from row `P - 1`; `target_ids = ids[P:]`. Prediction t therefore reads node `P + t - 1`, including the first response prediction from the final prompt node. The final target is deliberately not an input node. The overlapping response-node/target identities are checked at `:123`.
- **Roles and labels:** source, other prompt, history, and special roles use input metadata only. Special tokens take precedence in the exclusive role code. Correctness labels and constraint ownership are absent from graph arrays. Selecting known real decision windows is disclosed as sample selection and does not become a graph feature.
- **Lossless captured values:** every bf16 attention, residual, MLP, and V value converts exactly to float32. The code does not introduce float16 underflow, threshold zeros, or discard nonzero heads. Zeros already produced by native attention remain zeros. This preserves captured finite-precision values; it is not exact real-arithmetic model computation.

## Architecture, safety, and resources

- The reusable capture/validation/storage module is separate from the fixed-sample loading driver. Existing capture and research implementations are not modified. No new dependency is introduced.
- The model is loaded from local files and placed in eval mode with frozen parameters; capture also runs under `torch.no_grad()`. No training or resampling occurs.
- Inputs use `allow_pickle=False`; source-state token identity must match the sample. Existing output directories are rejected. Large tensors are separate `.npy` files, allowing memory mapping and avoiding whole-graph loading during later readouts.
- The driver writes settings and copies executing sources before capture, writes summary only after both archives are saved, and hashes all resulting files before writing the root manifest. The protocol correctly requires that root manifest as the completion marker. Individual array writes are not atomic, so consumers must honor this marker after interruption.
- The inspected traces have 731 and 682 total tokens, each with P = 535. They therefore produce N = 730 and 681 input-node graphs, respectively. Both sample metadata rows identify source 14375, with seeds 0 and 1. Their full attribute arrays total approximately 3.44 GB and 3.07 GB, before small metadata files. The loop releases each graph before capturing the next. Validation operates layer by layer, though the capture itself allocates the complete per-sample graph in host memory.
- GPU attention outputs are also materialized for the native forward. Feasibility at real 8B scale must come from the separate witness, not the tiny CPU tests. The fixed two-case driver is bounded in scope; the reusable function has no arbitrary-length memory cap and should not be presented as a generally bounded large-dataset capture service.

## Honest interpretation limits

The protocol correctly keeps full visible context and connections across grouped spans. It distinguishes a flat token graph with layer/head edge types from an unfolded `(layer, position, stage)` computation DAG, avoiding cross-layer path claims that ignore execution order. It also distinguishes offline access to the complete saved sequence from causal online readouts.

Retaining X and E enables the proposed ownership investigation; it does not show that ownership is decodable, that graph topology adds value beyond X, or that the model adopted a constraint. Two related traces cannot establish a generalizable high-dimensional probe. The document explicitly reserves X-only/E-only/X+E comparisons, relation controls, and independent-source evaluation for subsequent work, and states that no GNN or classifier is trained. The literature/novelty statements were outside this engineering review and are not independently approved here.

## Verification performed

Existing research conda Python; OMP/OpenBLAS/MKL threads set to 4; CPU only.

1. `PYTHONPATH=graph:reanchor/src .../conda_envs/research/bin/python -m pytest -q graph/tests/test_sample_graph.py`: **2 passed in 20.33 seconds**. Tests cover an actual tiny eager Llama, raw versus normalized final state, source-internal edges, prefix invariance after future extension, target alignment, and rejection of future edges/incorrect overlapping targets.
2. Independent tiny actual Llama in bf16: 2 layers, hidden size 16, 4 query heads / 2 KV heads, input `[1,2,3,4]`, targets `[4,5]`, P = 3, special ID 1.
   - All five large tensor families exactly survive float32 → bf16 → float32 roundtrip.
   - Every next residual exactly equals post-attention plus actual MLP output after bf16 addition rounding.
   - Maximum unrounded addition error is `0.0001087188720703125`; maximum attention-row sum error is `0.001953125`, consistent with stored native bf16 values.
   - Roles are `[3,0,0,2]`, preserving special/source/history interpretation.
   - All arrays save and reload through mmap with exact element equality.
   - A repeated save to the same directory raises `FileExistsError`.
3. Validator perturbation checks establish the two optional validation gaps above. Synthetic temporary archives were removed by their own temporary-directory context; no project artifacts were deleted.

No real GPU witness results, broad repository checks, ownership scores, or scientific acceptance are claimed by this review.

## Reviewed source identity

| File | SHA-256 |
| --- | --- |
| `graph/route_graph/sample_graph.py` | `ba7408a1b9b51484997f0fc7324eb3fdb62a004368e2c74634976739b94aee0d` |
| `graph/tests/test_sample_graph.py` | `a9c5d4c7f763ec931d767198a8e06087de3e70dc68faee969a30e49dccd24aec` |
| `reanchor/src/decoding/sample_graph_capture.py` | `d9373e3020ac31e44449b709e49799689f7a2836451e57ef1b35d1520eb9c11c` |
| `graph/docs/ATTRIBUTED_SAMPLE_GRAPH_20260912.md` | `6fc5c19dd5c7b12a7d306dcc50f44ca2a35084e7a5beb1915752902f948818e1` |
