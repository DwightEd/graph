# Attributed sample graph capture witness — 2026-09-12

Fresh agent `/root/sample_graph_witness` followed the documented invocation once, without editing code, installing packages, deleting outputs, retrying the capture, or performing Git actions. The command exited **0** and produced the root completion marker `reanchor/outputs/attributed_samples_20260912/manifest.json`.

Read before execution: `/root/.agents/skills/run-experiment/SKILL.md`, `/root/.agents/skills/shared-references/compute-env-contract.md`, `graph/.aris/compute/local.md`, and `graph/docs/ATTRIBUTED_SAMPLE_GRAPH_20260912.md`. No applicable `AGENTS.md` or project `CLAUDE.md` was found. `rg` was unavailable, so file discovery used `find`; this did not alter the experiment command.

The existing spec canonical SHA-256 prefix was `8d044d57`, matching the local environment ledger. The environment was reused. Preflight found GPU 0, NVIDIA GeForce RTX 4090, at **1 MiB / 24564 MiB** allocated according to `nvidia-smi`; the requested output directory did not exist, and the shared filesystem reported 5.7 PiB available. The GPU returned to 1 MiB after completion.

The following command was executed verbatim from `/share/home/tm902089733300000/a903202310/lys/research/reanchor`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=../graph:src /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m decoding.sample_graph_capture --output outputs/attributed_samples_20260912
```

The process was PID 12070, tool session 80629. Launch was approximately 13:51:42 UTC and completion was observed at 13:52:26 UTC, approximately **44 seconds** including Python startup/imports and completion-manifest hashing. This wall duration is approximate rather than an externally instrumented timing. The runner recorded **14.9282569885 seconds** from output-directory initialization through capture/archive creation, excluding initial imports and final manifest hashing. Its peak CUDA allocated memory was **17,764,282,880 bytes = 16.5443 GiB**. A sampled `nvidia-smi` reading during execution showed 17,784 MiB used; that is a sampled device reading, not a measured device-memory peak.

| Saved sample | Source | Seed | Prompt tokens | Target tokens | Input nodes | Raw array bytes | Maximum attention row-sum error |
|---|---|---|---:|---:|---:|---:|---:|
| 00012 | 14375 | 0 | 535 | 196 | 730 | 3,438,620,530 | 0.00302284956 |
| 00013 | 14375 | 1 | 535 | 147 | 681 | 3,071,125,577 | 0.00294685364 |

Both captures contain 32 layers, 32 attention heads, 4096-dimensional residual/MLP attributes, and 8 KV heads with head dimension 128. For each sample with N input nodes, the independently inspected arrays have shapes `residual=[33,N,4096]`, `post_attention=mlp_update=[32,N,4096]`, `values=[32,N,8,128]`, and `attention=[32,32,N,N]`. These arrays are float32; identifiers are int64 and roles int8. The next-token arrays have 196 or 147 rows, with eight stored top logits/IDs per row. Every `.npy` file opened successfully using memory mapping and `allow_pickle=False`. Total raw array storage is **6,509,746,107 bytes (6.0627 GiB)**; all output files together occupy 6,509,800,561 logical bytes.

The successful runner applied its per-layer finite-value, nonnegative-edge, strict-future-edge, attention row-sum, dimensional, role-value, and target-alignment checks before archiving each graph. The full attention shape includes prompt queries and keys. The built-in maximum raw MLP/residual-addition difference is 0.75 for both graphs. The validator records that difference but does not enforce a numerical threshold. As an additional independent archive check, I recomputed every layer's `post_attention + mlp_update` in CPU bf16 and compared the float32-converted result with the saved next-layer residual: **zero mismatched elements and maximum error 0.0 for both samples**. This validates the documented dtype-rounding qualification for this run.

Independent post-run verification established:

- **33 / 33 manifest entries** match their SHA-256 digests. The manifest covers every output file except itself, with no missing or extra files. Hash verification took 14.377 seconds; the complete archive-verification pass took 17.727 seconds.
- Manifest SHA-256: `bc92bef40a71c93fda30242996bba6ee74edcd15eb0557e0245e51fe91be560a`.
- Both input-file SHA-256 values match `settings.json`; all three archived executed-source files match its recorded code hashes.
- `input_ids` exactly match each saved natural sample's `token_ids[:-1]`; target IDs exactly match `token_ids[prompt_length:]`. The last target is excluded from the input-node set as documented.
- Both source-mask hashes match the saved source metadata. The run also checks saved-state and saved-sample token identity before using that metadata.
- `settings.json` records torch `2.8.0+cu126`, transformers `4.57.1`, the RTX 4090, bf16 compute, eager attention, and the local Llama-3.1-8B-Instruct model path. Model identity is represented by file names, sizes, and modification times, not independent weight-content hashes.

No operational discrepancy prevented executing the documented command, and no workaround was needed. The residual-addition threshold limitation above is a precision about the built-in validation scope; the independent dtype-aware comparison passed. The environment spec's `run_commands` still describes an earlier binding smoke; the sample-graph command is supplied by the task-specific document, which was followed successfully. This witness does not claim clean-environment installation reproducibility or rerun the separately documented tiny-model mechanism tests.

These two source graphs are **instrumentation artifacts only**. They replay saved natural responses in fresh complete forwards, with no newly sampled answers and no correctness or ownership labels supplied to the capture. The runner explicitly reports `constraint_ownership_identification: "not yet evaluated"`. Successful capture, checksums, and numerical consistency provide **no evidence of constraint-ownership discrimination, semantic relation correctness, model adoption, or generalization**.
