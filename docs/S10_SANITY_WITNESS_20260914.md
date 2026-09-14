# S10 independent sanity witness — 2026-09-14

Fresh agent followed the documented invocation once, verbatim, using the unchanged research environment. No code/doc changes, retries, full execution, dependency installation, GPU execution, or annotation reads.

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/reanchor/scripts/export_structural_features.sh sanity
```

Actual command exit: **0** (exec session 7270). Stdout reported exported 1/planned 12, then complete true/exported 12/model_forwards 0.

Independent read-only verification exit: **0**. Manifest complete=true, planned=completed=12, labels_read=false, model_forwards=0. Exactly 12 unique row artifacts match records; all 12 artifact SHA256 hashes match. The records SHA256, executed source SHA256, parent settings/input SHA256, and all 12 parent manifest SHA256 hashes match their declarations. Parent inputs were hashed as raw bytes, not parsed for labels; no annotation file was opened.

All five feature columns are finite. Every values array has shape (tokens, 5), dtype float32; offsets have shape (tokens, 2), dtype int64, are finite, and match both values length and record token count. Total tokens: 2445.

| Response ID | Tokens |
| --- | --- |
| 11859 | 47 |
| 11865 | 223 |
| 11871 | 195 |
| 11877 | 140 |
| 11883 | 173 |
| 11889 | 181 |
| 11895 | 176 |
| 11901 | 299 |
| 11907 | 206 |
| 11913 | 257 |
| 11919 | 416 |
| 11925 | 132 |

Output: `research/reanchor/outputs/s10_structural_sanity_20260914_v1`. Manifest SHA256: `afe968c1eae087dea20d02f25b7f37de974510e6643b421de25ba7d17070b4c1`.

Doc/reality divergences: none for the documented sanity expectations. Manifest `files` is empty; row artifact hashes are instead declared in the hash-protected records.jsonl and verified there. This is an execution/artifact-integrity witness only, not a scientific efficacy claim or a clean-environment rebuild witness.

Verified at 2026-09-14T03:59:07.412790+00:00.
