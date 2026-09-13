# Grounded graph reconstruction and feature document witness — 2026-09-13

**PASS for documented execution and artifact integrity.** A fresh agent read
`docs/GROUNDED_GRAPH_RECONSTRUCTION_RUN_20260913.md`, the local environment
ledger, the run-experiment skill/environment contract, and the latest
`refine-logs/grounded_graph_adapter_review_20260913.md` closure. All five
documented data/prepare/encode invocations ran sequentially, exactly once, and
exited 0. The independent CPU audit also exited 0. Completed at approximately
2026-09-13 15:07 Asia/Shanghai.

This establishes execution and provenance, not semantic ownership or detector
effectiveness. The complete reconstruction text is raw source copy, and its
9,220 pointers are observed source coordinates. No Qwen partial-generation
output or RAG gold annotation was read or used. The natural responses are
previously used development data; one natural source also occurs in the
reconstruction training split, detailed below.

## Environment and execution

Reused `/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python`.
The independent canonical environment-spec hash is `03909e02`, matching the
ledger. No installs, rebuild, implementation edits, artifact overwrite, retry,
or second simultaneous model load occurred. The documented environment
variables and command arguments were unchanged. Each command ran in a
nonblocking process session with `set -o pipefail` and appended `2>&1 | tee`
to a fresh witness log. No `CLAUDE.md` exists in the graph root; the supplied
local ledger and run document provide the environment contract.

The GPU preflight before preparation and the documented check before encoding
both returned GPU 0 at **1 MiB used / 24,564 MiB total**. The runner's existing
exclusive lock remained in use. The two GPU commands executed sequentially.
After both exited, GPU 0 was again observed at 1 MiB used.

| Document step | Session | Python PID | Exit | Log under `refine-logs/` |
| --- | ---: | ---: | ---: | --- |
| Source reconstruction | 25858 | 176384 | 0 | `grounded_graph_reconstruction_prepare_doc_20260913.log` |
| Reconstruction feature prepare | 37770 | 176473 | 0 | `grounded_graph_feature_prepare_doc_20260913.log` |
| Natural feature prepare | 52086 | 176762 | 0 | `grounded_graph_natural_feature_prepare_doc_20260913.log` |
| Reconstruction feature encode, GPU 0 | 14845 | 176891 | 0 | `grounded_graph_feature_encode_doc_20260913.log` |
| Natural feature encode, GPU 0 | 93365 | 177125 | 0 | `grounded_graph_natural_feature_encode_doc_20260913.log` |
| Independent CPU artifact audit, CUDA hidden | 62329 | 177225 | 0 | `grounded_graph_feature_independent_audit_20260913.log` |

During reconstruction capture, process-visible PID 176891 reported progress
32 → 176 → 240, with actual forward counts matching. `nvidia-smi` exposed
driver-visible PID 57728 while the session process PID was 176891; GPU memory
was observed at 16,122 MiB. Natural progress reached 8 → 36 with 36 forwards.
No additional experiment job was launched by this witness.

## Complete denominators and capture results

The CPU data command produced 240 source records, **192 train / 48 validation**,
with 64 train and 16 validation sources for each of QA, Summary, and Data2txt.
All source records have at least one pointer. There are **9,220 pointers**, zero
model forwards during reconstruction/preparation, and no unavailable examples.
The independent audit rederived the selected ordered roster and all six
task/split selection counts from the frozen annotation-free population input.
The 192 train and 48 validation raw source SHA sets are unique and disjoint.

| Quantity | Reconstruction capture | Natural development capture |
| --- | ---: | ---: |
| Examples / responses | 240 | 36 |
| Distinct sources | 240 | 6 |
| Available / unavailable examples | 240 / 0 | 36 / 0 |
| Target token denominator | 135,519 | 5,840 |
| All non-whitespace words | 90,670 | 4,733 |
| Maximum input tokens, no truncation | 4,357 | 1,756 |
| Actual observer forwards | 240 | 36 |
| Source node rows | 117,314 | 20,844 |
| Directed graph edges | 439,996 | 78,960 |
| Unavailable source nodes | 0 | 0 |
| Source rows per array, minimum–maximum | 157–2,078 | 209–1,495 |
| Query rows per array, minimum–maximum | 160–2,172 | 81–374 |
| Raw float32 array bytes | 4,142,415,872 | 437,190,656 |
| Runner seconds, including final verification | 126.1279 | 29.8405 |
| CUDA allocated peak bytes | 16,610,850,816 | 16,282,228,736 |
| CUDA allocated peak GiB | 15.47 | 15.16 |
| Prepare-manifest artifacts | 241 | 37 |
| Complete capture-manifest artifacts | 481 | 73 |

Runner seconds exclude process startup, initial model hashing, and model
loading; they are not complete command wall-clock measurements. CUDA peaks
are the runner's allocated-memory counters, not total driver memory.

## Independent integrity checks

Audit implementation:
`refine-logs/grounded_graph_feature_independent_audit_20260913.py`.
Machine-readable result:
`refine-logs/grounded_graph_feature_independent_audit_20260913.json`.
The script independently implements the checks rather than calling the
production feature loader or verifier. It loads only the local tokenizer;
CUDA is hidden and the audit performs zero model forwards.

- Verified reconstruction settings, complete manifest, source files, immutable
  live/snapshotted code, full input SHA, selected source order and task/split
  census, original prompt identity, raw source text SHA, deterministic split,
  pointer raw spans, owner containment, prefix warmup, uniqueness, and the
  complete 240-source / 9,220-pointer denominators.
- Verified both prepare manifests and exact complete artifact rosters:
  276 packet files; 276 NPZ files containing 552 arrays; 276 capture receipts;
  and both summaries. No available example was omitted, and there are no
  extra feature files or silently reduced denominators.
- Verified all 16 observer/model-directory files against their frozen sizes,
  modification timestamps, and SHA-256 values. Verified 50 reconstruction
  code snapshots and 53 code snapshots in each feature output against live
  files. All inventory, source, parent, settings, and artifact hashes matched.
- Retokenized every original prompt and response with the frozen tokenizer.
  Verified exact input IDs, BOS, response offsets, target IDs and positions,
  and `query_position = target_position - 1`. For all 36 natural responses,
  original frozen token IDs, offsets, response text, and row digests matched.
- Independently rebuilt the complete inventory node order, nine node-kind
  indices, source-token memberships, bundle token unions, and directed edge
  order. Every pooling index belongs to the source portion of the prompt;
  no response token enters a source pool. Rebuilt weak target token indices,
  first-token positions, owner row indices, and pointer supervision.
- Verified every receipt self-digest, packet/input/model/code binding,
  final-norm representation, one actual forward and one final-norm hook
  execution, BF16 observer dtype, and SDPA attention. All 552 arrays are finite
  float32, width 4,096, with exact row counts, shapes, and raw-byte SHA-256
  checks. Array source rows follow the full packet node order and query rows
  follow the full target token order.

## Natural development source overlap

All 36 natural rows carry `official_split=train` in the frozen input (12 per
task), and they were already designated reused development responses. One
of their six source texts is also among reconstruction **training** sources:

- Data2txt source ID: `13717`.
- Raw source SHA-256:
  `ed9d31beb2b9b17de1ab2686a850c6785bdf15feb851f5faa04f0a327be41fd4`.
- Six natural response IDs: `6303`, `6301`, `6300`, `6304`, `6305`, `6302`.

This does not violate the internal reconstruction train/validation SHA
separation, which passed. It means the 36-response natural aggregate is not
entirely source-heldout from reconstruction training. The frozen denominator
remains 36 responses / 4,733 words. Natural response text or gold labels were
not used as reconstruction training supervision. No inference about unseen
source generalization should be based on the full natural aggregate alone.

## Final manifest identities

| Artifact | SHA-256 |
| --- | --- |
| Reconstruction `manifest.json` | `dfc569bd01fd8752f6c7e1cae106b31527e60c4e7489862eb57d88ceea9af916` |
| Reconstruction feature `prepare_manifest.json` | `b72de791b5fcffb513bf0e046f3a9c4e347e1cb6fb5dc3be862202ae476f2e71` |
| Reconstruction feature `manifest.json` | `61bd0e2e217f88a6fb508d0b77cf1f4342f8d511ef16dd3ae67e41fc1d5fd967` |
| Natural feature `prepare_manifest.json` | `cd4474db71108a643439e12190401e34794a4a9633d449cfb79ca6d10e74fe81` |
| Natural feature `manifest.json` | `16d22d079b01c23513587d431374c29e4d754fec39f008cbec16e26b68467cfa` |

The parent was informed before preparation that all `route_graph/*.py`,
`next_iteration/{__init__,surface_graph,graph_boundaries,grounded_graph_data,grounded_graph_synthesize,grounded_graph_features,grounded_graph_feature_runner,grounded_graph_reconstruction}.py`,
the Llama model/tokenizer files, inventory, annotation-free population input,
and natural input must remain frozen. Their final identities passed. All five
documented commands and the audit completed without a doc/CLI/runtime mismatch;
the natural source overlap is a measured interpretation limitation rather
than a repaired or dropped datum.
