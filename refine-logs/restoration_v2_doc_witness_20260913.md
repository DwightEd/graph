# Restoration v2 fresh document execution witness — 2026-09-13

Status: commands 1–4 completed with exit 0; command 5 running; complete source, natural, training and prediction CPU audits passed. This is an engineering execution/provenance witness, not an external scientific integrity audit or evidence of method effectiveness.

## Contract and launch

- Fresh agent read `docs/GROUNDED_GRAPH_RESTORATION_V2_RUN_20260913.md`, `.aris/compute/local.md`, the run-experiment skill and compute environment contract before launching. The first run document hash was `18068ad7d95fb83bcaa8401bc89da8dfa9b1824a3e88030158c1ed591a02f6b2`.
- Existing environment canonical spec hash independently recomputed as `03909e02`; Python/package/model environment reused, with no installs, downloads or rebuild. No experiment code was edited by this witness.
- GPU preflight before command 1: index 0, 1 MiB used / 24564 MiB total. All six new experiment output directories were absent. No applicable `AGENTS.md` or `CLAUDE.md` was present in the workspace/ancestor paths checked. `rg` was unavailable; reads used standard tools.
- Root confirmed feature engineering Critical 0 / Required 0, then separately authorized commands 3 and 4 after pipeline closure. Dependence engineering subsequently closed Critical 0 / Required 0 (3 CPU tests); root authorized the complete six-command sequence and froze all v2 code before commands 3–6.
- Each documented two-line bash block is extracted from the document and executed verbatim once under `bash -c`. A nonblocking exec wrapper writes stdout/stderr to a fresh exclusive-create log and records launch/exit JSON. CLI owns the shared `reanchor/runs/relation_interleave_20260913.lock`; no automatic retry or output replacement is permitted.

| Command | Session | Actual command PID | Started UTC | Exit | Durable records |
|---|---:|---:|---|---|---|
| 1 source features | 51049 | 179414 | 2026-09-13T07:46:13.598805+00:00 | 0; 255.692 s wall | `restoration_v2_doc_witness_20260913_cmd1.{log,launch.json,exit.json}` |
| 2 natural features | 71004 | 179827 | 2026-09-13T07:50:41.651517+00:00 | 0; 71.788 s wall | `restoration_v2_doc_witness_20260913_cmd2.{log,launch.json,exit.json}` |
| 3 adapter training | 5417 | 179904 | 2026-09-13T07:52:15.664408+00:00 | 0; 386.308 s wall | `restoration_v2_doc_witness_20260913_cmd3.{log,launch.json,exit.json}` |
| 4 natural prediction | 49798 | 180312 | 2026-09-13T07:59:11.522734+00:00 | 0; 184.532 s wall | `restoration_v2_doc_witness_20260913_cmd4.{log,launch.json,exit.json}` |
| 5 source dependence | 45016 | 180492 | 2026-09-13T08:03:22.693086+00:00 | running | `restoration_v2_doc_witness_20260913_cmd5.{log,launch.json,exit.json}` (exit record pending) |

Actual `ps` observation confirmed PID 179414 was the documented restoration feature Python module, parent wrapper PID 179413. Launch, intermediate progress, manifest verification and final exits will be appended as observed.

## Completed feature launches

- Command 1: 240 available examples, 135519 tokens, 240 new observer forwards, runner seconds 167.423465, CUDA allocated peak 16610850816 bytes. Independent complete-manifest check verified all 722 artifact hashes and the settings hash; manifest SHA256 `4f0a0c2d0276efeca842e10656cec82f8502447772b19d5acffc32acded346c7`. Shared lock ownership was observed with `lslocks`. PID disappeared after exit, and GPU returned to 1 MiB before command 2.
- Command 2: 36 available examples, 5840 tokens, 36 new observer forwards, runner seconds 31.459463, CUDA allocated peak 16282228736 bytes. Independent complete-manifest check verified all 110 artifact hashes and the settings hash; manifest SHA256 `4cd8b49cc217b2f081b19ce87437900fa49e1a0036a4c3ae7139da045eab691a`. PID disappeared after exit, and GPU returned to 1 MiB before command 3.
- The 276 new forward count is for the actual source-erased observer captures. It excludes cached v1 forwards and adapter-only calls. Runner timers exclude some setup/final verification; process wall timers above are separate measurements.
- Original source/natural/erasure manifests before downstream work were complete, with hashes `61bd0e2e217f88a6fb508d0b77cf1f4342f8d511ef16dd3ae67e41fc1d5fd967`, `16d22d079b01c23513587d431374c29e4d754fec39f008cbec16e26b68467cfa`, and `984a870df71cfe22a15924b4f20118e32ad4cd9a7833b6f1ad3682f944ea2abf` respectively.

The complete source CPU receipt/array audit ran in session 96807, PID 179872, exit 0, measured audit time 117.597978 s, with CUDA hidden and no model forward. Its fresh audit log/result are `restoration_v2_doc_witness_20260913_source_audit.{log,json}`. The same audit of natural features ran in session 77515, PID 180034, exit 0, measured audit time 21.668994 s; corresponding records use `_natural_audit.{log,json}`.

## Independent full source array/receipt audit

All 240 rows passed recursive complete-parent verification, exact packet/node reconstruction from source inventory/tokenizer, all 16 model-file hashes, all 54 live/snapshot code files, and all 240 source-inventory artifact hashes. This checked every original and new array, not a sample.

- Census: 192 training / 48 validation, 135519 response tokens, 117314 source nodes, 439996 directed edges, 9220 pointer anchors.
- All original source X and H_full arrays match their original receipt byte hashes; every H_empty is finite float32, exactly `[tokens, 4096]`, and bound to its actual input/array receipt. Every original and new observer receipt records `torch.bfloat16` / `sdpa` and one actual final-norm hook execution / observer call.
- Independently recomputed all positive-overlap source-token sets and boundary crossings from the tokenizer. There are 135519 erased prompt-token positions and 237 crossing tokens in 237 examples. Every crossing token is replaced whole with its full prompt-character span correctly receipted.
- All executed inputs preserve length, BOS, every non-erased prompt position and the entire response-history suffix. Query positions are exactly target positions minus one; source-pooling node memberships stay strictly inside the prompt source positions.
- H_empty differs from H_full in 554045295 of 555085824 scalar elements. This is an observed intervention difference, not an efficacy statistic. No labels were read.

Command 3's actual PID 179904 was observed holding the shared GPU lock while performing parent/feature validation and loading. At the preceding progress notification no epoch log existed; this is recorded as preflight/loading, not optimizer progress.

The natural audit passed the same full checks on 36 responses / 5840 tokens, 20844 nodes, 78960 edges, six distinct inventory sources, 16 model files and 54 live/snapshot code files. All 36 examples have one correctly receipted whole-token boundary crossing; 23724 source prompt-token positions were erased. H_empty differs from H_full in 23830591 of 23920640 scalar elements. Natural packets contain zero training pointer anchors and all 36 rows have official `train` split; labels remained unread.

Subsequent actual logs confirmed graph epochs 1–3 and optimizer updates. Each epoch has 108945 training tokens / 7359 anchors and 26574 validation tokens / 1861 anchors. `nvidia-smi` reports GPU PID 186916; `/proc/179904/status` records `Ngid: 186916`, and PID 179904 holds NVIDIA device file descriptors, binding the GPU display identity to the documented Python process. This avoids incorrectly reporting the GPU namespace number as the workspace PID.

## Completed training and independent checkpoint audit

Both arms completed all 10 epochs. The complete training manifest and 45 artifact hashes passed independent verification: SHA256 `060204b81200704abe3d8384aebcafe57d0f7d5bc2f87a3ee5562f6c35040e32`. PID 179904 disappeared and GPU returned to 1 MiB before command 4 launched.

The training CPU audit ran in session 8543 / PID 180322, exit 0; result `restoration_v2_doc_witness_20260913_training_audit.json`, measured audit time 33.664804 s. It reverified the full trained parent and all 57 live/snapshot files, checked every epoch/checkpoint's finite weights, metadata, counts and objective arithmetic, and independently selected the earliest minimum validation objective over epochs 0–10.

- All four v1/v2 graph/no-edge initial state dictionaries are tensor-identical. Each arm has 1641089 parameters; numerical hyperparameters and the complete 192/48 source SHA split match v1 exactly. The source splits have no shared source-text hashes.
- Census: 480 optimizer updates and 612 adapter forwards, including initial validation and all train/validation passes; zero new observer forwards and no label access.
- Graph selects epoch 7: validation token CE 2.1420560357326073, pointer NLL 1.3017798382738852, objective 3.4438358740064925, coordinate pointer top1 0.6029016657710908. Checkpoint SHA256 `3e06e186887e1ec3a98bd0c86743808162c3457e6d0fa603d685346650e65e00`.
- No-edge selects epoch 7: validation token CE 2.1432387979733827, pointer NLL 1.211574551192488, objective 3.3548133491658705, coordinate pointer top1 0.629768941429339. Checkpoint SHA256 `9e9e5bb793f77fd04f03cf6d135a499d565f673a38833187f25ed0ea99f5840f`.

These are source-copy/coordinate training statistics. The separate source-dependence gate and natural development metrics remain pending at this update.

## Completed frozen natural predictions

Command 4 completed all 36 responses / 4733 words, zero unavailable words, 108 adapter forwards and zero observer forwards. All 73 manifest artifacts and the settings hash passed independent verification; manifest SHA256 `0f8a9b506e9a960c47ddf5d8bdf013091625bb7b5ba886775e7605679c57f69f`. PID 180312 disappeared and GPU returned to 1 MiB before command 5. Runner time was 47.744148 s, separate from process wall time.

Complete prediction/word/baseline CPU audit session 28142 / PID 180502 exited 0, measured audit time 2.098758 s; result `restoration_v2_doc_witness_20260913_predictions_audit.json`. It verified all 73 v2 and 73 v1 prediction artifacts, all 58 live/snapshot code files, all finite score/trace arrays, normalized pointer arrays, exact 36 source-node orders, and every word aggregation from token/character overlaps.

- All 11680 baseline token values (NLL and entropy over 5840 tokens) and 9466 baseline word values (two baselines over 4733 words) are exactly equal to frozen v1.
- Source overlap independently recomputed from source-text SHA256: 6 responses from adapter training sources, 30 source-unseen responses; zero validation-source overlap. All words are retained and available.
- For graph and no-edge arrays, `difference = logp_full - logp_restore = restore_nll - full_nll` and `restoration_difference = logp_empty - logp_restore = restore_nll - empty_nll` hold exactly for their saved float32 arithmetic. Across graph, no-edge and permuted branches the difference-family relation has maximum arithmetic residual 9.5367431640625e-7, consistent with separate float32 subtraction. No score family was exchanged, negated or selected using labels.
