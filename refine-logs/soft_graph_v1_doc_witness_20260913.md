**Outcome: authorized design-invalid early stop; population restoration verified. The 36-response graph run did not complete.**

Fresh witness agent `/root/soft_graph_v1_doc_witness` read `docs/SOFT_GRAPH_V1_RUN_20260913.md` and executed its fresh-witness command exactly once from `/share/home/tm902089733300000/a903202310/lys/research/graph`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u experiments/interleave_soft_graph.py --population-pid 153564 --output outputs/soft_graph_v1_20260913 --journal ../reanchor/runs/soft_graph_v1_20260913.json
```

- Persistent exec session: `67861`; wrapper PID: `155806`; graph child PID: `155856`.
- Wrapper preparation reported 36 responses and logical settings SHA256 `9833b851b479f6ffe53d4a601a947d53fc13eb4d161946daf5f2e9e33fa32cd8`.
- Input SHA256 documented for this run: `c2d5712bf5f08ddaf34427eb483e98f38433d9a09847a857efe44bbf0de0b264`.
- The final settings file SHA256 still matches the journal's frozen value `fb13d3dabe6333c15f00ae722993148018cecf8c949010f8b01172788753bec4`.
- The witness did not repeat CPU author preparation, restart the wrapper, start another GPU job, edit source/protocol/documentation, read annotation labels, or run evaluation. Every individual wait was at most 55 seconds.

The parent explicitly superseded the original no-stop instruction and requested graceful early termination before D. Its stated design diagnosis was that 1,704 frozen B edges yielded only 10 constructible edits, many grammatically poor, with segmentation and bag-word risk-scope defects that target-only D could not repair. These are the parent's findings and the authorization basis; this witness did not independently audit those design claims.

Before signaling, the witness recorded the live child command, cwd, parent PID, starttime, and phase files. The successful controller was `/usr/bin/python3` 3.10.12 with `os.pidfd_open` and `signal.pidfd_send_signal`. After opening the pidfd it rechecked:

- PID `155856`, Linux `/proc` starttime `970246307`, PPID `155806`.
- Cwd `/share/home/tm902089733300000/a903202310/lys/research/graph`.
- Command prefix `/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m route_graph.soft_graph_runner`, with the exact output directory and `--stage all`.

It sent exactly one `SIGINT` through that verified pidfd at Unix time `1789263534.69275`. At signal time progress was C, `27/36`, current response `9271`; the wrapper received no signal. An earlier controller attempt using the experiment Python failed with `AttributeError: os.pidfd_open` before opening any pidfd or sending any signal. The parent explicitly authorized the system-Python process-control fallback. One later read-only `/proc/.../fd` inspection encountered a disappearing file descriptor; subsequent lock evidence was collected without modifying processes.

Actual retained stage files after child exit:

| Phase | Completed response artifacts | Required responses |
| --- | ---: | ---: |
| A | 36 | 36 |
| B | 36 | 36 |
| C | 27 | 36 |
| D | 0 | 36 |
| Merge | 0 | 36 |

The 36 B artifacts sum to **72 actual feature forwards and 0 native forwards**. **Actual D native forwards: 0.** D never began: there is no D output directory/artifact, and the child traceback terminates inside C's `semantic_phase → finite → FrozenReader.ask` at `chosen.softmax(-1).cpu().tolist()` with `KeyboardInterrupt`. A/B/C work must not be reported as native measurements. Partial C cache data may remain for the interrupted response; it does not constitute a completed C artifact. No immutable full predictions or full-run evaluation were produced.

The scheduler journal records graph child return code **`-2`**. Persistent session `67861` returned wrapper exit code **`1`**, with `RuntimeError: audit or verified recovery did not complete; inspect the journal`. This is an interrupted graph execution, not a completed run. `progress.json` remains the stale pre-interruption `running`, C `27/36` record and was intentionally not rewritten.

The wrapper's `finally` resumed the same frozen population job from original PID `153564` under new PID **`159091`**. Its journal status is `population_resume_verified`, but the journal's stored resumed progress only shows `loading_model` at completed `13669`; the witness did not accept that alone as proof of resumed work.

Independent live verification established:

- PID `159091` runs `/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m decoding.ragtruth_population --output outputs/ragtruth_population_20260912 --resume`, with cwd `/share/home/tm902089733300000/a903202310/lys/research/reanchor`.
- Before pause: completed **13,669**, failed **0**, total **17,790**. A first resumed observation showed completed **13,685**. At Unix time `1789263878.1486878`, progress was `running`, completed **13,711** (**+42**), failed **0**, remaining **4,079**, response `5976`, condition `source_permute`.
- At final verification Unix time `1789263878.3853648`, child PID `155856` and wrapper PID `155806` were both absent.
- The witness independently recomputed all **13,669** pre-pause manifest SHA256 values from the journal snapshot: **0 mismatches or missing manifests**.
- Frozen population `settings.json`, `inputs.jsonl`, `input_manifest.json`, and `length_preflight.json` all match their recorded SHA256 values.
- `/proc/159091/fdinfo/3` confirms PID `159091` holds the original population `.lock` (inode `1438534287`). `/proc/locks` independently showed `FLOCK ADVISORY WRITE 159091 00:92:1438534287 0 EOF`.
- Graph `.phase.lock` inode `1443960119` and scheduler `relation_interleave_20260913.lock` inode `1439558900` had no matching held locks in `/proc/locks`; their wrapper/child owners had exited.

Evidence files: `outputs/soft_graph_v1_20260913/progress.json`, phase A/B/C artifacts, `../reanchor/runs/soft_graph_v1_20260913.json`, `../reanchor/runs/soft_graph_v1_20260913.audit.log`, `../reanchor/runs/soft_graph_v1_20260913.population.log`, and `../reanchor/outputs/ragtruth_population_20260912/progress.json`. The large manifest snapshot was parsed and hashed, never dumped in full. No partial-label evaluation was performed. The retained artifacts support auditing the early-stop diagnosis; they do not support a graph-effectiveness or native-effect claim.
