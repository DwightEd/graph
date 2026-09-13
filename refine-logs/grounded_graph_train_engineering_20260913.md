# Grounded graph adapter/train bounded engineering review — 2026-09-13

Scope: `next_iteration/grounded_graph_adapter.py` and `next_iteration/grounded_graph_train.py`, with just enough read-only context from the feature/reconstruction interfaces to understand packet shapes. I added CPU-only tests in `tests/test_grounded_graph_train_review.py`. I did not edit implementation files, launch GPU, read labels, or run feature/data preparation.

Verdict: **Critical 0, Required 1** before training launch.

## Required

1. **Checkpoint selection cannot choose the initial adapter, even when it has the best validation objective.**

   The adapter is intentionally initialized so `output.weight == 0`, making the initial hidden state exactly the frozen LM state while pointer projections are still defined. `run()` measures and writes `graph/initial.json` / `no_edges/initial.json`, but initializes `best = inf` and only updates `selected` inside the epoch loop. If every trained epoch worsens validation objective, the output still selects epoch 1 or later. This violates the protocol string `minimum source-validation token-mean CE + pointer-mean NLL; earliest tie`, and can freeze a worse adapter for the natural detection stage.

   Minimal fix: save an initial checkpoint per arm and initialize `best/selected` from `initial`, then replace it only on strictly lower epoch validation objective. If the intended policy is “best post-update epoch only,” change the protocol and docs, but that is a weaker launch policy because the training run can knowingly choose a worse-than-initial detector.

## Verified by targeted tests

- `GroundedGraphAdapter` starts with `hidden == query_states`, so initial token CE goes through the exact frozen LM hidden state. It also rejects graph edges that cross independent samples.
- Batch assembly offsets node IDs, query IDs, local edge IDs, target token IDs, and first-token pointer supervision across multiple examples correctly.
- `token_loss()` rejects trainable LM heads, preserving the frozen-head contract.
- `pointer_loss()` rejects a positive owner that is outside the eligible/available source set.
- A monkeypatched tiny CPU `run()` demonstrates the Required checkpoint-selection bug: initial validation objective `0.1`, epoch validation objective `0.3`, but selected checkpoint still records `0.3`.

## Test command

Run from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_grounded_graph_train_review.py -q
```

Result: **3 passed, 1 failed**. The failing test is `test_run_checkpoint_selection_can_keep_initial_if_all_epochs_worse`, which captures the Required blocker above.

## Non-blocking observations

- Graph and no-edge arms use the same initialization seed and epoch order; this is sufficient for the intended matched-order comparison.
- The no-edge arm keeps unused message/edge parameters in the module, but they receive no gradient when `use_edges=False`. That is acceptable for this ablation as long as reports describe it as “no observed-edge message passing,” not identical effective capacity.
- Training enforces source SHA train/validation separation across all feature entries, including unavailable entries, and refuses natural-response feature parents as training input.
- The module marks `labels_read=False` and `natural_detection_validated=False`; it does not claim semantic truth, native adoption, or original-generator causality.

---

## R1 closure after checkpoint-selection fix

Root fixed the initial-checkpoint selection blocker. Each arm now writes `{arm}/initial.pt` with epoch `0`, settings hash, and vocabularies; `best`/`selected` are initialized from the initial validation objective and replaced only by a strictly lower epoch validation objective. The existing monkeypatched regression now passes.

Command run from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_grounded_graph_train_review.py -q
```

Result: **4 passed in 13.53s**.

Final train/adapter status: **Critical 0, Required 0**.
