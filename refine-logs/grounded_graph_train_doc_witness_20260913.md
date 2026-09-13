# GroundedGraph train / natural prediction / evaluation document witness — 2026-09-13

Fresh agent followed `docs/GROUNDED_GRAPH_TRAIN_RUN_20260913.md`, `.aris/compute/local.md`, and the final train/predict/evaluate engineering closure reports. The three documented commands were executed verbatim once each, in order, with a background shell, `pipefail`, and `tee` to fresh logs. All three exited **0**. No installations, environment rebuild, runner/code edits, overwrites of experiment outputs, or experimental retries occurred. Environment canonical spec hash remained `03909e02` (research environment). The GPU was 1 MiB used before training, between GPU phases, and after completion.

## Execution receipts

| Phase | Session | Python PID | Shell PID | Exit | Runner-reported seconds |
|---|---:|---:|---:|---:|---:|
| Training | 84431 | 177478 | 177475 | 0 | 247.057 |
| Prediction | 17335 | 177766 | 177763 | 0 | 33.428 |
| Evaluation | 55390 | 177870 | 177867 | 0 | Not recorded by runner |

These timers exclude their startup/initial parent-verification stages; they are not full process wall times. All three PIDs were observed ended. Training peak CUDA allocation was 3,345,410,048 bytes (3.116 GiB). Prediction does not record peak CUDA allocation. No erasure command was launched.

Logs: `grounded_graph_train_doc_witness_20260913.train.log`, `grounded_graph_train_doc_witness_20260913.predict.log`, `grounded_graph_train_doc_witness_20260913.evaluate.log` in this directory. Exact executable: `/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python`. GPU phases used CUDA_VISIBLE_DEVICES=0 and the documented four-thread/offline environment. Evaluation used the documented CPU invocation.

## Frozen inputs and source boundary

The source-feature parent completed 240 available examples / 135,519 target tokens / 240 observer forwards. The natural-feature parent completed 36 available responses / 5,840 target tokens / 36 observer forwards. Parent-feature capture itself belongs to its earlier independent witness; this witness consumed its completed manifests.

Independently recomputed SHA256 of all 240 prompt source spans and checked the training settings against that exact roster. The split contains 192 distinct training-source texts and 48 distinct validation-source texts, with zero shared SHA. Each of QA, Summary and Data2txt contributes 64 training and 16 validation sources. Actual packet counts are 108,945 training tokens and 26,574 validation tokens, with **7,359 training / 1,861 validation supervised pointer positions**, total 9,220.

The 36 natural responses remain the existing development panel: six sources × six generators. Source **13717** is in the source-reconstruction training fold (six natural responses). The other five sources—11951, 13514, 14637, 15220, 15475—supply 30 source-unseen development responses. None overlaps the source-validation fold. Source-unseen does not make this previously used panel a new independent test. Overlap groups are preserved in the evaluation.

An early witness count incorrectly called the sum of `entries[].eligible_pointers` actual anchors (8,996 / 2,114). That field counts candidates. Reading the actual packets established the 7,359 / 1,861 counts above; the source-fold audit file was corrected, with the distinction and correction recorded. Frozen inputs were untouched.

## Training and checkpoint audit

Both graph and independently trained no-edge arms completed epochs 1–10, with epoch 0 also saved. All 22 checkpoints are preserved, finite, and bound to the training settings, arm, epoch, and node/edge vocabularies. Initial parameter tensors are bitwise identical between arms, and output.weight starts exactly zero. Each arm contains 1,641,089 adapter parameters. The frozen code uses the same seeded per-epoch sample order for both arms; there is no separate per-batch order trace.

| Arm | Selected epoch | Validation token CE | Validation pointer NLL | Joint objective | Coordinate pointer top-1 |
|---|---:|---:|---:|---:|---:|
| graph | 7 | 0.01886009 | 1.34798187 | 1.36684196 | 0.588931 |
| no_edges | 4 | 0.02155976 | 1.33163370 | 1.35319345 | 0.573885 |

Independent selection recomputation across epochs **0 through 10** matches the earliest minimum source-validation token CE + pointer NLL in each arm. Checkpoints: `graph/epoch_07.pt` SHA256 `c05e2112e43c92bea2a17e2bdde4785e5598dfd0defd22337ee9ce91fc5a1476`; `no_edges/epoch_04.pt` SHA256 `d0ce07fd00b05edbe303308b6745ecfb32347a13ffde3c6ceab5271a815105df`. Natural metrics were not used in selection. Coordinate pointer top-1 is source-copy supervision accuracy, not semantic ownership accuracy.

All epoch losses and all checkpoint tensors are finite. All 480 optimizer steps passed the frozen runner’s nonfinite objective/gradient checks; gradient clipping was 1.0. Raw gradient norms were not logged, so there is no measured gradient distribution. Training recorded 612 adapter forward batches in total and zero observer forwards.

The frozen LM head is Meta-Llama-3.1-8B-Instruct `lm_head.weight`, BF16 shape `[128256,4096]`, loaded from `model-00004-of-00004.safetensors`. I independently read the safetensors header and hashed its entire shard: `92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b`, matching the feature-bound model identity. The head is frozen by the executed training code and is absent from the adapter state dictionaries. No observer was rerun for training or prediction.

## Prediction and evaluation audit

All 36 response JSONs and 36 NPZs are present. Independently checked the exact full-word roster, token IDs/offsets and node order against the prepared packets; every token score and graph/no-edge pointer/gate/residual trace is finite. Each arm preserves 5,840 pointer rows with all source-node columns. Pointer row sums deviate from one by at most 5.96e-7. Recomputing all character-overlap-weighted word scores gave zero error. The fixed difference equals adapter NLL minus base NLL; larger remains risk. All 4,733 words are available; no unavailable words were dropped. Prediction ran 108 adapter forwards (graph/no_edges/destination-permuted graph), zero observer forwards.

Graph gate range is [0.02813, 0.98298], residual-norm range [0.37345, 60.35266]; no-edge gate range [0.06274, 0.93862], residual-norm range [0.21092, 43.77599]. These are adapter allocations, not observed native model routes. The permutation is destructive edge-destination rewiring, not a semantic counterfactual.

Before the evaluator launched, prediction exit/completion and all 73 artifact hashes were verified, and the evaluator matched its pretraining frozen SHA256 `8fd0ee2828de9420a7169f9347145c4656a6d22636b980fb0dc2bab2160d525d`. Its executed code freezes settings and its snapshot before the first annotation-byte access. File publication order is consistent with this boundary. Only final evaluation, followed by the independent final-evaluation audit, opened annotation bytes. The five canonical population parent hashes, annotation hash, full word spans and exact label text were verified. No score direction, checkpoint or protocol was changed after labels.

Independent weighted AUROC and threshold-grouped average-precision calculations used NumPy formulas, without importing sklearn or the experiment evaluator. All **441 metric cells** matched, including overall, 18 task/generator/official-split groups and both source-overlap groups; maximum absolute numeric error was **1.6209256159527285e-14**.

| Subset | Words | Error words | Unavailable |
|---|---:|---:|---:|
| all_words | 4733 | 260 | 0 |
| through_first_error | 3991 | 14 | 0 |
| post_first_error | 742 | 246 | 0 |

Through-first includes the first error word; post-first starts strictly after it. Responses without errors stay wholly in through-first. There are 14 error-containing responses. Metrics below are source-balanced; each subset reweights its contributing sources equally. No confidence intervals were run.

| Fixed score | All-word AUROC / AUPRC | Through-first AUROC / AUPRC | Strict-post-first AUROC / AUPRC |
|---|---:|---:|---:|
| base_nll | 0.535477 / 0.060010 | 0.552105 / 0.005228 | 0.516545 / 0.428023 |
| base_entropy | 0.523665 / 0.056583 | 0.621364 / 0.008077 | 0.486351 / 0.400464 |
| graph_nll | 0.533458 / 0.059613 | 0.537498 / 0.005343 | 0.516961 / 0.426095 |
| graph_difference | 0.505927 / 0.052714 | 0.390467 / 0.006276 | 0.512073 / 0.411346 |
| no_edges_nll | 0.532638 / 0.059847 | 0.529146 / 0.005025 | 0.514793 / 0.427345 |
| no_edges_difference | 0.491690 / 0.053128 | 0.289787 / 0.003473 | 0.502498 / 0.415469 |
| permuted_graph_difference | 0.510390 / 0.054336 | 0.406123 / 0.006264 | 0.529989 / 0.427531 |

| Source overlap | Responses | All words / errors | All-word base NLL / graph difference AUROC | Post-first base NLL / graph difference AUROC |
|---|---:|---:|---:|---:|
| unseen | 30 | 3816 / 232 | 0.537760 / 0.505174 | 0.525753 / 0.504918 |
| train | 6 | 917 / 28 | 0.528523 / 0.502210 | 0.513889 / 0.532118 |

The graph-difference score does not exceed base NLL AUROC on all words or strict-post-first in this small development panel. Destination-permuted graph difference scores higher AUROC than intact graph difference on both subsets. These observations support only the recorded detection rankings; they do not establish correct semantic owner, native adoption, accurate lookback, routing/aggregation, or continuous causal scope. Owner-payload source erasure remains unexecuted by this witness and must be run by the separate assigned witness.

## Integrity records and audit-output exception

Final rehash verified all 57 prelaunch frozen code files unchanged; training binds 55 of them, prediction 56, and the evaluator is separately frozen. New train/predict dependencies beyond feature parents are `next_iteration/grounded_graph_adapter.py`, `next_iteration/grounded_graph_train.py`, and `next_iteration/grounded_graph_predict.py`; `next_iteration/grounded_graph_evaluate.py` was also frozen before training. All prior feature-parent dependencies remain bound.

| Output | Manifest SHA256 | Verified artifacts |
|---|---|---:|
| grounded_graph_train_v1_20260913 | `899c8b7da2049e01ae487bf2c72b03cd581064021db7149bffb3fecb5f286c6a` | 45 |
| grounded_graph_predictions_v1_20260913 | `9f4d53423de3f30352199b03a52845cc6944a55056d446948885766c4b440be3` | 73 |
| grounded_graph_evaluation_v1_20260913 | `7cdb074730827bff9acdf4dfaa1d91da92d1bbe20e3942b49b26192661f87542` | 2 |

No documented invocation diverged from its executable contract. A separate independent audit subprocess (session **96604**) did exit **1**, after all its assertions and 441 numeric comparisons completed, when `json.dump` encountered a NumPy int64 in the documentation field `error_responses`: **`TypeError: Object of type int64 is not JSON serializable`**. This subprocess must not be described as exit0. Its exact traceback is retained in `grounded_graph_train_doc_witness_20260913.audit_error.log`, and the partial `grounded_graph_train_doc_witness_20260913.evaluation_audit.json` is preserved. A separate `grounded_graph_train_doc_witness_20260913.evaluation_audit_completed.json` copies its already-published check counts and completes the report fields from the frozen evaluation output; it did not rerun labels, metrics or any experimental command.

Additional machine-readable receipts in this directory: `grounded_graph_train_doc_witness_20260913.frozen.json`, `.source_fold.json`, `.training_audit.json` (audit session86050 exit0), `.prediction_audit.json` (audit session67536 exit0), `.evaluation_audit_completed.json`, and `.final_integrity.json`. The existing negative results and all old output directories were preserved. This is an execution and artifact witness, not scientific approval or clean-install reproducibility.
