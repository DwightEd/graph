# SourceRel-Mini training: independent documented invocation witness

Fresh witness read `docs/SOURCEREL_TRAIN_RUN_20260913.md` and executed the lexical command exactly once, without installs, edits, output overwrite, or retry. Actual exec session: **13958**, **exit 0**. CPU only. The supervision is original source-field pointer reconstruction, not natural-response ownership ground truth or hallucination labels.

Lexical completed on 883 sources / 7064 queries: source train 720 / 5760, source validation 163 / 1304, excluded queries 0. Validation top-1 0.8059815951, recall@5 0.9900306748, MRR 0.8682971808, NCE 1.2002207948; homograph subgroup 808 queries, top-1 0.8824257426; other-record-same-field subgroup 242 queries, top-1 1.0. These are reconstruction baselines only.

Independent post-run checks recomputed the complete source input file SHA-256, all 884 data artifacts, prepared feature artifacts, every output artifact, and all live/snapshotted code hashes. All matched. Exact hashes, all subgroup metrics, and document identity are preserved in `source_relation_train_doc_witness_lexical_evidence_20260913.json` (SHA-256 `4e362fe10579a35164b0ac689cf5ea7bdd87c3c7c63967b0eabe045be4946dea`).

The sanity training dependency has not completed yet. No training checkpoint is claimed at this stage. Later phases will be appended only after their actual completion.

## Two-source interface sanity completed

After feature witness session 57496 exited 0, independently verified its complete manifest and artifact hashes before starting exactly the documented 2-epoch sanity command. Actual head exec session **25965**, **exit 0**. Sources 13599 / source_train and 13610 / source_validation, each 8 queries. Finite loss and gradient assertions passed. Training query-mean NCE 1.7663611174 → 1.1366738081. Predeclared minimum-validation-NCE checkpoint selection chose **epoch 1** (validation NCE 2.1298903674, epoch 2 2.6308468580). This is an interface smoke test, not evidence of generalization.

Independent post-run verification confirmed all data, feature and training artifact/code hashes; actual checkpoint loads with weights_only=True, has 1049472 parameters, all finite, and its settings object digest / best epoch match actual settings and history. Checkpoint SHA-256 `6e1003954c2efa62caa0a40aac9e3d6c6b1e843ed530d5d2cfecf2556164b315`. Complete manifest SHA-256 `dd56a9b84aeee2c843de6bbb9bf2289f36ede94698e7d7398c51eb4c1a55b4c2`.

All exact input/code/manifest hashes and raw metrics: `source_relation_train_doc_witness_sanity_evidence_20260913.json`, SHA-256 `63cddc4f911aa50ddc2a13b6c5d543edfd518bb1456b292d4f6aeffa3a2bab3d`. Full training has not been started; it depends on the real full-feature complete manifest.

## Full 20-epoch documented training completed

Resumed only after full feature witness session 62985 exited 0 and the exact full-feature complete manifest and artifacts were independently verified. The run document was unchanged from the first two stages. Executed the documented full 20-epoch CPU command **once**, actual session **69570**, **exit 0**. No prior stage rerun, installs, edits, output overwrite, or retry. The 20 complete epoch records are retained.

Minimum validation NCE selected **epoch 20**, NCE **0.0968227681**. On 163 held-out sources / 1304 reconstruction queries, trained top-1 **0.9539877301** (1244 correct / 60 errors), recall@5 **1.0**, MRR **0.9766104294**. Same-pool TF-IDF top-1 **0.8059815951**, recall@5 **0.9900306748**; frozen causal hidden cosine top-1 **0.3021472393**, recall@5 **0.8320552147**. TF-IDF predictions are byte-identical to the independently documented lexical run.

The critical subgroup remains weaker: same-field/other-record validation queries **242**, trained top-1 **0.7727272727** (187 correct / 55 errors), versus TF-IDF **1.0**. Thus **55 of 60** overall head top-1 errors lie in that subgroup. Homograph validation queries **808**, trained top-1 **0.9665841584** (781 correct / 27 errors), versus TF-IDF **0.8824257426**. Overlapping subgroups must not be summed. The 100% recall@5 is on this field-reconstruction task only; natural ownership and a full hallucination detector remain unvalidated.

Independent audit verified original input SHA; all 884 data artifacts; full prepared/encoded artifacts and all 38,541 × 4,096 feature values finite; all live and snapshotted code hashes; complete output manifest; and a real finite 1049472-parameter checkpoint whose settings digest and selected epoch match settings/history. All 7064 saved predictions per method have unique, complete query coverage, valid same-source pools and source positive IDs, consistent top-5 positive ranks, and independently recomputed aggregate metrics.

Checkpoint SHA-256 `9c716c94ffffd7fc5d114f931f2973290f1dd798c227cef6d57dc2ddc1ed2ff6`; complete training manifest SHA-256 `db00af44f3470a42747a384a8299f873926a48b0a513287b9d455fd41b1b65b0`. Complete exact hashes, all subgroup counts/metrics, 20-epoch trajectory and checkpoint metadata: `source_relation_train_doc_witness_full_evidence_20260913.json`, SHA-256 `441f9b5e22b515655e73db187a1a704a2cb1f33dcb3cbc8a119c0440eb3fb6bc`. All three requested documented stages are now complete.
