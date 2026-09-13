# SourceRel-Mini M2 independent execution witness

Written 2026-09-13T05:09:15.070338+00:00. This fresh agent read `docs/SOURCEREL_TRANSFER_RUN_20260913.md` and executed its CPU prepare, GPU encode, and CPU predict invocations exactly once each in the existing research environment. No installation, code edit, output overwrite, or retry occurred. The parent confirmed the earlier full source encoder had actually exited before this GPU invocation; GPU encode additionally obtained its exclusive lock and free-memory gate. Full head training had actually exited before prediction, and its complete manifest was independently checked before prediction.

| Stage | Actual session | Exit | Outcome |
|---|---:|---:|---|
| CPU prepare | 97312 | 0 | 36 responses; 1619 complete original slots |
| GPU encode | 53795 | 0 | PID 171172; 687 documents; 172 actual forwards |
| CPU predict | 44672 | 0 | 36 response artifacts; 1619 preserved slots |
| Independent model/tokenizer hash audit | 60157 | 0 | All 16 model/tokenizer files passed fresh SHA-256 checks |
| Independent final artifact/slot audit | 7865 | 0 | Complete artifacts, finite features, source membership, and denominator checks passed |

The inventory has 687 unique views / 92,649 tokens, with maximum length 331. No text was truncated; unavailable-length views and unavailable whole-query pools are both 0. Feature array shape is 687 × 4096 float32, all finite. These are auxiliary canonical-view causal final-layer last-token states, not original generation trace states. Peak allocated CUDA memory was 16,226,466,816 bytes. Encoding loop reached 172 forwards in 11.175 seconds; runner summary time excluding model load was 26.569 seconds and includes final validation work.

| Task | Responses | Slots | Candidate supply | Source domain unavailable |
|---|---:|---:|---:|---:|
| QA | 12 | 470 | 0 | 470 |
| Summary | 12 | 527 | 0 | 527 |
| Data2txt | 12 | 622 | 622 | 0 |
| Total | 36 | 1619 | 622 | 997 |

There were 0 unresolved structured-source graphs and 0 empty broad-type pools in this roster. Data2txt query proposals comprise 598 text, 16 number, and 8 time-range slots. Every original slot ID appears exactly once in prediction output. Every candidate beam is a subset of its declared same-source broad-type pool and retains up to 5 candidates. Native certificate count and native forwards are 0; factual status remains unverified. Domain-unavailable slots remain in the full denominator.

The learned top-1 differs from the same-view frozen-cosine top-1 for 599 / 622 supplied slots. This measures ranking change only. No natural-owner ground truth was read or constructed, so no natural-owner accuracy or improvement is claimed.

All 36 responses are official train and come from the old design-exposed 36-response roster. Data2txt sources **13717 and 14637 both actually belong to source_train in the full reconstruction training settings**. QA sources 15220/15475 and Summary sources 11951/13514 are not in source reconstruction training. This is not a source-heldout natural-transfer result.

Independent final verification covered the feature prepare artifacts, completed feature artifacts, all 7 training artifacts, all 37 prediction artifacts, settings/parent-manifest links, 53 feature live-and-snapshotted code files, 52 training live-and-snapshotted code files, and 53 prediction live-and-snapshotted code files. All 16 model/tokenizer files passed an independent fresh hash read, and feature-model inventories agree between training and transfer. The checkpoint is the complete, nonsanity, full-training checkpoint; prediction output membership agrees with its actual internal_split map for every response.

| Artifact | SHA-256 |
|---|---|
| Run document | `99bf39ffe47498432ea1cfa1cb286e96320dd25c37fbf601c17ea76f107c2670` |
| Old 36 input roster | `c2d5712bf5f08ddaf34427eb483e98f38433d9a09847a857efe44bbf0de0b264` |
| Feature prepare manifest | `63c5bc287892e370b276d2c7b2e46148fc69f2211d0dcdf83e06c85ed975e301` |
| Transfer features.npy | `4b5dbd7955c7f77f7276102e3d9f9b8545e2b7a80c81943cfbfe5302ea1e849b` |
| Transfer feature manifest | `4fca4ab9d81ec781f526e197ff0195f717e9bad4c7159f4dfad1231d864cc423` |
| Full head training manifest | `db00af44f3470a42747a384a8299f873926a48b0a513287b9d455fd41b1b65b0` |
| Full head checkpoint | `9c716c94ffffd7fc5d114f931f2973290f1dd798c227cef6d57dc2ddc1ed2ff6` |
| Prediction manifest | `20fce3dce6e7710f209e597bd3d2a4f5a62fb8f835f955f8778c24e90f358da0` |
| Complete machine evidence | `06876db18499f7b7951102fcc7dbccd75c271408cdfa168524a115e9e01dfb80` |

All individual input, code, model, checkpoint, settings, feature, response, and manifest hashes are saved in `refine-logs/source_relation_transfer_integrity_evidence_20260913.json`. This witness establishes reproducible candidate-transfer execution and complete accounting. It does not establish applicable-constraint ownership, a lookback-node detector, routing/aggregation diagnosis, continuous hallucination-span detection, or a complete hallucination detector.
