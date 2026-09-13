# GroundedGraphAdapter source-erasure fresh document witness — 2026-09-13

Fresh document execution **passed, exit 0**, followed by a separate CPU-only independent audit **passed, exit 0**. All 48 source-validation examples completed, with 48 actual Llama forwards and 1,861 common coordinate anchors. This is an execution/integrity witness; it does not establish useful natural detection, semantic ownership, or native LLM routing.

I was the fresh document-following agent, not the erasure implementation author. I read `docs/GROUNDED_GRAPH_ERASURE_RUN_20260913.md`, `.aris/compute/local.md`, the `run-experiment` skill and compute environment contract, and `refine-logs/grounded_graph_erasure_engineering_20260913.md` (Critical 0 / Required 0; 11 combined CPU tests). I did not change experiment/dependency code, install packages, overwrite outputs, retry the GPU invocation, or read RAGTruth gold annotations. The failed Qwen-generated source-template data were not used.

Preflight verified the natural evaluation manifest was already `complete`, evaluator PID 177870 was absent, and GPU 0 was idle at 1 MiB / 24,564 MiB. The unchanged environment spec canonical hash was `03909e02`. Training and reconstruction feature parents were complete; all 55 training dependency hashes matched the live files. A 65-file preflight snapshot is embedded in the independent audit JSON. The new output and tee log did not exist before invocation.

The exact documented command ran once, inside a shell with `pipefail`, with stdout/stderr additionally captured by `tee /tmp/grounded_graph_erasure_v1_20260913.log`:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_erasure --training outputs/grounded_graph_train_v1_20260913 --features outputs/grounded_graph_features_v1_20260913 --output outputs/grounded_graph_erasure_v1_20260913
```

Background tool session: `50716`; Python PID: `178013`; shell PID: `178010`; GPU: `0`. Checkpoint loading and GPU allocation were observed. The largest sampled GPU use was 15,714 MiB; this is not a measured peak. Source progress advanced through every count 1–48. The source loop reported 22.095 seconds; the runner reported 65.978 seconds including final parent verification, excluding initial parent checks/model/adapter loading. No exact whole-command wall timer was recorded. The session returned exit 0, the completion manifest exists, the PID disappeared, and GPU returned to 1 MiB. There was no invocation or CLI divergence from the document.

The independent CPU audit (session `39868`, exit 0, no model or training forwards) verified the following directly from bytes and coordinates:

- Exactly the original ordered 48 validation source IDs, 16 each for QA, Summary, and Data2txt; all available and complete. Feature ancestry is `source_reconstruction`.
- 864 artifact hashes: erasure 97, feature capture 481, feature preparation 241, training 45. Four manifest/settings bindings, all 65 preflight hashes, and 165 live/executed-code pairs across feature/train/erasure snapshots matched. All 16 model-file size/mtime records and all 12 small model/tokenizer file hashes matched; the GPU runner checked its full model identity before loading. The independent audit did not separately rehash the four multi-GB weight shards.
- Re-tokenized all 48 original prompts and responses with the frozen tokenizer. Rebuilt source offsets and all 22,861 node token memberships from raw spans and bundle membership. Verified exact packet digests, original feature receipts, the pre-token shift, target IDs, response offsets, and owner identities.
- Replacement was the single ASCII-space token ID 220 at exactly the union of prompt-token positions belonging to the 1,139 target-owner nodes. All 22,494 selected coordinates were in source prompt spans; 22,019 input IDs changed and 475 were already space tokens. All other input IDs were unchanged.
- All 26,526 observed response-prefix tokens were unchanged; all 26,574 target tokens were unchanged. The 48-token difference is one final target per source, which is predicted but not included in its pre-token model input.
- Each receipt recorded one actual model forward and one final-norm hook. Both saved erased arrays per source were finite float32 with the original coordinate dimensions; all 96 raw array hashes matched. They contain 93,638,656 source values and 108,847,104 query values. Each erased source and query array differed from its original, and every erased owner vector remained nonzero. The frozen runner pools new Llama final-norm states at original source coordinates; it does not zero old vectors.
- All three branches and both adapter arms used the same 1,861 anchor targets. Original versus graph-only-erased base log probabilities were exactly equal for every source, consistent with fixed original queries. Every metric vector had its full denominator and finite values, all pointer probabilities were bounded, and adapter gain equaled adapter logp minus base logp (maximum floating residual 1.19e-7). The eight reported aggregate gain/probability drops were independently reproduced.

All means below first average anchors within each source, then average sources. Gain is adapter target logp minus base target logp; drop is original gain minus erased gain. Positive and negative source-level changes are both retained.

| Erasure branch | Adapter | Original mean gain | Erased mean gain | Mean gain drop | Positive / negative / zero sources | Mean pointer probability drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| graph_source_erased | graph | -0.00149246 | -0.00083525 | -0.00065720 | 23 / 25 / 0 | 0.39927561 |
| graph_source_erased | no_edges | -0.00150810 | -0.00040744 | -0.00110066 | 21 / 27 / 0 | 0.35258024 |
| full_source_erased | graph | -0.00149246 | -0.04279738 | 0.04130492 | 38 / 10 / 0 | 0.46680067 |
| full_source_erased | no_edges | -0.00150810 | -0.01502190 | 0.01351380 | 32 / 16 / 0 | 0.40933255 |

Each of the four branch/arm combinations reduced coordinate pointer probability on all 48 sources. Gain drops vary by task:

| Branch | Adapter | Task (16 sources each) | Mean gain drop | Positive / negative |
| --- | --- | --- | ---: | ---: |
| graph_source_erased | graph | Data2txt | -0.00041358 | 5 / 11 |
| graph_source_erased | graph | QA | -0.00319323 | 4 / 12 |
| graph_source_erased | graph | Summary | 0.00163520 | 14 / 2 |
| graph_source_erased | no_edges | Data2txt | -0.00018010 | 9 / 7 |
| graph_source_erased | no_edges | QA | -0.00260586 | 5 / 11 |
| graph_source_erased | no_edges | Summary | -0.00051602 | 7 / 9 |
| full_source_erased | graph | Data2txt | -0.00055026 | 7 / 9 |
| full_source_erased | graph | QA | 0.07099715 | 16 / 0 |
| full_source_erased | graph | Summary | 0.05346789 | 15 / 1 |
| full_source_erased | no_edges | Data2txt | -0.01396900 | 5 / 11 |
| full_source_erased | no_edges | QA | 0.04183422 | 14 / 2 |
| full_source_erased | no_edges | Summary | 0.01267619 | 13 / 3 |

The source-coordinate pointer is sensitive to erasing its owner payload. The reconstruction-gain evidence is weaker: original anchor mean gain is slightly negative for both adapters, graph-only erasure gives a small negative average gain drop, and full erasure makes gain more negative chiefly in QA and Summary. Data2txt does not have a positive average full-erasure gain drop. Thus a positive full-erasure drop should not be described as removal of a demonstrated positive reconstruction benefit. Full erasure changes both source and query states; it cannot isolate the graph source path. The graph-only branch holds a query that already contains original prompt information.

The same-position space intervention is an explicit perturbation, not a natural missing-evidence example. QA/Summary owners can be contexts rather than a single value. Duplicate or substitute evidence may remain, so retained gain alone does not establish a formatting shortcut. This witness supplies no semantic-necessity certificate, no hallucination binary truth, and no native-routing conclusion. Natural-evaluation interpretation is outside this agent’s label-blind audit. The document’s requirement that both natural evaluation and this source-dependence control support benefit remains an empirical gate; completion alone does not satisfy it.

Artifacts: [independent audit JSON](grounded_graph_erasure_independent_audit_20260913.json), [erasure summary](../outputs/grounded_graph_erasure_v1_20260913/summary.json), [erasure manifest](../outputs/grounded_graph_erasure_v1_20260913/manifest.json), [erasure settings](../outputs/grounded_graph_erasure_v1_20260913/settings.json). The audit JSON contains every selected source ID, per-source coordinate counts, complete task statistics, and preflight hashes.

Erasure manifest SHA256: `984a870df71cfe22a15924b4f20118e32ad4cd9a7833b6f1ad3682f944ea2abf`. Settings SHA256: `047da13a960339989ffce5c4bb20c6260b43ee81f3cac38de42d90f4f7da2fd5`. Executed erasure code SHA256: `355b8f6a3f4fceb89101c14591b05c472afdc94d0eb0bd7951438715a372d9c2`.

Independent audit timestamp: 2026-09-13T07:24:29.735294+00:00.
