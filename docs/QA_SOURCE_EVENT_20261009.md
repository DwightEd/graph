# Fixed QA source-event readout and graph results

The fixed primary `event_fisher_native` reached complete QA DEV AUROC/AP **0.805974979/0.248453289** and official QA TEST **0.814914777/0.203465731**. Both passed their predeclared point gates. The TEST is a historically exposed exploratory replication, and the default detector remains unchanged. The method uses cached observer measurements with zero natural hallucination-label fitting and zero new LLM/GPU forwards.

The DEV population is 1,008 answers from 168 sources, with 147,981 valid tokens and 12,123 positive tokens. TEST covers all 900 answers from 150 sources and all six generators, with 124,817 valid tokens and 6,751 positives. FIT contains 4,026 answers from 671 sources; FIT, DEV and TEST sources and answer IDs are disjoint. Historical pilot cases remain separately reported.

| Split | Fixed field | Token AUROC | AP | TP at old normal-token FP budget | FP | Normal-answer alarms |
|---|---|---:|---:|---:|---:|---:|
| DEV | `old_native` | 0.772523120 | 0.218084651 | 2,136 | 5,414 | 438/701 |
| DEV | `null_native` | 0.803398309 | 0.216436188 | 1,671 | 5,414 | 511/701 |
| DEV | `event_fisher_native` | 0.805974979 | 0.248453289 | 2,493 | 5,414 | 327/701 |
| TEST | `old_native` | 0.779724664 | 0.172794084 | 1,204 | 3,847 | 447/740 |
| TEST | `null_native` | 0.810774323 | 0.164607873 | 822 | 3,847 | 561/740 |
| TEST | `event_fisher_native` | 0.814914777 | 0.203465731 | 1,457 | 3,847 | 349/740 |

The matched budgets above use evaluation labels to choose diagnostic cuts. They are oracle comparisons, not deployment calibration. Each field also retains its original FIT-mixture95 cut. At the unchanged primary cut **0.6258911648439406**, TEST has 1,401 TP, 3,684 FP, 5,350 FN and 343/740 normal-answer alarms. A mixed-truth FIT quantile does not guarantee a normal-token false-alarm rate.

The primary's TEST AUROC increment versus `old_native` has source-bootstrap 95% CI [0.029935337, 0.041375475]. Its increment versus `null_native` has CI **[-0.000254638, 0.007944968]**, crossing zero; the corresponding DEV interval [-0.000541673, 0.005655853] also crosses zero. Both use 300 source-cluster repeats with seed 42. Passing point gates does not establish reliable superiority over null calibration in AUROC.

TEST within-answer AUROC improves from 0.726760637 to 0.756749618. Strict-first-error AUROC improves from 0.651676758 to 0.762487727, while strict-first AP falls from 0.006201020 to 0.004313729. At the matched normal-token FP budget, clean first-error detections are **DEV 9/307 to 4/307** and **TEST 3/160 to 3/160**. Better ranking does not imply successful early alarm behavior.

| TEST generator | Tokens | Positive tokens | Primary AUROC | Primary AP |
|---|---:|---:|---:|---:|
| `gpt-3.5-turbo-0613` | 12,415 | 142 | 0.828197245 | 0.059539770 |
| `gpt-4-0613` | 17,839 | 19 | 0.408931419 | 0.000895170 |
| `llama-2-13b-chat` | 25,117 | 1,559 | 0.786409708 | 0.184265998 |
| `llama-2-70b-chat` | 22,442 | 1,617 | 0.830811404 | 0.300979966 |
| `llama-2-7b-chat` | 30,721 | 2,307 | 0.746279063 | 0.179438594 |
| `mistral-7B-instruct` | 16,283 | 1,107 | 0.843137658 | 0.309436020 |

The pooled TEST result reaches 0.8; three generator groups remain below it. The GPT-4 subgroup has only 19 positive tokens. All generator rows are reported without selecting a preferred group.

For each actual generated token, the source readout consumes four cached native log probabilities: source present/absent, each with local/full context. For condition `c`, define the artificial equal-view probability `p_c = 0.5*exp(logp_c,local) + 0.5*exp(logp_c,full)`. The primary effect is the signed Bernoulli Fisher-Rao coordinate difference:

$$
\theta(p)=2\operatorname{atan2}(\sqrt p,\sqrt{1-p})\in[0,\pi],\qquad
e=\theta(p_{\mathrm{absent}})-\theta(p_{\mathrm{present}})\in[-\pi,\pi].
$$

Positive `e` means source presence suppresses the actual token under this mixture; negative `e` means it helps the token. This is a bounded change in an observed-token-versus-complement probability coordinate, not a factual probability or a measurement of semantic constraint binding. Equal-view mixing can cancel disagreements. Stable `log1p`/`expm1` and `atan2` formulas preserve representable endpoint behavior without adding epsilon or inventing complement mass from rounded logp=0.

The same event's logp difference and probability difference are fixed controls. Equal-source FIT empirical mid-CDF references preserve exact ties. Let `r=F_FIT(e)` and `z=F_FIT(0)`; the bounded zero-anchored source coordinate is `a=0.5+(r-z)/(2*max(z,1-z))`. Its zero-effect value is 0.5 and its slope is at most one. The unchanged route rank gives the unary `u=0.75*a+0.25*r_route`. TEST copies the frozen event/global references, CONFIG thresholds and native 16-bin null reference; none are fitted, selected or rebuilt on TEST.

The graph uses source-present post-token local attention from **layer 15**, averaged over its **32 physical heads**, with strict answer-history lags 1 through 8. Each raw edge is divided by `max(1,raw_degree_sender,raw_degree_receiver)`. All answer nodes enter the offline symmetric arithmetic objective before valid-token filtering:

$$
\min_s\;\frac12\lVert s-u\rVert^2+
\frac{\lambda}{2}\sum_{j<t}w_{jt}(s_t-s_j)^2,\qquad\lambda=0.5.
$$

The banded system `(I+0.5*L)s=u` gives the score field. This is a scalar continuity prior from one captured layer; it does not use signed head-value contributions or reconstruct the original generators' complete causal graphs. Source help can increase the probability of an incorrectly bound word, and graph smoothing can suppress a local error signal.

The complete TEST diagnosis places 3,843 of the 3,847 false positives above the same cut already at the unary stage. Of 5,294 false negatives, 5,054 are already below it in the unary. At the primary's matched cut **0.6241014589850521**, the graph removes 1,780 FP but loses 240 TP, while adding 5 TP and 4 FP. Among 2,685 error tokens with nonpositive source effects in both views, only one is detected at this matched cut; all 2,685 are missed at the unchanged FIT95 cut 0.6258911648439406. The cohort includes three tokens with a zero effect in one view. The matched cut is below 0.625, so these are observed threshold outcomes, not a universal proof that such a token cannot exceed any cut. Of 539 old-baseline TP lost by null calibration, the primary recovers 390 and still misses 149.

The historically named TEST word `not` (`12219:223`) is TP with unary/graph 0.682017/0.664411. The cited count `2` (`12297:106`) remains FN with 0.582602/0.573364. The words `private`, `normal` and `taxed` are reported in the historical pilot; their answer `15604` also belongs to unlabeled FIT and is absent from official QA DEV/TEST. The pilot is not an independent holdout. In that pilot, `private` and `normal` remain FN; the graph lowers `taxed` from 0.646924 to 0.570331. These cases do not justify moving a threshold or changing the primary after evaluation. Complete diagnostic exports contain every FP/FN/change, raw four log probabilities, source effects, anchors, unary/graph scores and neighbors.

Nine fields remain fixed: `old_native`, `null_unary`, `null_native`, and unary/native pairs for event Fisher, event logp and event probability. `old_native` is the historical **source-route** baseline, matched against archival `source_route_native_huber` within 1.1e-8; the actual TEST maximum difference is 9.67811e-9. `null_native` retains its matching native local/full logp-conditioned 16-bin FIT references. Controls are reported under their original names.

The previous KL-fidelity experiment kept the earlier source readout and changed only propagation geometry. Its fixed primary failed all four DEV gates: AUROC/AP 0.802642569/0.214660818, 1,632 TP at 5,414 FP and 515 normal-answer alarms, versus `null_native` 0.803398309, 1,671 TP and 511 alarms. The source-event experiment therefore changes the finite source readout while retaining the earlier arithmetic graph. The fixed logit and other propagation controls were not promoted after evaluation.

The seven scientific source files are mapped below. Existing shared helpers provide native cache/identity loading, zero anchoring, route references, graph normalization and evaluation.

| Source | Responsibility |
|---|---|
| [source_event.py](../experiments/flow_latent/local_graph_transport/source_event.py) | Stable equal-view log mixture; bounded Fisher coordinate; same-event logp/probability controls; finite/nonpositive input contract. |
| [run_source_event.py](../experiments/flow_latent/local_graph_transport/run_source_event.py) | FIT event references and zero anchors; fixed nine-field FIT/pilot/DEV scoring; full-answer storage; score freeze before official label evaluation. |
| [run_source_event_test.py](../experiments/flow_latent/local_graph_transport/run_source_event_test.py) | All-900 TEST identities and raw cache loading; frozen references/thresholds; archival baseline check; nine-field freeze; one TEST evaluation. |
| [test_source_event.py](../experiments/flow_latent/local_graph_transport/test_source_event.py) | Synthetic endpoints, direction, view/condition swaps, extreme probabilities and Fisher-metric checks. |
| [kl_graph.py](../experiments/flow_latent/local_graph_transport/kl_graph.py) | Existing normalized arithmetic solver used by the source-event run; separate KL Newton solver and logit-coordinate control. |
| [run_kl_graph.py](../experiments/flow_latent/local_graph_transport/run_kl_graph.py) | Frozen FIT/pilot/DEV propagation experiment and its fixed controls/evaluation. |
| [test_kl_graph.py](../experiments/flow_latent/local_graph_transport/test_kl_graph.py) | Stationarity, curvature, arithmetic preservation, endpoint and propagation-control checks. |

Canonical records are the [DEV plan](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_20261009/EXPERIMENT_PLAN.md), [five DEV/FIT/pilot commands](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_20261009/RUN.md), [TEST plan](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_test_20261009/EXPERIMENT_PLAN.md) and [TEST RUN](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_test_20261009/RUN.md). The actual executions reused the existing research Python, CPU4/offline settings and environment spec `f3203164ff2c0938c39f4b7ac5d1cbc230f6f22571416130b9ebb66781409132`, with no installation or environment rebuild. Completed output directories refuse overwrite.

The following two commands are the exact recorded TEST invocation, run sequentially from `/share/home/tm902089733300000/a903202310/lys/research/graph` after the frozen scope/precheck and root GO. They are an execution record; the completed `outputs/qa_source_event_test_20261009` must remain preserved.

```bash
env PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_source_event_test --stage score
```

```bash
env PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_source_event_test --stage evaluate
```

The synthetic checks use the existing CPU environment and do not score natural examples:

```bash
env PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest -q experiments/flow_latent/local_graph_transport/test_source_event.py experiments/flow_latent/local_graph_transport/test_kl_graph.py
```

Both literal TEST commands exited 0 once. All 900 nine-field scores froze before official GT evaluation. Independent guards checked 31,287 direct files, 69 current project Python files with copied bytes, and all 900 exact consumed capture field pairs before/final execution. Capture scope covers only `local_attention[0,1:]` and `answer_ids`, not entire large NPZ files or unused hidden arrays. The legacy QA pack was opened lazily for only five identity arrays. The initial witness-helper schema error and the legacy NULL freeze's absent snapshots category remain recorded; current copied scientific code is independently bound. These are execution-integrity checks, not claims of semantic correctness or pristine historical capture provenance.

Current evidence is in the [DEV results](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_20261009/RESULTS.md), [DEV token diagnosis](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_20261009/TOKEN_DIAGNOSTICS.md), [TEST results](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_test_20261009/RESULTS.md), [TEST token diagnosis](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_test_20261009/TOKEN_DIAGNOSTICS.md) and [TEST execution witness](/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_test_20261009/EXECUTION_WITNESS.json). Full-node arrays, score freezes and complete FP/FN/change CSVs remain under `outputs/qa_source_event_20261009` and `outputs/qa_source_event_test_20261009`; the earlier KL arrays remain under `outputs/qa_kl_graph_20261009`.
