# Refactor validation, 2026-09-30

- Full retained repository suite: **412 passed in 14.70s** (`python -m pytest -q`).
- All 19 explicit root CLI stage help routes passed.
- Frozen fixed baseline: exact scores and thresholds for QA, Summary and Data2txt in every fit/development/test partition; all **2700 test answers / 424408 test tokens**, maximum score error **0**. No annotation array accessed, no model forward needed.
- New native message checks: eager and SDPA tiny Llama, causal ports, no-op replay, whole-head finite/VJP agreement, physical site/alignment rejection and hook cleanup. Tiny-model checks do not validate real-8B fidelity.
- Graph core: exhaustive random tiny-graph optima, 240 continuous-case min-marginals, tied cuts, independent direct scores, scope/fork/support exits, continuity and capacity limits, missing evidence, and empirical reference invariants.
- Fresh `experiment-bridge` code review (gpt-5.6-sol xhigh; same-family/provisional): **ACCEPT bounded refactor/core** after fixing unknown-gate coercion, invalid native coordinates, and fork/anchor validation. No remaining blocking/high software finding in reviewed scope.
- Static retained-code audit: no missing local imports or syntax errors; existing untracked routing_likelihood imports remain valid.

Environment: Python 3.11, torch 2.8.0+cu126, transformers 4.57.1, NetworkX 3.6.1; tests execute on CPU. Raw logs are in `outputs/graph_refactor_20260930/`. Shared review/deletion/experiment records remain under `/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/gsm8k_states_20260929/`.

No new natural-data detection efficacy is claimed. The semantic observer, source-disjoint reference assembly and 8B finite fidelity/graph ablations remain pending. See [architecture and theory](ARCHITECTURE.md) and [cleanup scope](CODE_CLEANUP.md).


# RAGTruth 全测试复验与防御性检查清理

2026-09-30。用户要求删去冗余检查并运行三任务测试。当前新消息图尚未接通自动关系提案、R/E/U回答构造及独立参考校准，本轮没有新图检测成绩，也没有用空事件或人工事件冒充全量预测。

本轮实际完成的是原模块中的旧 fixed 与三个独立 token 标量对照的完整CPU重评分和官方标注评价；真实模型测量复用历史缓存，新8B前向为0。每任务900答/150来源，合计2700答、424408有效token，全部生成器、官方test。已有test被历史研究多次查看，本轮是探索性复验。

## 运行

在 graph 根目录、现有 research 环境执行：

```bash
python main.py token baseline --stage run-test --output outputs/token_backtrace_readout_20260930_v2/test_recheck
```

实际exit 0。入口按 fit → score_test → evaluate_test 顺序执行；训练参考与阈值不读标签，全部三个任务分数冻结后才读取评价标注。固定基线本轮从标量重新计算，没有复制旧预测；对比指标、参数及阈值均未据本轮test调整。

## 结果

| 任务 | 方法 | AUROC | AP | TP | FP | FN | 完整字符span覆盖 | 正常回答有报警 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QA | base | 0.890665 | 0.353886 | 1722 | 2008 | 5029 | 25/235 | 10.54% |
| QA | token_pair | 0.743627 | 0.138181 | 987 | 4320 | 5764 | 0/235 | 63.78% |
| QA | odds_pair | 0.802761 | 0.163808 | 1057 | 4078 | 5694 | 0/235 | 60.68% |
| QA | odds_full | 0.799441 | 0.160594 | 1006 | 4078 | 5745 | 0/235 | 62.16% |
| Summary | base | 0.752177 | 0.158178 | 923 | 4324 | 2562 | 36/244 | 25.29% |
| Summary | token_pair | 0.630378 | 0.049232 | 401 | 5261 | 3084 | 0/244 | 95.11% |
| Summary | odds_pair | 0.677592 | 0.056275 | 448 | 5290 | 3037 | 0/244 | 95.40% |
| Summary | odds_full | 0.689958 | 0.057721 | 465 | 5488 | 3020 | 0/244 | 96.26% |
| Data2txt | base | 0.758040 | 0.104684 | 1064 | 6878 | 6343 | 86/1054 | 29.91% |
| Data2txt | token_pair | 0.607302 | 0.055681 | 438 | 8207 | 6969 | 3/1054 | 99.69% |
| Data2txt | odds_pair | 0.627879 | 0.060129 | 581 | 8049 | 6826 | 2/1054 | 100.00% |
| Data2txt | odds_full | 0.620336 | 0.061215 | 634 | 8032 | 6773 | 6/1054 | 98.44% |

base 是旧固定无监督基线，保留历史单元/窗口聚合；token_pair、odds_pair、odds_full 均逐token评分，不做邻词风险平均。主对照 odds_full 三任务均弱于base；来源bootstrap的AUROC差95%区间全部低于0。字符并集与完整span覆盖包含空格，是本项目自定义定位指标，不声称官方scorer。一次重叠不算整段检出。

## 代码与复核

- 在原 messages.py、readout.py、global_graph.py、pipeline.py 删除重复的类型/形状/范围/有限值/schema检查及仅验证报错的测试，不新增算法目录。保留原生因果掩码、未知scope/缺测含义、校准分辨率、容量公式、钩子释放。算法所需布尔输入和物理地址由调用端按既定格式构造。
- 52项token_backtrace数值/算法测试通过；全仓393项通过（15.88秒）。前轮412项减去19项非法输入报错测试，不能把测试数减少解释为方法提升。
- 三任务四种分数及token地址与历史v1逐值一致；旧fixed同时对独立历史基线核验。来源train/test互斥；AUROC/AP/TP/FP另算一致。这是同agent复核，不冒称fresh外审。
- 一次复核脚本起初误把只含odds_full/logic的v2包当四方法包，报KeyError；改为实际包含四方法的v1包后通过。没有修改评分或重跑模型。

## 仍未完成的请求范围

新消息图在三个任务全测试集上的效果仍未验证。缺少的是实际检测流程：自动提案与表达scope → 来源对比 → 对齐R/E/U → 原生消息VJP → 独立A/B校准 → calibrated events。现有graph score只是最后的已测事件读出；本次清理不能补上这些算法模块。后续应在原模块接通这条链，再运行新图全测试，不能继续把基线复验、toy图割或单元测试称为新方法完成。

原始产物：`graph/outputs/token_backtrace_readout_20260930_v2/test_recheck/`，含protocol、calibration、thresholds、三个任务test分数及evaluation_labels、test_frozen、test_results、verification、metrics.csv和pytest.log。旧结果原样保留。
