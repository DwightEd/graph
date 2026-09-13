# Artifact manifest（2026-09-11）

| 文件 | 内容 |
|---|---|
| round-0-initial-proposal.md | 冻结问题锚点、初始文献和两路线比较 |
| round-1-review.md … round-5-review.md | 五轮独立审查原文 |
| round-1-refinement.md | 强零模型、精确信号／路径、主读出完整稿 |
| round-2-refinement.md | 共享尺度／active mask 与二步解释修订完整稿 |
| FINAL_PROPOSAL_20260911.md | 最终完整方法方案的日期版本 |
| FINAL_PROPOSAL.md | 最终完整方法方案的固定入口 |
| REVIEW_SUMMARY_20260911.md、REVIEW_SUMMARY.md | 审查摘要的日期版本／固定入口 |
| REFINEMENT_REPORT_20260911.md、REFINEMENT_REPORT.md | 演化、反驳记录与剩余问题 |
| score-history_20260911.md、score-history.md | 主观方法评分演化，不是实验结果 |
| REFINE_STATE.json | 五轮上限结束状态；科学结论仍为 REVISE |

实现验证见 ../docs/IMPLEMENTATION_REVIEW.md。自然数据准备及历史 smoke 输出在被 Git 忽略的 outputs/path_residual_20260911/。没有新方法的真实预训练模型检测性能产物。

2026-09-13 当前迭代及时间戳版本索引见 [项目输出索引](../MANIFEST.md)。O4/O5已执行，P2全量进行中；当前方案和状态更新于2026-09-13T01:46:22+08:00。旧条目保持历史状态。

## Native audit implementation 20260913_042134

新增route_graph/audit_*.py、causal_groups.py、frozen_reader.py、evidence_anchor.py及真实native后端。入口audit_runner，调度experiments/interleave_native_audit.py。自然36条清单outputs/native_audit_design_20260913；审查native_{validation,pipeline,scheduler}_engineering_20260913.md。此时未启动GPU，部署前复审尚未关闭。

Native audit 2026-09-13T04:47:53+08:00: docs/NATIVE_METHOD_MODEL_20260913.md, docs/NATIVE_AUDIT_RUN_20260913.md, outputs/native_audit_v1_20260913/settings.json, experiments/evaluate_native_audit.py, experiments/summarize_native_audit.py; full36 launch preflight, no empirical result claimed.

Native v1/v2 actual results: docs/NATIVE_AUDIT_V{1,2}_RESULTS_20260913.md and outputs/native_audit_v{1,2}_20260913/evaluation.json. Both native0. V3 frozen and running: docs/NATIVE_AUDIT_V3_RUN_20260913.md, outputs/native_audit_v3_20260913/settings.json; executed_code/manifest.json includes26 sources. Interface review cloze_anchor_v3_integration_review_20260913.md closed. New primary methods read: native_graph_literature_2026_followup_20260913.md; structural reviews cloze_next_structural_review_20260913.md and2026_primary_diff_structural_review_20260913.md (supervised mainline suggestion not accepted).

## Event candidate model checkpoint — 2026-09-12T23:46:48.405954+00:00

- docs/EVENT_MATCHER_MODEL_20260913.md — implemented CPU proposer/capture interfaces and explicit missing GPU/semantic integration; no effectiveness claim.
- route_graph/event_matcher.py, route_graph/span_feature_capture.py and tests — 44 CPU tests, independent C0/R0 review.
- refine-logs/event_matcher_minimal_spec_review_20260913.md, event_matcher_engineering_review_20260913.md, event_graph_end_to_end_failure_review_20260913.md — real collaboration reviews; Codex MCP unavailable.
- outputs/source_event_inventory_20260913 — 36 response/6 source inventories, 2 literal graphs, 47 file hashes verified; no labels/model calls.


# Native v3：完整负结果，2026-09-13

36回答来自6个官方train来源（每任务2来源×6生成器），已经是开发批次，不能作独立泛化证据。输入、模型和代码冻结；完整评价后才加入RAGTruth标签。

|项目|实际结果|
|---|---:|
|A/B/C/D/merge响应数|各36|
|词数 / 标注错误词|4733 / 260|
|语义或机制评分覆盖|0|
|错误召回（含弃权）|0|
|问题 / uncertain / invalid|368 / 271 / 97|
|native forwards / 原始输入介导产物|0 / 0|
|来源reader实际请求|339|
|A阶段所有reader返回请求|1064|

所有词弃权时AUROC 0.5、source-balanced AUPRC 0.0545634633仅为常数分数诊断；precision未定义。不得解释成有效检测器。QA 1463词/168错误；Summary 1536/38；Data2txt 1734/54。

实际原始输出诊断显示：339次来源生成中165解析成功、174失败。只删除末尾句号的只读what-if能恢复140个完整根，但305个诊断完整根的233个answerable中160仍缺非空answer或quote；102个包含原文不存在的引用，41个引用中含<MISSING_SLOT>。不把这些what-if回写预测。5个多mask非目标失败span只是reader报告的诊断模式，不是约束归属真值。

独立审计核对180个阶段产物和上游链、输入/标签哈希、完整词坐标和评价时序，完整性通过，科学主张不受支持。自动定位未测；唯一回看节点准确率在无独立节点标注时不可识别。观察者为Llama回放六模型文本，并非原生成器轨迹。

运行见证：文档命令执行一次；wrapper/native均exit0；population恢复PID153564，12574→12602→12623且失败0，旧12574份manifest及冻结文件/代码一致。审计报告中的pending是审计当时快照，以独立见证的最终运行记录补充，不改写旧审计观察。

关键产物：
- outputs/native_audit_v3_20260913/evaluation.json（标签只用于完成后的评价）
- outputs/native_audit_v3_20260913/diagnostic_summary.json（无标签）
- refine-logs/native_v3_source_reader_barriers_after_A_20260913.json（无标签，非恢复预测）
- refine-logs/native_v3_integrity_20260913.{md,json}
- refine-logs/native_v3_doc_witness_20260913.md

本结果促成架构调整：严格A/B审计不再作为所有检测分数的入口；新主线用全span软图风险并与同reader无图比较，严格证据作为覆盖层。设计可编码不等于有效，下一批必须检验图增益、native覆盖与连续错误边。


## 最新运行：soft graph v1 完整36回答（2026-09-13T08:41:31.225750+08:00）

新的主线检测器已冻结并由独立文档见证启动，child PID 155856，wrapper session67861，输出 graph/outputs/soft_graph_v1_20260913。settings digest 9833b851b479f6ffe53d4a601a947d53fc13eb4d161946daf5f2e9e33fa32cd8，文件SHA fb13d3dabe6333c15f00ae722993148018cecf8c949010f8b01172788753bec4，47份代码冻结。72项相关CPU检查通过，独立runner/pipeline复核9通过，Critical0/Required0。当前没有新效果评价。

执行A结构图、B高维特征/候选、C有限关系判断、D每span原始输入介导/定位、merge全词四版本评分。完整结构见 graph/docs/SOFT_GRAPH_METHOD_20260913.md，运行指令见 graph/docs/SOFT_GRAPH_V1_RUN_20260913.md。原全量暂停于13669/17790、失败0，由同一独占wrapper完成后恢复。未修改任何旧冻结代码/结果，未跟踪文件均保留。

本版仅以观测回答span的完整logP测依赖，不把它冒充正误决策差。严格A/B层仍单独保留，本批不声称已区分错误路由和正确路由后聚合失效。标签仅在全部预测完成后评价。v3全部弃权/native0的完整负结果已归档；以下状态均按时间视为历史。

---

Artifacts: route_graph/soft_graph_{energy,structure,phases,runner}.py, target_dependence.py; experiments/interleave_soft_graph.py/evaluate_soft_graph.py; tests/test_soft_graph_{energy,pipeline,freeze}.py and test_target_dependence.py; exact CPU mapping preflight36/4733words/276fallback leaves. The temporary heterogeneous-control KeyError was caught by the structural-control regression before freeze and corrected; 70+2 tests pass.

## 2026-09-13 surface negative → fact scope / typed native iteration

- docs/SURFACE_OWNER_V1_RESULTS_20260913.md; outputs/surface_owner_v1_diagnostics_20260913: completed145pair/1619target read-only diagnostics; no native/labels.
- outputs/reader_output_cpu_audit_20260913; refine-logs/reader_cpu_doc_witness_20260913.md:10 fixed requests,30CPU forwards, no semantic-success claim.
- next_iteration/fact_scope.py, fact_scope_mask.py; tests/test_fact_scope*.py:23pass, supplied-assumption graph solver.
- refine-logs/fact_scope_primary_methods_20260913.md; semantic_provider_decision_20260913.md; fact_scope_graph_revision_20260913.md: primary method readings and accepted design.
- next_iteration/typed_hours.py, typed_hours_prepare.py; tests/test_typed_hours*.py; typed_hours_engineering_20260913.md: narrow all-week source algebra and exact B/A handoff; full CPU prepare pending at this entry.
- next_iteration/typed_native_measure.py, typed_native_runner.py; experiments/interleave_typed_native.py; tests/test_typed_native_measure_review.py: bounded layer/query search, typed A origin and raw native node features; engineering C0/R0, GPU pending.


## 2026-09-13T12:54:03+08:00 — population / typed results and SourceRel-Mini implementation

- Population COMPLETE17790/17790 failed0; `refine-logs/ragtruth_full_evaluation_review_20260913.{md,json}`: all36 groups, hashes verified; descriptive observer scope only.
- `docs/TYPED_HOURS_RESULTS_20260913.md`, `docs/TYPED_NATIVE_RESULTS_20260913.md`: full17790 compile2contrasts; soletrain native90forwards/raw/strong0. Freshdoc witnesses completed; oldpending entries historical.
- `outputs/typed_hours_owner_metadata_correction_20260913.json`, `next_iteration/typed_owner_metadata.py`: additive370 root/name-ID correction; independent impact review, no frozen rewrite or GPU rerun.
- `next_iteration/source_relation_{data,model,features,train,transfer}.py`; current plan `refine-logs/SOURCEREL_EXPERIMENT_PLAN.md`, method `docs/CURRENT_METHOD_20260913.md`.
- M0 fresh883sources/7064queries; fullCPUfeaturepreflight38541docs/3954619tokens/max1200; no truncation. Data/features/train independent reviews C0/R0 (scope engineering). TFIDF valtop1.8059816/top5.9900307. Sanity real21featureforward and2head epochs exit0; no effectiveness claim. Full features/20epochs/natural transfer pending at this entry.


## 2026-09-13T13:26:02+08:00 — SourceRel complete / post-first complete / inventory repair

- `docs/SOURCEREL_RESULTS_20260913.md`: actual full20epochs/source-val95.40%, samefield-otherrecord77.27% vsTFIDF100%; natural M2complete622proposals/1619slots, structural negative. All4 freshdata/features/train/transferwitnesses completed with hashes; no RAGgoldtrain/naturalGT.
- `next_iteration/population_postfirst.py`, `outputs/population_postfirst_v1_20260913`, `refine-logs/population_postfirst_results_review_20260913.md`: actual rootCPU session97043exit0,17790verified,504oldmetricpairs matched; post-firsthistoryJS.5127/entropy.5480; descriptiveonly.
- `refine-logs/source_relation_failure_refinement_20260913.md` superseded at relevant boundaries by `source_relation_failure_boundary_revision_20260913.md`; current `CONSTRAINT_INVENTORY_PLAN_20260913.md` / `next_iteration/constraint_inventory.py`: codewritten, engineeringreviewrunning/fullCPUcompilepending, no newrankerGPU.


## 最新核验与方法修正（2026-09-13T14:00:27.286414+08:00）

完整来源库存已实际完成并独立审计：2965来源/17790引用，1241059组件，unknown3049、长文本字段4083、mapping failure0；CPU339.217秒，执行与独立审计均exit0。报告 graph/refine-logs/constraint_inventory_doc_witness_20260913.md。SourceRel与post-first的完整负结果保持，不能将字段重建95.4%当自然归属准确率。

用户指出反复局部实验的死胡同后，进一步明确：冻结Qwen推理图只作为外部语义参照，不足以回答内部信息是否能判定归属。新 docs/REASONED_GRAPH_METHOD_20260913.md 和 next_iteration/reasoned_graph{,_runner}.py 已写，工程审查尚未完成，未prepare/未GPU执行；其当前提示图仅field/record/context粗粒度，完整组件图仍保存在库存，不能混称。当前还在审查联合来源指针与grounded token重建的轻量内部图模型目标，未实现/训练，不宣称已选定有效结构。

当前没有GPU实验在运行。准确回看、适用归属、路由/聚合、连续范围四项仍未闭合。保留所有旧文件与分支，以下运行状态均为历史。


## 当前联合图模型与实际状态（2026-09-13T15:07:49.713783+08:00）

主线已写成单一 GroundedGraphAdapter：完整source图的4096维特征，两步128维消息传递，真实预测前h[t−1]查询，门控残差经冻结LMhead。固定自然检测score=logp_base−logp_adapter。当前方法/执行入口为 graph/docs/CURRENT_METHOD_20260913.md、GROUNDED_GRAPH_MODEL_20260913.md 和 GROUNDED_GRAPH_TRAIN_RUN_20260913.md。

Qwen来源模板实验因确证错误owner停止：6/240源，19/48机械可用但有段落归属错误，已落盘7298forward，SIGINT130；中断批额外forward未知。所有原始文件保留，不用于训练。替代的来源原文坐标重建已完成240源、9220锚点，192/48全文SHA隔离；训练特征135519token/240forward，自然36答5840token/4733词/36forward均exit0。自然开发6源中13717在source-only训练内，单独报告重合，不称独立test。

训练/自然预测/评价/实际payload擦除代码均已实现并完成工程检查。训练执行见证正在启动，尚无新自然检测数值；不称方法收敛。旧SourceRel95.4%仍只属弱字段检索；全量17790的post-first弱/负结果保留。准确回看、自然适用约束、原LLM路由vs聚合、连续影响范围仍未闭合。

以下为保留的历史检查点；当前状态以上述入口和真实manifest为准。



## A2-v1完整负结果与v2受控迭代（2026-09-13T15:46:56.948789+08:00）

v1训练/自然预测/评价/48来源实际擦除均完成。固定graph差分source-balanced AUROC全词0.505927、首错后0.512073，base NLL分别0.535477/0.516545；36答/6源/4733词是反复使用的开发集，6答来源进source-only预训练，30source-unseen同样无改善。结果/实现/完整分母见 graph/docs/GROUNDED_GRAPH_V1_RESULTS_20260913.md。

固定query擦来源X后坐标指针概率降0.399276，生成gain drop为−0.000657；原始gain本就−0.001492，因此不能把full擦除的正drop叫正收益被移除。source-copy任务中H_full已经含完整source，图生成分支可能被绕过。当前v2保持240来源、划分、9220锚点、参数、10epoch和loss，只换为真实all-source-erased H_empty查询及解码基底，原X保持；新特征/训练/预测/评价代码已工程闭合，fresh-doc见证正在启动真实capture，来源依赖诊断待审查。尚无v2数值。

当前方案与可执行指令：graph/docs/GROUNDED_GRAPH_RESTORATION_V2_20260913.md、GROUNDED_GRAPH_RESTORATION_V2_RUN_20260913.md。source48 anchor正增益+固定query真实X擦除降益为机制门控；全词和strictpostfirst两个差分分别报告，不翻方向。准确回看、自然适用约束、原模型路由/聚合、连续范围仍未解决。外部Codex MCP不可用，科学审计不称PASS；保留所有旧文件/未跟踪文件和分支。

以下均为保留的历史状态；当前以最新段落及实际manifest为准。

