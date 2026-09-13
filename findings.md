# Research findings

2026-09-12，临时研究判断：[pending Codex review]，claim_supported=no。

本轮两层attention上下文重加权没有在预声明主层超过直接读取；128条件全模型正确，无法识别错误。旧路径residual在108个完整且参考适用的测试回答上未优于熵，报警召回亦无优势证据。C1/C2均不成立为当前论文主张。原始捕获provenance不完整、来源覆盖排除、测试数据已查看的限制均保留。

失效假设：G主要重加权已经足以选值的直接边；被污染的历史也可能改变上下文代理，而模型的真实V/O/MLP仍保持正确选择。该解释尚未由干预证实。下一轮先用独立的同对象不同动作任务验证关系可识别性，再研究有符号消息。不得在本批数据调层/翻转风险方向/拼入置信度后继续当盲测。

完整数字、对照、来源区间、复现和限制见 [本轮报告](docs/METHOD_ITERATION_20260912.md)。

最终核验还发现候选竞争不充分：两种构造数值同在native top4的条件只有62/128，原始/逆序仅5/6条。必须区分source token覆盖与关系候选覆盖，不能用空格等第二候选制造已识别关系的印象。

2026-09-13：O4固定数字/位置、Before/After互换，4个受控世界原生随关系选对；最终query E4/4翻转、X0/4。N/A/G仅3/4、2/4、2/4，未实现自动归属。O5只有t131单query E充分，但此前全部query联合E也充分；32单层均不充分；source-rise top8对单点充分集合覆盖0/1。不能假定唯一回看节点或用总量替代归属。原始阶段缺失时长是下一项，不把添加规则后的成功写成自然修复。4077条QA部分评价已查看，后续确认性评估另冻方法/来源。详见docs/RELATION_MECHANISM_RESULTS_20260913.md；claim_supported=no，REVIEW_UNAVAILABLE。


# Native v1 natural-data result and next iteration

Recorded 2026-09-13T05:36:30.868925+08:00. Development batch: 36 responses from six official train sources (two per task, six generators per source). This is not the full RAGTruth population and not a test-set result. The observer is Llama3.1 replaying responses; Qwen reader predictions are not ground truth.

All A/B/C/D/merge predictions completed before the independent annotation join. Official response annotations were verified against the pre-run evaluation manifest, exact identities and response hashes. Evaluation: outputs/native_audit_v1_20260913/evaluation.json. Executed sources are preserved under executed_code/; current source has begun v2 and must not be substituted when reproducing v1.

| Quantity | Result |
|---|---:|
| All response words | 4733 |
| Annotated error words | 260 |
| Questions: invalid / uncertain / supported / unsupported | 103 / 56 / 13 / 0 |
| Semantic scored words | 287 (6.0638%) |
| Annotated error words with a semantic score | 3 (1.1538%) |
| Error words detected at frozen threshold 0.8 | 0 |
| Recall including abstentions | 0 |
| Actual native forwards / processed native tokens | 0 / 0 |
| Mechanism coverage | 0 |

The all-word source-balanced semantic AUROC is 0.525407 and AUPRC 0.057373; mechanism ranking is the abstention baseline (AUROC 0.5). These do not establish a usable detector. Thirteen supported predictions are reader estimates, not thirteen independently verified correct answers. Zero risky windows means this run provides no test of the internal graph's ability to distinguish constraint ownership. It neither establishes nor refutes sufficiency of internal information once valid natural decision windows exist.

Failure diagnosis before annotation join identified broad paragraph/answer extraction: 75 self-QA answer mismatches, 17 hidden-answer premise overlaps, 7 absent/ambiguous quotes; 19 source/verification invalid-JSON calls and 9 citation failures. These categories can overlap. The new design was motivated by these pre-label front-end failures, not by selecting favorable labeled examples or changing thresholds. The same six-source roster is now explicitly development data. Any later generalization claim requires new untouched sources.

V2 replaces freeform question generation with atomic event-role extraction, exact surface/antecedent bindings and deterministic leave-one-role-out questions. Every remaining role must have an exact separate condition-table entry, with source quotations tied to the same event. Missing non-target premises cause uncertainty, including for target-absence estimates. All earlier contrast/origin/control tests remain. CPU tests do not establish factual accuracy. V2 will rerun this identical full roster into a fresh output and preserve every failure/abstention.

Execution caveat: native audit child exit 0; wrapper exit 1 because its 120-second population restoration check expired during file validation. Original journal remains population_resume_unverified. Subsequent independent observation confirms original population restored as PID141861 and advanced 11515 to11562 with failed0. A new scheduler only increases this verification deadline to900 seconds; actual PID/state/held-lock verification remains required.

Result-to-claim: claim_supported=no for accurate automatic lookback localization, applicable-constraint identification, routing-derived hallucination detection, and continuous-error span recovery. Codex MCP unavailable; this parent judgment is [pending Codex review]. A separate collaboration reviewer is checking integrity, explicitly not impersonating that backend.


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


## A2-v1 完整负结果（2026-09-13）

自然检测有效、正确约束自动归属、准确回看、路由/聚合区分、连续错误因果范围：均无支持。当前模型可运行，但未收敛有效方法。停止增加同一任务epoch；下一版仅改变source信息路径，以真实全source擦除H_empty作query/残差基底，同时保留原X，独立检验anchor正恢复收益和固定query来源依赖。详见GROUNDED_GRAPH_RESTORATION_V2_20260913.md。

详细结果 docs/GROUNDED_GRAPH_V1_RESULTS_20260913.md；claim_supported=no [pending Codex review]，外部审计unavailable。新版须针对实测生成旁路，不重复指针好即归属正确的推断。
