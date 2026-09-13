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
