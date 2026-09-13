# 下一轮的可执行边界：先修完整来源图

SourceRel完整训练/自然迁移负结果见 docs/SOURCEREL_RESULTS_20260913.md。当前不继续单向量头、同池训练或native；先修所有source原文可寻址的候选图。该CPU阶段不证明图能识别归属，只纠正此前根本没有正确候选的表示缺陷。

1. 全部source scalar owner保留，包括unknown-valued字段及不限长度的review_text。
2. 字符串内部提供完整word atoms与surface components，保留decoded-char到原始Python字符串raw span的精确映射；不支持的literal spelling明确保留whole field并标component mapping unavailable，不能伪造坐标。
3. field/context/record使用provenance bundles；weekday-key inventory保存实际7key成员，但不证明回答需要all-days约束。query约束由后续typed/语义provider单独证明。
4. QA/Summary也保留全文、context envelopes和word/surface组件，不再整体域外。包络不是事件或单事实，包含关系不代表采用关系。
5. 所有kind均保留，不用scalar type硬过滤address数字。源图构造不读回答内容或幻觉标签；每个response记录原source_span以便原prompt坐标还原。

执行顺序：有意义的escape/Unicode/unknown/长评论/地址组件/weekday域非语义事实/一般文本provenance检查；完整冻结RAGTruth roster的source去重图编译（2965来源，17790refs），保存分任务/节点/组件/unknown/mapping-unavailable分母和全部哈希。预检查不靠出现率或标签选子集。必要时保留失败目录并修新版本，不覆盖。

模型尚待收敛：q_relation/q_owner/q_condition并无自动真值provider，不能先假定这些角色桶正确再写分数。当前只保留精确target-coordinate mask+完整上下文，删去all-surface-slot遮蔽。late interaction可作检索算子，但不能把MaxSim、同句/同bundle或已有七天key变成真假、共同事件、量词证明。下一GPU ranker前必须明确自动query表示、匹配目标、训练/自然视图差异与未知处理；不调用LLM自评标签冒充gold，不设任意10对照门限。
