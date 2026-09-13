# 原生消息图方法：实现与完整自然验证

2026-09-13。替代上一版“下一项O6”，O6已按用户要求停止作为下一实验。主线graph；reanchor保存机制任务和原冻结全量。完整结构见FINAL_PROPOSAL.md，第五轮规格可实现但实证未通过。

## 问题锚点

对冻结LLM，在不以幻觉标签训练的条件下，测量来源和历史的信息实际传递、聚合与输出采纳；区分疑似错误被继续沿用、被反对/纠正以及新陈述开始。事实区间和影响范围分别评价，熵只提供候选起点。

## B1：自动完整链条（MUST）

从已冻结、无标注的population inputs.jsonl取official_split=train。每任务按SHA256("20260913:native-audit:"+source_id)选前2来源，排除旧单例14375；保留6个原生成器，预计6来源36回答。选择不得读取labels/quality。清单先落盘，不能只挑产生证书的错误例。

A Qwen语义问答及冻结B/A → B Llama native独立候选与control pool → C Qwen盲于native效应的候选条件核对 → D Llama固定组位置/来源介导/pairedR/MLP交互/层带/模板 → CPU完整回答输出。所有失败保存并计入覆盖。每Claim≤256、每回答≤1024真实模型forward，donor也计数；不是全量部署成本承诺。首次自然batch测真实时延/峰值显存，不事先宣称若干GPU小时。

最低输出：每原回答全词score/abstain、每Claim C/N/U、对比来源和完整input IDs、native/reader候选分流、每干预F及逐tokenlogp、位置/来源/跨角色/MLP及失败状态，原始输入/代码/模型/模板哈希。不得人工输入错误位置或12/14候选。GPU只在代码审查和必要CPU检查通过后独占运行；原population需原协议安全恢复。

B1的通过仅代表完整链条可运行且分母正确；不能以Qwen标签为独立事实真值。检测准确率另按RAGTruth实际span标签评价。

## B2：覆盖、约束归属与连续区间（MUST，B1之后）

在任何自然标签用于指标之前冻结方法/模板/阈值/来源清单。分别报告语义问题覆盖、C/N对比覆盖、native位置证书覆盖、输入介导覆盖、pairedR和MLP交互覆盖、弃权原因。门槛依FINAL_PROPOSAL：80%断言词语义覆盖、70%风险Claim可对比、50%风险Claim有至少一种完整内部选择性证书；各任务/C/N不可互相掩盖。

真实检测指标对RAGTruth原始标签计算，semantic-only为同一个reader强基线；AUROC/AUPRC、固定阈值precision/recall、完整错误span及恢复边界，按source聚合不确定性。RAGTruth没有唯一回看节点或真实因果影响终点标签，不报告伪造的“节点准确率”。延续/纠正关系若缺人工核对只报告模型估计分布与实测依赖，不能伪造gold。

若位置有效但输入介导失败，停在position effect；若MLP非槽变化主导，记broad event；如果总体证书覆盖不足，不把少量成功案例写成完整方法收敛。按最大的实际失败环节修订，而不扩已知关系实验。

## B3：主线速度与图读出必要性（条件MUST）

只有B1/B2给出可靠、覆盖足够的教师信号，才考虑一个预测实测Delta/Omega的轻量图读出蒸馏。对比同节点属性MLP/集合模型，证明拓扑对应有增量；训练不使用自然幻觉标签。若教师效应不区分，就不训练GNN来掩盖缺少信号。

最终主线需同时报告全量吞吐和质量，不能把每Claim256次机制教师当成可直接大规模部署的检测器。原始机制8条件全量与新方法结果分目录。

## 当前执行状态

- P2原冻结17790机制任务：运行中，实时progress.json为准，保持冻结。
- 新有限事件/E-V/pairedR内核与真实Llama重放后端：实现，17项CPU检查通过，独立复审进行中。
- A–D自动编排：尚待实现；不提供虚构可执行的新方法全量命令。
- B1新自然GPU任务：未启动。B2/B3未启动。
- 科学结论claim_supported=no；方法规格ready_to_implement；Codex MCP实验审计REVIEW_UNAVAILABLE。
