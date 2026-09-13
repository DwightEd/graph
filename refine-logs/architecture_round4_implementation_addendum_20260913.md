# 修订4的单卡执行补充

2026-09-13。此补充替代修订4 §10 的旧两阶段执行安排，须并入最终完整方案。

1. A/Qwen：仅从task/response抽取Claim/关系问题；问题自洽；隐藏原答案的全文来源QA；条件核对、C/N/E/U和冻结对比B/A。缓存语义参考证据，但native proposer不接收它。回答抽取不得由RAGTruth labels/quality选位置。
2. B/Llama：只接收原生输入、完整冻结B/A与source/history角色覆盖区间。执行candidate-blind搜索；冻结最多3个候选及一个有长度/质量/可达性匹配的control pool（每候选最多16个，按输入hash裁定）。保存未发现/预算状态，不删除失败Claim。pool此时没有reader语义标签。
3. C/Qwen：给已冻结候选/pool及需要测试的来源输入单元做条件适用性、矛盾/无关与引文核对。它接收问题、来源、候选原文位置，**不接收F、Δ、native排序、输出mechanism标签或测试标签**；输入顺序按原文位置重排。候选id保持映射。若pool中合格控制不足，D标matched_control_unavailable，不回B找容易成功的候选。
4. D/Llama：对B冻结候选、C筛出的hash前两合格控制执行位置/介导/pairedR/MLP/层带/N模板验证。D的任何失败不回填B候选。每类预算包括缓存未命中的baseline/donor；无结果不记0效应。
5. CPU：合并图与覆盖；分开semantic-only、position、origin、跨角色、MLP、continuous/ correction、unresolved。所有中间文件包含输入/模型/模板/代码hash，阶段D核对B/A及候选未变。

单4090分阶段处理整批回答，每阶段只有一个模型驻留，阶段间明确释放/重载；不按单条回答来回装卸两个8B。现有population占用GPU时不能并发加载新模型，之后运行需独占调度并恢复原冻结进程。

Native后端的donor必须与每条B/A recipient使用相同序列长度、dtype、attention实现。η=1 donor与recipient的selected V逐值相同是升级介导的前置条件；若不同则measurement_invalid，不用容差把sham漂移掩盖。

当前代码：causal_contrast.py与native_audit.py已经实现计算及重放算子；没有实现上述阶段A–D的自动数据编排，不能发“新方法全量可跑”的命令。
