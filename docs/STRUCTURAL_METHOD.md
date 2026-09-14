# S10：结构与熵的组合

读取当前 token 的预测前测量：熵 H、negative margin、来源注意力 S、远历史注意力 R，以及先逐层求跨头标准差再平均的来源注意力分歧。远历史 R **不包含最近16个 token**；D=R-S 不等同旧结果的总历史位移。

最近8步的熵记忆 M[t]=max(H[j] exp(-(t-j)/4))，j 从 max(0,t-8) 到 t-1；首token设0。完整输入为 H、margin、S、R、跨头分歧、D、H差分、D差分、M、H×D、M×D。所有特征只读取 t 及之前；不使用真实首错、未来文本、回答总长或相对位置。第一个token即可评分。

四个固定读出：instant（前6项）、combined（全11项）、without_entropy、without_attention。另报告原始熵、原始margin、原始远历史位移和绝对位置对照。两套目标分别为RAGTruth原错误token标签、每个标注span首token；没有假造关键实体或完整语义单元标签。

每套固定L2逻辑回归C=1，训练按source等权。校准集拟合正斜率缩放和截距、固定正常token 5%FPR阈值，测试不选择参数。完整模型/预测冻结后才读取测试标签。来源级bootstrap比较combined与瞬时组合、各单项，报告回答内AUROC及正常回答误报。

数据：QA、原回答模型llama-2-7b-chat，839 official-train回答与150 official-test回答。official-train来源按固定SHA排序前80%训练、后20%校准，test来源互斥。特征是现有Llama3.1 observer缓存，0次新增GPUforward。官方test曾用于历史描述性分析；本次为固定新方法的来源留出评价，不称完全未见测试。

实现是在真实内部信息上组合轻量统计与时序状态；没有先验保证比单一位移更强，最终以未调参的留出指标为准。
