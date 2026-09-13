# GraphAdapter训练目标原文复核（2026-09-13）

来源：[Can GNN be Good Adapter for LLMs? §4.2](https://arxiv.org/html/2402.12984#S4.SS2)，WWW2024，已读完整§4.2式8–14及§4.3接口；不能把文中节点分类效果移作RAGTruth幻觉检测证据。

关键细节：GNN处理节点句向量，MLP融合图表示与冻结LM预测前状态，经同一个LM head输出；式11–13对**概率**取均值后做token CE，并非hidden residual或logit均值。此分支设计旨在让优化更多关注LM本身难预测的token。冻结transformer状态可以缓存。

我们的推论（不是论文结论）：当前A2是门控hidden residual、直接adapter CE，不等价于原文概率混合。若自然检测显示只适配来源格式，后续可以检验明确的目标改动；不能因论文用了图而推断我们的分数有效。概率混合CE对adapter目标log-prob的梯度幅度是 p_adapter/(p_base+p_adapter)，因此它会相对降低base已高置信且adapter尚未学会的token的权重。这是可分析的优化差异，不是幻觉定位证明。

本轮训练已冻结，不据尚未出现的自然结果修改协议、分数方向或epoch选择。下一步先读真实graph/no_edges、base NLL、固定差分及payload擦除结果，再决定问题来自来源依赖、域迁移还是检测统计量，避免又增加一个无关模块。
