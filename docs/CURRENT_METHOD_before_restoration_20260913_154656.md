# 当前方法和证据入口

当前主线为单一 GroundedGraphAdapter：冻结Llama主干，来源完整图的4096维状态经过两步128维消息传递，查询使用真实预测前 h[t−1]，门控残差经冻结LM head产生条件token概率。训练联合来源坐标指针与原文token重建，固定检测分数为 logp_base − logp_adapter。

具体结构、训练协议、分数及未验证问题见 [GROUNDED_GRAPH_MODEL_20260913.md](GROUNDED_GRAPH_MODEL_20260913.md)，实际命令见 [GROUNDED_GRAPH_RECONSTRUCTION_RUN_20260913.md](GROUNDED_GRAPH_RECONSTRUCTION_RUN_20260913.md)。目前编码执行中；训练/自然评分代码已写、审查中，尚无新检测结果。

来源完整图已完成2965来源的全量构建和独立审计。旧SourceRel的95.4%是弱字段检索成绩，自然迁移失败；见 [SOURCEREL_RESULTS_20260913.md](SOURCEREL_RESULTS_20260913.md)。旧全量17790 RAGTruth机制与strict post-first已完成，连续错误信号弱。Qwen弱模板生成本轮已因确证错误owner提前停止并保留，未用于训练。

当前尚未解决：准确自动回看定位、自然适用约束真值、原LLM路由与聚合失效区分、连续错误影响范围。新图attention是适配器的分配预测，似然差不是因果证据；自然全词/首错后评价和source payload擦除对照仍需实际执行。
