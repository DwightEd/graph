# 当前方法与结果入口

A2-v1 GroundedGraphAdapter 已完成，固定差分全词AUROC0.505927、首错后0.512073，均未超过base NLL。当前主线继续迭代A2-v2来源恢复模型，尚未收敛有效检测器。

图节点为完整来源的field/context/component/bundle，4096维原prompt隐藏状态，两步128维关系消息传递，9节点种类/4边种类。v1用H_full[t−1]查询图并加门控残差，source-copy CE与坐标指针NLL联合训练；实际擦除诊断显示指针敏感但生成未受益，详细[完整结果](GROUNDED_GRAPH_V1_RESULTS_20260913.md)。

v2的单项改动：再做真实observer前向，把prompt所有source重叠token替换space220，缓存H_empty[t−1]作为query与解码残差基底。图X仍来自原完整source，原H_full也保留用于baseline。相同240来源/192训练48验证、9220锚点、135519tokens、1,641,089参数、10epoch两臂。新图分配仍为适配器的推断，不是原LLM实际路由。详见[设计](GROUNDED_GRAPH_RESTORATION_V2_20260913.md)、[逐条运行命令](GROUNDED_GRAPH_RESTORATION_V2_RUN_20260913.md)。

新版代码已实现；feature与train/predict/eval工程C0/R0，真实capture正在启动，尚无v2结果。固定两种风险分别为logp_full−logp_restore和logp_empty−logp_restore。所选模型必须在48留出来源坐标锚点相对empty LM有正增益且固定query真实source-X擦除使增益下降。自然36答仍为复用开发集，两个风险、全词/首错后和source重合全部报告；官方2700test未跑本模型。

全量RAGTruth17790机制与postfirst已完成，结果弱；完整来源库存2965来源已完成并审计。准确自动回看定位、自然应采纳约束归属、错误路由与正确到达后聚合失效、连续错误影响范围仍未闭合。

当前拓扑覆盖source内部包含/成员关系以及逐token query到source节点的软分配；尚未实现回答span之间的传播边、自动回看定位或图自适应分段。使用Llama3.1 observer重放六种模型的给定回答，不能把其状态称为那六个生成器的原生内部轨迹。
