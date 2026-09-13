# 证据约束的消息依赖图：当前可执行结构

2026-09-13。主线 graph，机制库与原全量任务在 reanchor。当前是冻结模型、无幻觉标签训练的完整离线检测实现，入口 `python -m route_graph.soft_graph_runner`。尚无这一版的自然数据效果；不能称研究已收敛。v1/v2/v3 严格审计均没有实际native干预，v3全部4733词弃权。旧审计器 `route_graph.audit_runner` 保留为严格A/B证据层，不能再阻断所有检测分数。

## 节点与拓扑

来源、回答分别建立带原文字符/token坐标的图。Qwen只输出角色指针，程序重建引文；Data2txt安全解析原始Python字面量的字段路径、记录成员关系、值类型与None状态。抽取失败的词保留为raw-token或unparsed-leaf节点，不伪装成已确定的谓词事件。

冻结Llama的第7/15/23层（0-based）post-block residual按每个非空白字符等质量池化，节点为3×4096维。Unicode多token共同覆盖字符时均分质量。source在完整prompt且尚无response时捕获；response在完整原始token序列因果回放中捕获。当前节点只池化自身span，既不把后续状态汇入前文，也不截断其原有上下文。向量、映射、模型和代码都有哈希绑定。

边包括事件-角色、字段-记录、回答角色-竞争来源以及前文-当前span。候选不施加一对一source容量限制，同一事实可以被多处回答复用；同值但不同owner保留为不同节点。初始叶span由事件边界和标点形成，完整事件不会被标点拆断；经确认并通过原生对照的reuse边形成可跨远距离、可不连续的图区域。它不是一个截断context的滑窗，也尚不是已验证的真实错误边界。

## 来源竞争与关系判断

角色候选的固定unary cost为0.55 hidden距离+0.25事件上下文距离+0.10词面距离+0.05角色距离+0.05字段路径距离。事件内role pair共享来源事件/记录的关系cost以0.25加入；未知关系保留中性值，不由拓扑相似推断事实支持。每通道top4并集、beam32、保留8个assignment。温度0.2的assignment softmax只构造候选边际，非事实概率；按role/source/event唯一键去重，同一role总质量不超过1。

每claim取4条候选边做语义判断（不是每role四条）；剪枝、同值竞争与未搜索范围保持。局部证据不足只能U。完整source+当前span由冻结Qwen有限选项输出S/C/N/U；每条具体source occurrence输出S/C/I/U，I为其他owner或无关，局部I和U进入风险前合并U。None字段保留原文出处，强制U，不当false或全文缺失。历史边从所有前文span以冻结高维特征提出2个候选，输出reuse/correction/quote/new_topic/unknown；仅reuse概率至少0.8允许传播。

这些分布都是模型预测。没有隐藏来源QA答案生成，没有把reader输出当评价真值，也没有用RAGTruth标注学习权重。

## 信息是否实际传递

对每一个原回答span计算完整目标token序列的logP之和；保留原prompt和所有此前response，不输入后续response。native自动测试每claim固定2条source边、1条history边，预算80实际forward；层集合与特征层相同。每条边对全此前response query区间测真实V消息门控，并做最多6次新增forward的query二分，保留父组、不可达子区间与预算未搜索部分。此版本未穷举32层，不能把区间排名称为唯一回看节点准确率。

来源介导测量将某个原始输入span embedding缩放0/0.5/1，donor重新前向，捕获选定V；recipient仅替换这些消息，其他位置与后续计算重新执行。这样能区分key位置的影响和原始片段经这些V传入的影响。保存真实donor字节。要求原位sham、donor/recipient sham与未受影响前缀精确一致、重复误差不超过1e-5。

对照在看语义与native效果前冻结：同source kind；parsed role同role且不同事件；literal同值类型、同规范化字段schema、不同记录；history同抽取类别；长度和state norm均在0.5至2倍。再由有限reader判断无关，固定取前两个，不看效果补选。缺双对照时只报位置测量，不能提升图风险。

令delta=原始logP减干预logP；权重w=max(0,tanh((delta-max(0,双对照delta))/0.5))。负效应保留诊断，不自动把当前claim判更正确。另报within-role route敏感度；它不是“错误路由”证书。当前单事件干预不能区分正误选择，严格A/B与MLP交互仍由独立审计层负责；本批不声称完成该机制证明。

## 固定图推断与连续错误

Lglobal=log((qC+qN+1e-6)/(qS+1e-6))。
Lsource是source候选pi×pool_quality×w×(qS_local+qC_local)加权的局部冲突/支持logit，截断到[-2,2]并除以max(1,权重和)。
Lhistory只沿前向reuse边传播此前截断logit，乘当前native权重、reuse概率、具体slot/event映射质量和前条evidence coverage。纠正、引用、新话题不传播，也不把前条风险翻负。

L=Lglobal+Lsource+Lhistory；三项系数1。Unknown只通过evidence coverage把sigmoid(L)向0.5收缩。另报directional confidence；不能把coverage叫预测正确率或校准后概率。

连续错误预测是每span独立来源支持判断与有方向的reuse信息传播共同形成的风险序列；图区域给出可能的传播范围。错误边界真值只在最后用RAGTruth评价，RAGTruth不提供唯一内部节点真值。

## 必须同步比较的四个版本

1. qwen_fullsource_no_graph：同一reader完整来源判断。
2. qwen_matcher_no_native：相同候选关系和拓扑，不使用native调制。
3. graph_no_history：加入source原始输入介导测量。
4. full_graph：再加入有方向的history复用。

全词输出、失败和中性0.5均在分母。所有版本共享同一批reader产物。若full_graph没有增益，则关系/原生图检测增益主张失败。6个来源已经被前三版用于开发，不作独立泛化证据；参数不按本批幻觉标注调整。原生全span测量成本需实际记录，此版本是研究离线检测器，尚无低成本部署结论。

## 方法依据与尚缺证据

借鉴SRLScore的角色化事实比较、FENICE的原子claim与来源对齐、FGWEA的高维候选和关系局部代价，但保留source复用与unknown；不用其一对一KG对齐假设。HalluSpan的方法细节提示了角色mask和输入侧证据对齐，但其RAGTruth监督loss不进入本主线。原始论文阅读与具体差异见 refine-logs/unsupervised_alignment_structural_review_20260913.md、event_matcher_minimal_spec_review_20260913.md、detector_vs_certificate_design_review_20260913.md。

尚缺：自然数据有限reader准确率；候选owner召回；图相对同reader增益；node定位外部真值；错误路由与正确路由后聚合失效的严格A/B覆盖；连续错误范围与恢复边界的可靠性；独立来源泛化与运行成本。这些需要逐项实测，不能用代码完成代替。
