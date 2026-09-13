2026-09-13后续更新：SourceRel-Mini全量训练/自然迁移均已完成，但未通过自然归属验证，详见 [完整结果](SOURCEREL_RESULTS_20260913.md)。以下“尚未训练/待完整训练”描述是先前结构草案；当前不继续此单向量头。新的完整source inventory在 `next_iteration/constraint_inventory.py`，仅修坐标/包含关系/候选覆盖；automatic query scorer仍在细化，主线尚未实证收敛。

# 当前模型结构与实现边界（2026-09-13）

当前没有已经验证有效的完整主线。旧surface owner在36回答上合格B/A=0；新的typed-hours已经跑完17790行CPU编译，但6198条Data2txt只得到2个B/A，其中train1条。该train窗口完成90次原生前向，自动定位到4query×16层的联合影响组，强证书仍未通过。不能把这些结果写成准确自动回看或归属已解决。

主线在graph，reanchor保留全量observer机制任务。以下结构区分已经可执行的模块与仍须实现/验证的模块。

## 1. 样本图不是截断窗口

底层是完整prompt/response的原始token坐标和原生隐藏状态h[layer,token]。来源节点保留每次实际出现的位置、字段/事件、记录身份、时间、否定和量词；同值不同记录不能合并。回答事实用可重叠、非连续的原文成员集合表示，主体可以被多个事实共享。

因此显示的span只用于定位；输入仍完整、跨span长边仍保留。同主体、相邻或同一句不自动导致风险传播。`next_iteration/fact_scope.py`、`fact_scope_mask.py`已实现三值约束组合、共享变量和精确坐标遮蔽；它们不会自行生成语义真值。

采用原文成员集合而非必须独立改写的句子，参考[PropSegmEnt](https://aclanthology.org/2023.findings-acl.565.pdf)的重叠命题表示；事实语境保留参考[Claimify](https://arxiv.org/html/2502.10855v2)的选择、消歧和分解。二者不提供我们需要的100%自动位置保证。

## 2. 约束归属：SourceRel-Mini，尚未训练

当前最大缺口是回答关系与来源事件的对齐。计划用轻量的来源关系编码器代替自由生成答案或单个有限标签硬门：

- query：原文目标span被遮蔽的完整parent及此前必要上下文；不依赖自由SRL先给出正确角色。
- candidate：来源occurrence及其父记录/事件/条件的高维表示；不输入观察到的错误答案作为检索线索。
- score：query/candidate各4096→128投影、broad-type逐维缩放后的归一化余弦匹配；图中的明确same-record/event/condition关系作为固定一致性因子。允许同一来源被多个回答span复用。
- 训练：仅从来源本身的关系重建生成目标，owner InfoNCE配合同值不同owner、同实体不同条件等负例；只有真正遮住待预测边时才追加关系重建损失。没有RAGTruth幻觉标签、原生干预效果或Qwen真假标签作为训练目标。
- 输出：owner候选集合、角色提议与未知状态。它不直接输出真假；NULL也不等于来源全文未说明。

遮蔽依据坐标与同一值的依赖关系，不全局删除相同字符串。例如另一个独立日期中的数字不因恰好等于目标数值就被删掉。来源位置/ID只承担监督和产物定位，不能作为模型输入的绝对位置捷径。

来源图已经显式给出的same-record边不能再作为直接可见的预测目标，避免训练变成关系查表。先用固定图因子，图消息网络仅在来源留出验证显示必要增益后增加。模型设计与反例边界见 `refine-logs/source_relation_encoder_design_refinement_20260913.md`。数据/特征/训练CLI已实现，883来源的720/163划分和全量长度预检已完成；真实两来源sanity已有checkpoint，仅作接口检查。完整小头训练及自然迁移尚未完成。

图关系的共享变量借鉴[GraphFC](https://arxiv.org/html/2503.07282v1)，但不采用贪心绑定后永久替换；保留全部一致赋值和冲突。文本高维表示也不能仅凭“来自LLM”就当作检索空间：[LLM2Vec](https://arxiv.org/html/2404.05961v2)和[Qwen3Embedding](https://arxiv.org/html/2506.05176v3)都使用了专门的表征训练目标，检索相关性仍不等于约束成立。

## 3. 关系核验与真实B/A

归属与真假分开。在适用owner确定后，检查当前关系的所有条件，再比较目标值。已实现的typed-hours根据七天字段、量词和同句WiFi核验单个端点；其它自然语义仍未知。只构造精确原文单值修改，不把源文本自由生成的新句子冒充原决策。

真实9273：原回答6 PM，来源七天均16:0，支持替换4 PM；所有非目标字符保持。编译结果保留原token前缀、完整句末后缀、目标mask与七天来源key。完整CPU覆盖结果见 `TYPED_HOURS_RESULTS_20260913.md`。这条窄语法覆盖不足，后续不以不停增加营业时间模板代替通用归属模型。

## 4. 原生定位与传递

已经可执行：`next_iteration.typed_native_runner`。固定F=logP(B完整后缀)−logP(A完整后缀)，从所有共享prefix query、所有层开始，自动搜索层/query子集并保存联合父组、非邻接query组合和未搜索前沿。它是来源A条件下的定位，不是从内部图独立发现正确owner。

节点高维特征与拓扑都实际落盘：原始B baseline的32层×143token×4096维bf16状态，以及固定来源节点、query坐标、实际测量的层/query超边。不能把组效应拆成虚构的单边因果分数。

随后测两类不同干预：A消息content衰减，以及A原输入变化仅经同一A的V消息地址进入原recipient。保留full/half/sham/repeat、固定来源控制、目标slot/上下文概率分解和补集。大效应本身不升级为采纳；缺控制、剂量反向或未搜索部分都保留。

实际自然窗口结果：observer原本偏A（F=-9.01194）；自动组query707–710、layers16–31，移除消息Δ=-12.86270，原输入→V的全移除Δ=-15.25483，但half Δ=+0.23241、选定范围controls不足。故只报告条件性raw影响，不能称幻觉修复、稳定采纳、唯一点或原Llama2生成原因。详见 `TYPED_NATIVE_RESULTS_20260913.md`。

## 5. 路由、聚合与连续span仍未闭合

错路由需独立确定的错误B来源，固定同一层/query及B移出质量，比较B→A和B→无关控制。当前typed窗口无这样的B，不能把其它日期、WiFi或历史数字凑作错误owner。

正确A信息到达仍不能自动证明“聚合出错”。需要独立操纵后续组合、重现修复，并排除无关控制；已有MLP交互最多是调制证据。本次origin半强度不稳定，不能绕过它发更强结论。

连续错误范围应沿具体值/关系的复用边定位：较早错误值是否被当前事实采用，是否产生有符号的history→当前F效应。纠正、仅引用和独立新话题须区分；未发现边不等于影响已终止。当前图数据结构可表示这些边，可靠的自动关系供给和自然传播结果仍没有完成。

下一步方法工作是SourceRel-Mini的数据视图/模型实现和来源留出验证；原生部分优先修复源干预的语义与稳定性定义，而不是继续在相同例子增加任意门或降低阈值。旧RAGTruth全量机制测量已完整完成17790/17790，failed0，不能代替新方法效果评价。


2026-09-13 元数据追加更正：旧 `constraint_graph.owner` 的 root-owner ID 实为 business-name scalar anchor。真实记录根是 `literal_graph.root`；370个来源的对应更正见 `outputs/typed_hours_owner_metadata_correction_20260913.json`，独立影响审查 `refine-logs/typed_owner_metadata_correction_review_20260913.md`。仅 owner 图元数据命名有误，全部七天来源值、B/A原文、对齐和native数值保持；旧冻结代码/产物未改，不需要重跑GPU。
