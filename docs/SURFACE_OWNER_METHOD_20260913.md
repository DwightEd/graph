# 当前方法修订：来源归属先于正误决策干预

状态：实现与验证进行中；不是已收敛的方法或已验证的机制结论。

## 问题锚点

需要解决的是：在真实正误决策窗口中，内部信息是否足以区分约束归属；自动定位承担回看/筛选的节点，区别错误路由与正确证据到达后的采纳失效，并沿事实复用关系定位后续连续幻觉。主线实现属于 graph；reanchor 保留机制实验与全量作业。不能用“关系影响输出”代替这些问题。

## 本轮删除的失败依赖

soft_graph_v1 的完整 A/B 上有426个claim、1704条来源边。旧 source-pointer bridge 只能构造10个未验证候选，还出现 `she doesnon misses...`、`the cafe 4.0...`、将一天营业时间代入“每天”等错误。原因不只是 JSON：合法坐标没有保证语义槽位完整，也没有保证来源归属。地址单元与不完整anchor还造成断句，bag风险被赋给整句词。

因此主动停止 soft v1：A36/B36/C27/D0，特征前向72，native干预0。全部部分产物保留，没有局部标签评价或新效果结论。独立见证确认原population恢复增长、13669份旧manifest与四份冻结元数据不变。详见 `refine-logs/soft_graph_v1_doc_witness_20260913.md`。

新版本不再要求Qwen先生成subject/predicate/object坐标。Qwen只对程序固定的片段与来源节点作有限选择判断。

## 图的具体表征

`surface_graph.py` 构造三类节点与关系：

- Base：完整原文句；明确引用接在前句，列表序号接后句。保守处理的歧义从句保持在同一上下文内。Base不是“只含一个事实”的假定。
- SurfaceSlot：数值、时间/数值范围、日期、年龄/时长、平衡引号或最多6词的保守内容片段。内容片段不冒充名词短语或语义角色；过长片段不强制截成固定窗口。数值范围、名称中的`&`、原文收缩词保持完整，引用序号不作事实值。
- SourceOccurrence：自然来源中的具体片段及完整父句，或Data2txt字面值、字段路径与实际父记录。相同值在不同记录仍是不同节点；None与布尔值有明确限制。

包含边、字段/记录归属边由原文坐标或安全AST解析确定。跨来源候选边由下面的匹配器提出，不能当事实支持边。跨回答事实复用/修正/引用/新话题关系仍需单独判断；图区域允许跳跃连接，不自动填满中间文本，不把一个错误值的风险广播到整句。

## 节点高维特征与归属匹配

对每个目标，将该值替换为固定类型占位符，以完整父句作为owner query。来源侧同样遮住所选occurrence并保留父句或字段记录上下文。这些字符串由原文确定，LLM不生成改写。

冻结 Llama-3.1-8B-Instruct，在第7/15/23层读取每份masked context末token的4096维post-block residual，得到3×4096表示。使用一致渲染、实际attention mask和position IDs，不静默截断。编码输入、token IDs、模型与代码身份、向量字节hash、实际前向数随产物保存。

这是辅助重编码特征，不是原回答生成时的内部轨迹。后面的native干预必须重新使用原始observer token IDs；两者不混淆。

`surface_owner.py` 首版无训练参数。只在类型相容的来源中计算固定代价：

`0.65 × 三层masked cosine distance + 0.25 × 非目标上下文词面距离 + 0.10 × 字段路径/owner词面距离`。

目标值的字面重合与目标值hidden state不进入owner分数。保留top4，但报告全部类型相容候选数、未搜索数、同值occurrence数和0.03代价内的所有近似并列。未知与无适用来源始终允许；没有一对一容量假定。固定分数是候选排序，既不是归属真值，也不是支持概率。

为什么这样设计：错误目标值可能把普通语义检索拉到同值的错误对象；遮住它直接切断这条捷径。来源父上下文保留“是谁的哪个属性”，避免只比较孤立数值。但若剩余上下文本身错误，mask并不能保证归属正确，必须经过下一层验证。

## 固定候选的有限验证

先对每个目标独立判断SCNU：只判断该目标在完整句中的事实命题，不因同句另一断言错误而把目标判错。完整source用于证据，earlier response只解析指代。这个结果是条件预测，不是词级真值。

只有目标C+N≥0.8、且程序可构造单槽位替换时，才继续其余验证。每个进入native的候选必须同时通过：

1. 原目标C+N≥0.8；
2. 改后完整Base的所有断言获得支持，S≥0.8；
3. 固定来源occurrence在其实际owner/条件下支持该替换，S≥0.8；不能偷换成其他记录中的同值；
4. 语法、非目标命题、单位、量词、指代及范围保持，P≥0.8；
5. 编辑仅改变同主体/事件的一个对象、数值、时间等事实值，V≥0.8；主体、谓词、否定、条件关系或多槽位改变不能进入此版本。

这五项仍是同一个冻结reader的预测，阈值不是校准准确率。每个问题及原始logit结果绑定不可变缓存、模型身份与候选；消费时再次验证。正确/未知目标跳过其余检查，但保留跳过分母。相同surface、引号、None、bool、长review文本等关闭状态分别报告。

`surface_verifier.finalize` 只在全部条件成立后构造B/A。保留完整前文，原分支到原Base末端，另一分支只改精确target span；直接计算canonical共同前缀和完整continuations，不再借用旧地址/标点单元向后扩大被验证的文本。真实tokenizer必须与原observer回放完全对齐，完整continuation超过96token则记录未解决，不能截断。

## 正误窗口与原生机制的接法

本轮GPU runner只验证候选供给与归属接口，native计数固定为0。它产出的合格对照是下一段原生机制程序的输入；不能把准备好的对照称为已定位回看节点。

原有native模块可复用的量是 `F = log P(原错误完整continuation) - log P(来源支持的完整continuation)`。它比只测观测文本logP更接近真实竞争决策，但它解释的是冻结observer，不是RAGTruth六种原生成器未保存的内部过程。

接入顺序已在 `refine-logs/source_pointer_native_bridge_spec_20260913.md` 规定：选定来源occurrence → 所有共享前缀query/所有层的组搜索 → 冻结位置与raw-origin控制池 → 有限语义筛除控制 → 固定V地址下的raw-input介导干预 → 同query/layer的source/history等质量路由对照 → source×MLP交互。控制对象和搜索预算必须先冻结，不能失败后补搜更容易的证据。

仍需严格区分：位置影响、来源信息经某V到达、抽象owner pointer、错误采纳，是不同层次。raw-origin→V因果路径不能单独证明abstract binding；MLP交互也不能单独证明“聚合出错”。不满足具体判据时应保留原因和测量，不能换一个机制名称宣告解决。

## 已执行的自然预检与当前可运行接口

`outputs/surface_owner_preflight_v2_20260913`：冻结36回答、6来源、4733词；237 Base、1619 SurfaceSlot，1618有类型相容来源top-k。词面预检749个目标owner近似并列。2377个唯一masked documents，总108701 observer tokens、最长812，均小于4096。

该预检只使用词面排序，reader/特征/native前向均0，没有标签。5799个可构造替换只是未验证候选，不能与旧10个不同粒度候选直接换算准确率或方法提升。v1预检与其源码快照保留；v2修复了实际发现的尾引用/下一条序号边界。

新GPU入口：`python -m next_iteration.surface_runner --preflight outputs/surface_owner_preflight_v2_20260913 --output <新目录> --stage prepare`。CPU准备阶段检查真实tokenizer、原observer对齐、单token有限标签、模型文件及执行快照；`--stage all`随后顺序执行features→finite，保存B/C产物。独占GPU时可用 `experiments/interleave_surface_owner.py`，复用已核验的population暂停/恢复事务。

尚未报告本轮GPU结果。当前未证明：dense owner优于词面排序、合格B/A有足够覆盖、准确自动回看定位、错误路由与错误聚合可区分、连续幻觉边界可恢复。后续以这些实际缺口推进，不再运行仅证明“关系影响输出”的对照。

## 相关方法细节与采用范围

阅读笔记已保存在refine-logs。下列启发与本方法设计推断分开记录：

- [Mixing Mechanisms](https://arxiv.org/html/2510.06182v2)：检索可混合位置、词面与reflexive策略；反事实答案不在原上下文中时的不同层patch可帮助区分pointer与payload。这里据此拒绝把一般因果影响直接称为绑定定位。
- [Language Models Use Lookbacks to Track Beliefs](https://arxiv.org/html/2505.14685v2)：binding lookback与answer lookback不同；其可学习子空间依赖构造任务监督，不能拿其层号或准确率替代自然RAGTruth定位验证。
- [GraphSeg](https://aclanthology.org/S16-2016.pdf)：先以完整句为节点，再用图关系形成区域。这里只采用完整句上下文与图关系分离的结构，不把主题相似或最小长度合并当事实复用。
- [Do Language Models Track Entities Across State Changes?](https://arxiv.org/html/2605.30233v1)：local/world状态读出不同，query条件化聚合需分别验证。其构造状态probe不能直接充当自然回答的归属真值。
