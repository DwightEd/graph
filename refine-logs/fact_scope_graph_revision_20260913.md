## Accepted implementation update — 2026-09-13

The full-vocabulary CPU diagnostic has completed:10 requests/30 forwards, all valid letter+EOS, high valid-label mass; semantic-reader failure persists. Choose typed Data2txt source algebra as the next provider and keep QA/Summary unresolved, as reviewed in semantic_provider_decision_20260913.md. Do not launch another identical Qwen finite batch or claim that a format fix solved semantics.

Implemented fact_scope/fact_scope_mask have23 independently passing CPU checks and C0/R0 for supplied-assumption graph solving. Scope tables must match exactly the target fact; unknown/unverified constraints never eliminate an owner. Target masking uses coordinates/roles, not global bans on strings such as 1/on. A caller-supplied table is not a provider certificate.

Typed-hours implementation binds all7source day constraints, exact single endpoint edit and preserved WiFi to original observer token coordinates. The full17790 label-free CPU prepare is next, with a predeclared train-only native selection: fixed hash, first per source, at most12 sources. No effect-dependent replacement. Full population denominators and unresolved cases remain in the parent.

Native method/code is now next_iteration/typed_native_measure.py +typed_native_runner.py. It reserves layer/query search budgets, preserves all-layer joint parents and nonadjacent query unions, and outputs the smallest tested A-directed group or a clearly named raw fallback. This is evidence-conditioned localization, not internal owner discovery. Controls are selected from source schema at baseline before search; only those IDs may be rechecked in the selected scope. It records raw A-position and raw A-input→sameA-V messages even with unavailable controls, but cannot certify a selective contribution then. Target high-dimensional nodes are actual baseline h[layer,token] bf16 values captured without another forward. Topology joins these coordinates to measured query/layer hyperedges; a group effect is not fabricated per-edge attribution.

No source-B occurrence is fabricated, no default history competitor is substituted, and no aggregation verdict is inferred. F baseline already preferring A remains in the roster and is never described as error repair. Detailed scope: typed_native_design_review_20260913.md. The following text preserves the method derivation; references to the pending readout diagnostic are superseded by this update.

---

# 事实约束图修订：先使“归属问题”成立，再测原生采用

2026-09-13。状态：架构修订，未验证新效果。surface_owner_v1完整负结果已保留；以下不改变其阈值或解释。当前独立CPU读出诊断与GPU population并行。

## 问题锚点

对冻结LLM，在不以幻觉标签训练的条件下，测量来源和历史的信息实际传递、聚合与输出采纳；区分疑似错误被继续沿用、被反对/纠正以及新陈述开始。事实区间和影响范围分别评价，熵只提供候选起点。

必须解决：自动回看定位、当前证据条件归属、错路由与正确信息到达后未采纳的区分、连续错误范围。主线graph，reanchor保留原全量机制任务。单4090，已部署Qwen3-8B/Llama3.1-8B，无幻觉标签训练。原始生成数据来自6个模型，Llama重放只代表observer，不冒称原模型内部。

## 真实失败及取舍

最新36回答：1619词面槽，44过target C+N门，145候选，合格对照0。非互斥失败：完整edited Base124，保留125，value-only145，归属51。首拒绝124只是门顺序，不能据此诊断124都是多事实错误。独立实现复核C0/R0，Qwen首token标签语义判断失效；正误对照缺失发生在native之前。

停止扩展任意content_run→content_run替换。日期、谓词、实体、关系条件必须通过当前关系问题确定可比性；仅“词面类型相同”不能承担这个工作。停止用整个parent sentence的单个末位隐藏状态余弦、任意固定加权来解释owner识别成功。停止把纯标签格式检查当作语义校准。也不回到未验证的自由SRL结果作为全部候选硬前置。

主架构仍是一个图上的条件推断与干预算法，不新增GNN/Adapter。图真正解决的是同一指代的约束一致性、一个事实跨越多个片段、不同事实共享片段，以及有符号的沿用边；不是给无效词面候选再加邻接矩阵。

## 三类节点和两类边，避免一个“图”混淆两种证据

1. 原文节点：完整prompt与response的token/字符坐标、真实残差向量h[l,t]、V/O消息引用。底层保留完整上下文和跨span边，不因分组删除输入。
2. 事实因子：一个可独立核验的关系，连接原文token集合M_f、关系/事件变量、主体与对象变量、时间/地点/条件/否定/量词约束。M_f允许重叠且非连续；渲染的自然语言释义只供语义解释，不能替代native原文。
3. 来源occurrence：保留文本中实际出现位置、字段path、记录身份、事件和约束；同值不同记录不合并，同实体不同事件不合并。未知/null与明确负断言分开。

语义边是“该token参与该事实”“这些提及引用同一变量”“该来源满足该事实的哪些约束”；均保存未验证/未知状态。原生计算边是(layer,head,q,k)消息以及干预测得的方向/幅度。二者用原文坐标相连，语义相似不升级为因果采用。

事实窗口不是词序区间的互斥切块。显示窗口可用M_f的最小包络；评分只指向其具体target members。主体可以同时属于营业时间和WiFi两项事实，这不使两事实自动共用真假或错误传播边。

## 适用性：完整关系查询与共享变量求解

对每个待核验事实f，构造关系问题Q_f，遮住被质疑的**完整语义答案**；保留主体、关系、单位、时间、条件、否定及量词。若否定/条件本身是目标，其变化属于该事实的“关系型对照”，不能混入value-only证书。可有多个互斥解释，保留集合或unknown，不猜一个。

问题忠实性用原回答和其此前上下文检查：回答是否确实表达Q_f及所称答案。不能把source推断反写为response原有含义。来源匹配只看Q_f及非目标约束，不看所称错误值或原生干预效应；完整来源仍在最终判定中可见。

归属变量z_v取来源实体/事件/记录候选的集合。每条已验证关系对其涉及变量施加允许组合表，而不是各词独立选top1。关联关系的组合通过同一个z_v自然连接。推断输出全部一致赋值、各变量剩余候选、约束冲突、未验证约束及未搜索尾部。关系图只在同一变量上要求一致，不强制不同变量一一对应来源，允许别名/重复表述。

为判断当前目标的适用owner，**留出当前被质疑关系的值约束**，仅利用指代结构和其它已核验前提。若其它前提也互相冲突，则输出constraint_conflict，不能把目标值拉向“最像它”的来源消解冲突。若仍有多个owner，输出owner_ambiguous，不能将错误值作为tie-breaker。精确枚举超预算输出search_incomplete；不能在找出一个可行赋值后称唯一。

这部分可用小型约束求解器实现。它使共享关系约束真正参与归属；不假装能凭拓扑自行判断哪个语义关系为真。高维特征只提议candidate，不形成允许组合的事实真值。当前Qwen首token接口不可靠，新的关系读出必须先通过真实请求诊断；完整自然答案/证据引文与可检查结构是首选替代，具体输出协议在诊断后冻结。

## 自动回看：先定义被改变的完整决策，再定位计算节点

D_f = logP(原错误承诺完整后缀 | 原prefix) - logP(同一关系下的支持后缀 | 原prefix)。原输入和两后缀逐token精确对齐，所有原文非目标断言保留；分别记录目标项与上下文项。若只得到局部改进、其它独立断言仍错误，则只能叫focal contrast，不能叫整句正误对照。

仅value对照可发source-owner routing证书；条件、否定或整关系变化的对照另记关系敏感性，不能继承value型定位的解释。全文未说明且无支持值时，不硬造一个正确数值。

搜索域是两分支共享prefix的全部可见query与全部层。先保存层/跨层联合组的效应，再定位query组；不只查熵峰/标点。必须保存联合有效而单层无效的组，不能因32个单层均失败而判没有回看。结果是最小已搜索充分组及未搜索域，不承诺唯一点或100%召回。

具体source B→A路由证书只在一个固定层上比较B→正确A与B→两个不适用控制U，固定同样query和从B移走的逐head质量；单层使进入该层的QKV在三个反事实相同。联合层组可有条件依赖效应，但不能沿用单层的严格等移出质量解释。

## 传递、采用与聚合：逐层证据，保留无法区分的情况

用同一V接收地址A∪B，分别改变A/B原始输入，验证相反的完整D_f介导方向；再用上面的同移出质量路由比较，才叫条件性wrong-source-occurrence routing。仅有attention高或隐藏状态相似不够。

“正确信息到达但输出不采纳”需要正确A的原输入对接收消息有可重复介导效应，输出仍偏B，并检查后续模块对D_f的作用。现有evidence×MLP四状态交互只可叫MLP调制；不能自动叫聚合错误。必须能控制到达消息并独立改变后续聚合，在同一原生决策上复现修复，且对无关控制无效，才允许更强结论。当前没有这种自然证据，保留unresolved，不能在路由实验失败时默认归给MLP。

## 连续错误和自适应分组

独立计算每个事实的来源不支持状态；只有较早事实的具体值/关系被后来事实复用，且原生history干预有选择性地改变后来D_f，才建立有符号adopts_value_from边。更正/撤回、仅转述、独立新事实各自单列。

自适应组沿这种事实复用边形成；shared_entity、同主题、词面相似、相邻和同父句均不单独触发错误传播。图可跨中间无关句保留长边；显示连续span仍逐片段输出，不能把中间无关词染红。未找到边只叫未证实延续，不叫影响已结束。

## 收敛门及最小验证

1. 来源/原回答关系结构：真实36开发回答，完整覆盖与unknown分母；语义读出和图一致性先有效，才能发正误对照。人工逐案阅读只能诊断，不能作为自动准确率数据；开发集不得称未见确认集。
2. 原生决策图：有真实合格对照后再测自动节点组、source竞争/历史复用、后续模块；同读者非图、删除共享变量约束是必要比较。记录成功覆盖、选择性、实际前向成本；不能只报成功案例。
3. 冻结完整预测再在未用于设计的官方来源划分上加入RAGTruth词级标签，评价全词PR/AUROC、连续错误、覆盖与弃权。回看因果组没有人工真值，按干预恢复/删除稳定性报告，不能把熵峰当GT。

现阶段不能称收敛。下一项代码是事实关联表示/共享约束求解器和读出可观测性修复，均直接针对已观察到的失败；不再启动“关系信息影响输出”的例子。
