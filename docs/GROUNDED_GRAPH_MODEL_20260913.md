> 版本状态：这是已完成且自然检测无增益的A2-v1结构说明，结果见 GROUNDED_GRAPH_V1_RESULTS_20260913.md。当前受控改动见 GROUNDED_GRAPH_RESTORATION_V2_20260913.md；以下旧执行状态为历史。

# 来源指针与条件文本重建的单一图适配器

## 问题与范围

对冻结LLM，在不以幻觉标签训练的条件下，测量来源和历史的信息实际传递、聚合与输出采纳；区分疑似错误被继续沿用、被反对/纠正以及新陈述开始。事实区间和影响范围分别评价，熵只提供候选起点。

当前先闭合“真实预测前状态是否可用于区分来源约束”这一建模缺口。SourceRel只学习辅助JSON视图的指针重建，不能据其95.4%声称自然归属成功。新模型直接读取原始observer的预测前状态，并增加来源条件下的token重建目标。Qwen回答侧真假判断只作外部参照，不参与训练、筛样或loss。

本方法是一个**可失败、可直接评价的条件预测模型**。指针分布不是正确owner真值，似然差不是因果效应。无需先得到严格B/A证书才能输出全词检测分数。自然RAGTruth错误标签仅在全部预测冻结后加入，直接评价分数是否有用；不能再只报告合成字段验证分数。

## 完整张量接口

原输入 `P || y` 在冻结 Llama3.1-8B 中teacher forcing，取最终RMSNorm后的 `H∈R^(T×4096)`。预测 `y_t` 只使用 `H[t-1]`；source节点的特征只池化prompt内来源token，因果mask使它们不含回答。原始prompt、任务问题、source边界和token化均保留。

来源图采用已审计的 `constraint_inventory_v1_20260913`。每个field/context/component/包含bundle有独立节点；不同出现位置不合并。节点初始特征为原始source token状态的精确跨度均值，bundle池化成员token的并集，避免重叠组件重复计权。可学习节点类型仅表示观察到的库存类别。图边只有实际包含/成员关系及反向边，不输入推测的正确owner或语义角色。

单一可训练组件 `GroundedGraphAdapter`：

```text
Z0 = LayerNorm(W_source X + node_type_embedding)       # N × 128
Z(k+1) = LayerNorm(Zk + mean_incoming SiLU(Wk Zj + edge_type))
q_t = W_query H[t-1]                                  # 128
u_tn = q_t · W_key Zn / sqrt(128)
A_t = softmax(u_t over all mapped source nodes)
c_t = Σ_n A_tn W_value Zn
g_t = sigmoid(w_gate · [q_t,c_t] + b_gate)
H_adapter[t] = H[t-1] + g_t W_out c_t                 # 4096
p_adapter = softmax(W_frozen_LM H_adapter[t])
```

两步稀疏消息传递使子字段→父记录→其它子字段可传递。不同样本之间不得有图边或attention。所有source节点都可作为候选，不按回答数字/日期的词面类型排除地址或长文本。source绝对ID不作为learned embedding；节点重排必须等变。`W_out`初始为零，初始输出等于冻结LM；pointer loss仍可学习query/source路径。主干、embedding与LM head均不训练。

图在这里实际改变节点向量与下一token预测，不是给检索分数贴图名。它自己的attention是预测分配，不能冒称原LLM采用路径。

## 无幻觉标签的训练数据与目标（当前原文重建版本）

先按来源原文SHA分内部train/validation，只用RAGTruth官方train来源，三个任务各64 train/16 validation，共240来源。全文SHA决定split，同源不跨split。

**Qwen弱模板版本已停止，不进入训练。** 实际只完成6/240来源，19/48模板机械可用，但发现确证的错误来源归属，例如把第3段的 Toss 绑定到第1段；SIGINT退出130，原始产物和失败分母全部保留。详细审计见 `refine-logs/grounded_graph_data_doc_witness_20260913.md`。JSON合规不提供语义监督保证。

当前替代是**来源坐标条件重建预训练**：保留原任务prompt，以完整来源原文作为teacher-forced重建文本。Data2txt用字段payload首词作坐标锚点，QA/Summary用已出现两个上下文词之后的稀疏原文词锚点；前8个全局词不作指针目标，每来源最多64个锚点。原文字符跨度与owner库存ID可精确重建，source位置不是自然回答的正确owner真值。完整原文token作CE监督；锚点只监督首个重叠token：

```text
L_gen = mean negative log p_adapter(verbatim source reconstruction token)
L_ptr = mean negative log A_t(source-coordinate owner)
L = L_gen + L_ptr
```

固定10 epochs、AdamW lr5e-4/weight_decay0.01/clip1、每batch8个来源，graph与no_edges使用相同初始化seed和样本顺序。按source-validation的token均值CE + pointer均值NLL选epoch，包含epoch0，严格改善才替换。没有自然回答训练、幻觉标签或label调方向。

这会面临来源复制向自然回答迁移、QA原问题与重建目标不一致的风险，不把它藏在字段准确率后面。完整图保留七天bundle，但本轮不生成all-days语义监督，不宣称多条件AND已学会。

**反复制对照预先要求**：在source-heldout重建中，对指针owner的source payload位置做实际输入擦除并重新编码，保持回答前缀与目标不变；报告原始/擦除后的目标token支持和pointer置信度。不能把上下文混合后的h向量置零冒充payload删除；即使对照有差异，也只说明此模型利用来源，仍需自然检测检验。

## 推理与预定检测分数

在所有真实回答token上计算，不要求语义reader先判错，不需要知道首错位置：

```text
s_t = log p_base(y_t | P,y_<t) - log p_adapter(y_t | P,y_<t,G)
```

较高 `s_t` 被预先定义为较高错误风险：原模型倾向生成而来源图条件模型降低支持。它是待检验的signed score，不是校准概率。固定同时输出原LM NLL、adapter NLL和差分，禁止看到RAGTruth结果后翻转方向。词分数为属于该词的token分数按原字符覆盖加权均值，未覆盖词保留显式不可用状态。所有任务、所有词、首错及之前和严格首错后分别评价。

单模型三个同输入对照：训练同维度但去消息边的版本；固定已训模型在原图/保持节点特征的边置换图上推理；节点ID重排等变检查。源payload遮蔽必须重新跑source encoder，不能把已混入上下文的native向量置零就声称语义内容被移除。这项是独立诊断，不是无成本假消融。

## 回看、路由、聚合与连续范围如何接入

模型先提供每一步的source分布及token分数。分布变化用于候选范围，不能把attention最大值称为回看点；真实回看仍由固定目标下的query/layer联合干预、补集与控制测量。指针A只是候选适用证据。原LLM消息中实际采用B必须另外测量。

对冻结、原文可重建的B/A，使用完整后缀 `F=logP(B)-logP(A)`，测来源原输入→V→目标介导。错误路由需固定质量的B→A与B→控制比较；正确A到达后的聚合失效需后续操作与选择性修复的额外证据。没有足够控制或剂量不稳仍为未决。外部语义预测产生的A必须标为条件假设，不能用它证明自己的语义准确率。

连续span首先保留全词风险轨迹，再利用具体来源/历史图候选区分事实组。相似owner、相邻句或高历史attention都不自动传播错误；历史→当前目标的有符号效应与reuse/correction解释分开记录。图分组改变聚合成员，不删除长距离上下文。当前适配器代码尚无已验证的自动分段或传播算子，不把源指针变化包装成该问题已解决。

## 最小可执行顺序与否定结果处理

1. 冻结240来源的原文重建和精确source-coordinate指针，完整核验输入及分母；失败Qwen模板不作训练数据。
2. 编译原始source/query高维特征，训练单适配器和同参数无边版本，epoch仅按内部source validation的固定联合loss选择。
3. 对旧36自然开发回答输出全部token/词分数，并在预测冻结后加入独立RAGTruth标签。主要看差分是否超过原NLL、是否超过无边版本，以及首错后的表现；仅合成loss改善不算成功。
4. 真正提供自然增量后冻结官方test推理；B/A机制与连续范围验证分别进入后续关口，不许用弱标签代替它们。

当前代码：`grounded_graph_adapter.py`、`grounded_graph_reconstruction.py`、`grounded_graph_features.py`、`grounded_graph_feature_runner.py` 已完成，相关16项CPU检查通过。数据/特征正在按 `GROUNDED_GRAPH_RECONSTRUCTION_RUN_20260913.md` 实际执行。`grounded_graph_train.py`、`grounded_graph_predict.py`、`grounded_graph_evaluate.py` 已写，训练/预测/评价尚未执行，接入审查继续。没有这版方法的自然检测效果结果，不能称收敛。

## 方法依据

[GraphAdapter §4.2](https://arxiv.org/html/2402.12984)将冻结LM的预测前表示与GNN表示融合，经LM head做next-token训练，使用LM/图分支概率平均；本文不把其节点分类结果移作幻觉检测证据。这里保留可检查来源指针，并直接比较条件预测分数；是否有用必须由真实回答检验。

[InPars](https://arxiv.org/html/2202.05144)与[Promptagator](https://arxiv.org/html/2209.11755)支持自然化弱查询适配，但不保证生成关系正确。与它们的retrieval目标不同，本模型还对原始预测前状态学习来源条件token重建。严格无监督事实真值或100%回看定位均未被这些方法提供。
