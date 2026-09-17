# 连续标签假设：理论、可识别结论与受控训练

## 本轮问题及范围

检验“长错误片段在训练中被重复计权，导致模型偏向段内续错”，并区分它与
“原LLM输入已经携带错误历史状态”。研究对象是原 CHARM / node_only，输入、
标签、图、来源划分、阈值校准原则都保留。不新增检测属性，不把标签作为模型输入。

只读取已有分数得到的结论属于观察研究。改变损失权重、重训原网络属于训练目标干预。
原LLM的事实支持性和错误历史的因果区分，需要独立的文本反事实；本轮没有该实验的自然数据。

## 命题1：逐点BCE无法直接学习标签排列

对已经固定的输入向量，node_only输出 p_i=f_theta(x_i)。给同一批数据、同一组参数：

    R(theta) = (1/N) sum_i BCE_alpha(f_theta(x_i), y_i)

任何共同置换 pi 都满足：

    sum_i BCE_alpha(f_theta(x_pi(i)), y_pi(i)) = N R(theta)

参数梯度也相同。证明只用有限求和的交换律；推理输出随输入同样置换。
前提是样本计算彼此独立且使用相同确定计算。测试使用原网络eval模式关闭dropout。
训练随机dropout若掩码随样本共同置换也相同；仅重置随机种子却改变掩码与样本的配对，
不保证逐次完全相同，但期望风险相同。分批顺序会影响有限步SGD；这不等于模型有标签转移状态。
对完整GNN，只置换节点而不同时置换边会改变输入，因此不能套用该检验。

结论：缓存(x,y)共同打乱，不能检验原LLM是否在x中留下了错误历史。
把y独立打乱则毁掉了类别关系，不是“只去掉连续性”的对照。

## 命题2：长片段会改变经验风险的权重，但频数不等于梯度贡献

正例损失可写成：

    sum_s sum_{i in S_s} ell_i = sum_s |S_s| * mean_{i in S_s}(ell_i)

因此原逐token损失赋予span的总系数与长度成比例。
对带样本权重w、正类权重alpha的单token logit z：

    d ell / dz = w * [(1-y)*sigmoid(z) + alpha*y*(sigmoid(z)-1)]
    grad_theta ell = (d ell/dz) * grad_theta z

长段可能有更多项，但容易的续错项可能已经饱和，且参数梯度可能抵消，
不能仅从“续错占96%”推出“96%训练梯度来自续错”。
新训练history保存normal/onset/continuation的精确未归一化|d ell/dz|和weighted loss，
不把它们冒称参数梯度范数。各batch仍按原token数归一化，跨batch的这些绝对和为诊断记录。

若假设同长m的簇内观测方差为sigma^2、等相关rho、簇间独立，则直接展开方差得到
Var(mean)=sigma^2/N * [1+(m-1)rho]。这只是示例假设；本数据不估rho，也不宣称得到梯度ESS。
span_population.json中的(sum m)^2/sum(m^2)仅描述span总权重不均衡，不是统计独立样本数。

## 命题3：再加权究竟改变什么

设 eta(x)=P(Y=1|X=x)，m_y(x)=E[w|X=x,Y=y]。总体带权BCE条件风险为：

    R_x(q) = -alpha*eta*m_1*log(q) -(1-eta)*m_0*log(1-q)

令导数为零得到（0<eta<1，权重正）：

    q_w(x) = alpha*eta*m_1 / [alpha*eta*m_1 + (1-eta)*m_0]
    logit q_w = logit eta + log alpha + log(m_1/m_0)

若m_1/m_0为常数，Bayes最优排序不变，只平移logit。
若span位置/长度相关权重使m_1/m_0依赖x，最优排序可以变化。
这个结论不保证有限容量、有限优化的网络实现Bayes最优。
所以再加权改善只能解释为训练目标干预的效应，不能单独宣称移除了所有统计依赖或找到了语义机制。

## 命题4：连续性先验和当前观测分开

一个示意二状态HMM有转移p01,p11；历史观测后验p_(t-1)，则预测先验：

    pi_t = p01*(1-p_(t-1)) + p11*p_(t-1)
    logit P(Y_t=1 | X_1:t) = log[p(X_t|Y_t=1)/p(X_t|Y_t=0)] + logit pi_t

第二式来自Bayes公式；需要HMM的条件独立假设。它说明连续性先验可以抬高分数，
但不等于当前node_only显式执行这个滤波。它只有f(x_t)，历史也许被原LLM编码进x_t。

如果一个退化评分只使用上一词真假一比特 s_t=g(y_(t-1))，固定上一词真假之后，
其条件AUROC必为0.5（所有分数相同）。本次条件AUROC明显更高，排除了这个极端解释。
它不排除更丰富的历史、词类、位置或前文语义所提供的预测信息。
代码中的previous_gold仅做分组；oracle_previous_label.json是不可部署的金标诊断，禁止当作基线成绩。

## 已有分数上的可重复分析

```bash
python -u -m experiments.charm_structure_audit.main --mode continuity \
  --continuity-stage observe --models node_only charm_in --bootstrap 2000
```

默认读取ROOT=outputs/charm_structure_audit_qa/QA/seed_0。
新目录audit_continuity_observe/，不读取checkpoint/图/embedding，也不重训。

- transitions.csv：上一词标签×当前词标签四格的数量、分数与报警；每答首词没有上一词，排除并保留总体分母。
- conditional_ranking.csv：给定上一词标签的pooled和同回答/source均衡AUROC。
- conditional_answers.csv：逐回答条件排名与分母。
- growth_summary.csv：固定50对窗口，后半减前半的真假logit间隔变化和配对AUROC变化；按source bootstrap。
- paired_growth.csv：每对原值，便于核对是不是少数长段主导。
- span_population.csv：仅合并重叠、不合并相邻标注。84个标注token span与79个连续错误游程不是同一单位。
- 两张PNG分别显示观察转移和配对前后分数；日志与表均明确是观察统计，不是随机化四格实验。

对数几率仅为显示稳定用clip[1e-7,1-1e-7]；分数排序和报警不裁剪、不改阈值。
来源区间为探索性，未多重校正。当前test反复使用，不能视为新的独立确认集。

## 核心新实验：四种损失，保持正例总权重

```bash
python -u -m experiments.charm_structure_audit.main --mode continuity \
  --continuity-stage train --models node_only --continuity-seeds 0 1 2
```

新目录audit_continuity_train/。默认4种权重×3个种子，共12个节点网络；不运行8B。
需要完整图比较时显式 --models charm_in，另给 --output。

原fit/select/calibration/test来源ID复用；本包配置是658/82/82/149，不能拿整个QA训练数量代替fit数量。
宽度、GNN轮数、batch size、learning rate、AdamW、warmup/cosine与原配方一致。
本轮统一使用预先固定的最终epoch，默认原配置50轮，不按pooled AP提前停止。
这避免选模型规则继续偏向大量续错。selection AP只记录，不用于本轮选checkpoint。
所以对照必须用本轮token重训，不能把旧早停checkpoint当作单因素基线。

令N+为FIT错误token总数、M为FIT合并重叠后的标注span数、m为span长度：

| scheme | 正例的w_i | 保持/改变 |
|---|---|---|
| token | 1 | 本轮重训基线 |
| span_equal | N+/(M*m) | 全FIT正例总权重保持N+，每span总权重相同；会改变各回答的正例权重 |
| onset_half | m/2给首词，m/[2(m-1)]给其余词 | 每span总权重保持m，首词获得一半；singleton仍为1 |
| random_onset_half | 将onset_half的同一权重列表在span内随机置换 | 权重多重集、每span总量、最大值都相同，只打乱强调位置；每epoch保持相同 |

所有正常词权重1，所有标签原样保留，alpha与token基线相同。
同seed共用模型初始化、数据顺序和优化步数；随机权重生成不消耗torch RNG。
GPU scatter可有底层非确定性；多seed报告，不能以完全逐位一致为自然训练保证。

只有单span的回答也纳入全局span_equal，因此它检验跨回答长span权重效应。
onset_half与random对照才更直接隔离“起点位置”与“把权重集中到一个词”的区别。
起点权重可能很大，本轮不按结果调整上限；同权重随机对照揭示由优化方差造成的变化。

每个模型单独在原calibration正常文本token上用同一目标FPR定阈值；test不调阈值。
输出fits.csv与paired_seed_deltas.csv，同时保存每个模型的总体/首错/后续起点/续错指标，
同回答、同配对、长度分层、正常间隙附近的转移观察和前后半间隔。
fit_weights.csv保存实际每span总权重及首词权重。history.json的诊断顺序为normal,onset,continuation。
选择/校准/测试标签都不参与FIT权重归一化。

已完成fit重跑跳过；中断fit从该模型起点重训，非epoch级resume。
只覆盖新实验目录。自动continuity_review.tar.gz仅收表格/说明/图，不含checkpoint。

## 判据（预先固定，不从test挑最好的解释）

1. span_equal对首错/续错区分差距的影响：检验长span总权重偏向。
2. onset_half相对token及random_onset_half：若只比token好但不比random好，不能把变化归因到首词位置。
3. 同seed分别报告AUROC与同校准FPR下召回；只报警多但排序不变可能是工作点变化。
4. 上述变化只是训练目标效应；不能证明当前token支持性，也不能证明生成LLM的self lock-in。
5. 前文错误/正确与当前支持性两因素需要独立文本反事实、原LLM回放和来源核验；本轮没有原LLM实验结果。

## 下一阶段的可识别因果量（设计，未冒称已运行）

固定当前候选句、长度、位置、前文模板。H=此前另一条独立事实是否错，C=当前候选是否被材料支持。
构造4个独立审查的世界，前文错误不能逻辑改变当前事实真假；数值/实体跨模板轮换。
以冻结检测器的原LLM重算输入评分z(H,C)：

    history_on_correct = z(1,correct)-z(0,correct)
    current_at_clean = z(0,error)-z(0,correct)
    current_at_wrong = z(1,error)-z(1,correct)
    interaction = current_at_wrong-current_at_clean

这才分别检验错误历史把当前正常内容推高、当前支持性效应及二者交互。
已有缓存x和分数无法生成这4个世界；置换缓存位置不是do(H)。
多模板来源均衡复验，观察QK、实际value写入与白盒确定的通道，不能只看prompt总量。

## 参考（理论推导与引用关系）

- PyTorch BCEWithLogitsLoss官方定义：
  https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
  用于确认实现的损失；上面的梯度、置换和带权Bayes公式在本文直接推导。
- Menon et al., Long-tail learning via logit adjustment, ICLR 2021：
  https://arxiv.org/abs/2007.07314
  为类别先验/损失调整的统计背景，不作为“连续span机制已被证明”的证据。
