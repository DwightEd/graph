# 离线证据输入与状态保持：实现与实验协议

32答正常高分审计及状态含义修正见 [DYNAMICS_STATE_IDENTIFIABILITY.md](DYNAMICS_STATE_IDENTIFIABILITY.md)。
旧H模式后验不能解释为幻觉概率；本轮没有发布新的预测型检测读出。

2026-09-23。状态：**已实现原生采集、无标签拟合、离线检测及评价；尚无新的自然数据 AUROC**。
本文件不替换已经可运行的 R/A/H 基线，不将已有基线成绩写成新方法成绩。

## 1. 本轮决定

- 用户明确允许后续 token：任务是离线逐 token 检测，可以观察完整回答。
  LLM 仍在原生因果掩码下 teacher forcing；检测器可以使用未来轨迹、FAI 和后向平滑。
- 实现包含逐头结构、原生多方向响应、输入协方差和后续复用的输入驱动状态模型；不以最简标量组合为目标。
  无自然幻觉标签训练，无外部真假教师。
  不预设还要接一个监督二分类器；模式后验本身提供候选排序。
- 必须区分读取、下游敏感度与语义支持。前两项有原生可观测量，第三项不能仅凭范数、符号或熵确认。
- 保留层、头、来源身份及有符号向量；首错、延续和标注 span 仅是评分后的评价分组。
- 先小规模采集/核对，复用原生前向与 teaching；不重新跑全量数据或删边消融。

目标假设：某些错误在新信息引入时缺少稳定来源约束，随后形成可持续的历史状态。
正确生成也会高熵、回看、多来源整合和历史复用，所以这些现象均不是单独的错误定义。
要求的是表示能够区分这些情况；不能保证自然首错总有高熵、所有回看都是 reanchor。

## 2. 论文核对及不能照搬的部分

核对 Information Flow 原文第 3--7 页、附录 H，及官方仓库
`263bec3c82a835707c124c17dc2d7c350fc522c1` 的
`proposed/Ours/llama/{llama.py,main.py}` 与 `proposed/calibrator.py`。

| 参考 | 实际方法 | 本设计的使用边界 |
|---|---|---|
| Information Flow Reveals When to Trust Language Models | 局部残差/多头向量的 L1 相容性生成贡献矩阵；反向提取 emergence order；跨层矩阵乘积得到路径布局；与外部相关性排名比较 | 排序是来源进入主要路径的顺序，不是真值顺序。FFN 在图节点内合并，不能由此判断其是否破坏语义。最终 XGBoost 校准使用标签，不能照搬为无监督结果 |
| Reasoning Fails Where Step Flow Breaks | 对每个目标 token 的损失计算 attention×gradient，再按步骤汇聚；干预阶段另用便宜代理 | 原生敏感度与便宜代理必须分别命名；不能用整句 loss 的一次梯度冒充各 token 的梯度；原论文取绝对值，不能由其反推支持方向 |
| Attention Illuminates LLM Reasoning | WAAD 描述回看距离，FAI 描述被后续位置调用 | 此任务允许使用 FAI，但未来被重复引用也可能是重复错误；距离与 FAI 均不能识别正确约束 |

这里借鉴“沿真实计算看来源的作用与顺序”。不会将排序、状态空间模型或局部导数本身称为新发明。

## 3. 对齐与观测对象

回答 token y_t 的原生预测 query 为 q=P+t-1。采集输入仍不包含 y_t，
但离线检测器可以读取 y_t、完整回答，以及其他原生 query 的观测。
每条缓存必须保留 response/source ID、query、target、token ID。

生成器与观察器目前不同：Llama-2-7B 的回答在 Llama-3.1-8B 中 teacher forcing。
观测描述的是观察器对该前缀的处理，不能直接宣称是生成器当初出错的过程。

三条轴必须分开：物理层 l、生成位置 t、来源 b。候选的来源单位为文档中的句子/数据字段，
保留到原始 token 的映射；不能根据金标错误边界切块，也不将整个 prompt 合成一个来源。
不同块数、长块和分词数量会影响熵，因此保留块内分布与块间分布，比较时控制块数/长度。

逐头原始消息仍是 m_(t,l,h,j)=W^O_(l,h)[a_(q,j,l,h) W^V_(l,h) x_normalized_(j,l)]。
attention、消息范数和 signed message 都保留，但不单独承担“信息注入”的定义。

## 4. 注入的操作定义：当前写入的下游响应

对当前 query 的某头、某来源块，将其实际消息记作 m。令 F 是从该写入位置到最终残差的
原生后续计算，则局部响应为 d=J_F m。它回答“沿实际消息方向略微改变写入，
下游状态怎样响应”，而不是“删掉该事实会怎样”。所有 source/head 向量保持独立。

状态传递使用共同固定的低维基 P：d_state=P J_F m。
不能每个 token 各自旋转 PCA 基后直接比较，也不能把不同 token 的 top-k 候选槽位当相同语义轴。
输出解释另外使用 d_choice=C_t J_F m；C_t 根据当下实际词及竞争词定义，显式随 t 变化。

实际采集器 `teaching/state_audit/src/state_audit/native_response.py` 直接对原生最终 hidden 和 logits
求 VJP，不用手写的 FFN 近似替换原生网络。对每个 query 保留一个图；默认 8 个固定 hidden 方向，
加实际词及 3 个原生竞争词的中心化 logits，共 12 个方向，`gradient_batch=4`。
过去 KV detach；同一 query 后续层的 Q/K/V、FFN、RMSNorm 及原生混合精度计算均在导数图内。
按 GPU 批量内积生成每层、头、来源块的方向响应，以及逐边响应能量；不是每边重跑模型。
不同层的导数对应不同写入位置，不能相加冒充一份互不重叠、守恒的残差贡献账本；
跨层拼接/投影用于学习统计结构，并不作这种归因声明。
P 来自固定 seed 的正交随机投影，方向数量是计算预算，不能声称覆盖完整残差语义空间。
此共同读出之后的头特征/来源输入压缩，才在独立训练来源拟合。

初版 choice 通道用观察词与 top 候选的中心化 logit 对比，而不只用一对 margin，
避免概率梯度在低熵处饱和。候选 budget、尾部概率质量均记录；它不是语义真值方向。
surprisal 和完整词表熵仍单独保存，用于解释与基线，不硬塞进风险加权和。

RMSNorm 的导数必须包含尺度随输入改变的项。对 F(x)=x+MLP(RMSNorm(x))：

    J_F v = v + J_MLP J_RMSNorm v
    MLP(z) = W_down [SiLU(W_gate z) * (W_up z)]

gate 与 up 两条支路都要求导。只冻结 gate 再计算一个线性映射，会漏掉 gate 的响应。
`teaching/state_audit/analysis/ffn_transport.py` 实现了 bias-free Llama 的这个局部 JVP，
测试分别与 autograd 和方向有限差分核对；此函数不等于完整 attention/跨层/跨 token 传递器。
目前是 float32/float64 的数学参考；不能声称逐位复现实际 BF16 模型的归一化累积与舍入。
接入原生混合精度模型时，需要另核对其 forward/JVP 误差。

FFN 的原始写入 f 与 m 反向，不能判为不支持。若它去掉无关维度、重新编码概念或旋转表示，
最终 d_choice 仍可能保留相关作用。应同时展示输入方向、经过 FFN 的方向及最终选择响应。
几何保持也不等于语义保持：错误来源可能一直强烈影响输出。

已有 `iter_native_traces` 的 detached KV 梯度只覆盖当前 query 的下游计算。
它不能追踪“过去来源→历史位置→当前 query”的完整导数，不能声称跨 token 因果归因。
全前缀参考导数只在小样本上计算；大范围的历史传递需要缓存各历史位置的状态并使用近似算子，
同时报告与小样本参考的误差。局部参考与完整参考不混称。

## 5. Reanchor：回看观测与状态刷新分别表示

保留每个 l,h,b 的绝对证据读取量 A_t(l,h,b)，距离按原始位置计算，特殊/模板 token 单列。
同时保留它相对过去窗口的变化 ΔA，不预先压成一个全层均值。

这可以同时表示：

1. 从 local 转向远处 prompt；
2. 始终有远距读取，但突然集中到另一个证据块；
3. 同一个证据块被再次集中读取；
4. 多个头并行负责不同来源。

不使用“距离必须增加”或“entropy 必须超过阈值”作为入口；绝对读取通道避免漏掉持续回看。
高读取只是“回看候选”。其后的 d_state、来源排序及历史位置复用决定是否观察到状态刷新。
所有 token 都评分，事件曲线只用于解释，事件检测错误不能导致后续 token 不被评分。

允许计算 FAI：当前位置对未来 query 的 attention 接收量。
记录未来可用 query 数及距离分组；回答结尾缺少未来观测是右删失，不是低 influence。
FAI 是后续使用量，不是成功 grounded 的证据；语法桥梁和错误内容也会反复被调用。
实现同时使用 FAI、未来 query 上的消息响应范数及后向递推。
FAI 与读取/FFN/不确定性进入联合结构向量，完整 emission 协方差建模其线性依赖，
不把它们的独立置信度相乘。它仍是近似统计模型，不能保证消除一切观测冗余。
预测行 t 的 query=P+t-1；回答词 j 的 key=P+j。严格未来 query 满足 t>j+1，
因此保存 N 个预测行时，回答最后两词的严格未来观测为空。缺测用掩码表示；不当作真实低影响。
原论文 Auto-Emergence 如做对照须使用其真实层间图，不能将本地 saliency 排名冒名为该算法。

## 6. 来源不确定性：分布与协方差，而非给 entropy 乘一个风险权重

对每层每头，把非零来源敏感度归一成 π_(h,b)。记录归一化前的总强度，
没有有效来源时标记 missing，不能把全零向量解释成 entropy=0 的确定来源。
由 attention 得到的读取分布与由敏感度得到的作用分布分别保存。

    H_within = Σ_h w_h H(π_h)
    JS_heads = H(Σ_h w_h π_h) - Σ_h w_h H(π_h)

前者是头内来源分散，后者是头间来源分工/分歧；JS 高不能自动叫冲突。
`source_moments` 的聚合仅用于诊断，逐头分布和向量不被丢弃。

对同一头的来源响应 d_b，在共同坐标中设强度 s_b=||d_b||，方向 ξ_b=d_b/s_b，
π_b=s_b/Σs_b。保留方向混合的矩：

    μ = Σ_b π_b ξ_b
    Σ_direction = Σ_b π_b (ξ_b-μ)(ξ_b-μ)^T
    实际净响应 = (Σs_b) μ = Σ_b d_b

不能用 π_b 再加权本身已带强度的 d_b，导致消息幅度被算两次。
若将方向混合作为输入不确定性的工作模型，按 (Σs_b)^2 缩放到输入单位。
该协方差描述不同来源解释的方向分歧，不是已知确定性消息之和的抽样误差。

这是一种“来源作为替代解释”的混合近似，不是已经证明的语义不确定度。
同样的高熵，如果来源指向相同方向，Σ_direction 可以为 0；指向相反方向时则较大。
并行互补证据未必是替代选项，因此先保留逐头向量，不把跨头 JS 直接注入噪声项。
实现对每个来源块先通过训练端固定的跨头线性投影，再计算块间方向协方差；
同一来源在不同头的响应共同进入这一投影，保留这些交叉项，而不是把头协方差独立相加。
协方差强度系数在无标签训练中估计；它是工作噪声模型，不是假定的真值不确定度。
选择空间的响应协方差用于解释对当前选择的分歧，固定状态空间的协方差用于跨 token 递推。
二者坐标不能互换。来源数量/分块改变引起的熵变化也不能当首错。

首错附近允许出现高来源熵、头间分歧或模式后验不确定，三者分别报告。
不设定“首错必然高熵”，也不将模型知道的金标首错用作 reset。

## 7. 已实现的联合结构—动力学模型

`dynamics_observations.py` 保留每层每头的读取质量、来源集中、分布变化、来源响应熵、
方向一致性/方差、实际词与竞争词响应、远距读取、未来复用；并保留 FFN 响应向量。
块内逐 token 的响应强度与块净响应分别保存，避免一个块内部的方向抵消被误读成从未读取。
源块熵同时保存原值和除以 log(块数) 的值；缺测有显式标志。块划分是提示文本的连续区间，
标点且至少 8 token 或达到 32 token 切块；不是模型发现的事实边界。
输出 entropy、surprisal、候选集合外概率质量作为额外观测，不直接加权成风险。

层、头轴展平后学习联合低维投影，不先求平均头。训练源等权、最多抽 8192 行拟合加权 SVD；
profile 默认 16 维，来源输入与历史向量默认各 8 维。测试来源只应用冻结的坐标、尺度。
保存训练解释方差及测试 `projection_error.json` 的标准化能量残差，不能按测试 AUROC 挑维度。

记 z_t 为共同 native hidden 坐标，v_t 为上述联合结构的投影，u_t 为所有头的来源响应投影。
h_t 包含 z_(t-1)，以及按每个头实际读取的过去位置汇聚、再跨头投影的历史状态。
位置对齐为：key j 对应已采集状态行 j-P+1；排除当前 query 自身，绝不偷用当前 z_t 作为历史。
FAI 属于 v_t，不放入 h_t 的因果历史项。

两个模式 E/H 共享同样的联合观测维度；H 固定 B_H=0，E 可用新证据输入。
这是统计模型约束，不是对 LLM 删除消息。联合条件 emission 为：

    o_t = concat(z_t, v_t)
    μ_(s,t) = concat(A_s h_t + B_s u_t + b_s, c_s)
    o_t | s_t ~ Normal(μ_(s,t), Q_s + ρ B_aug,s Σ_input,t B_aug,s^T)
    B_aug,s = concat_rows(B_s, zeros)
    Q_s = L_s L_s^T + 0.03 I
    p(s_t | s_(t-1)) = T[s_(t-1),s_t]

Q 是完整协方差，状态变化和头结构之间的协方差也被拟合；不是若干单特征概率独立相乘。
Σ_input 来自来源方向混合，ρ>0 由训练估计。来源不确定性因此既出现在结构分布中，
又改变输入更新的协方差。这个双重表示由联合模型学习，不能宣称已识别真假来源。

拟合在 `dynamics_fit.py`：先固定坐标；用参考集路由中位数初始化两个工作模式（无标签），
ridge 初始化 A/B；Adam 直接优化前向算法边缘似然与 A/B 的小幅 L2 正则。
按 source 等权、每答长度归一；完整异方差协方差参与求导。默认 60 轮，保留训练目标最佳参数，
保存逐轮目标、训练模式占比。T 和初始模式概率一起学习，无金标 reset，无最大 span 长度。
这替代早期文档中尚未实现的 EM 计划，不声称闭式训练，也不另训真假二分类器。

离线 γ_t(k)∝α_t(k)β_t(k)，主排序固定为 γ_t(H)。模式“历史主导”不等于真值标签；
两模式可能塌缩或无法区分，不能看测试 AUROC 翻转命名。正确历史复用与强错误证据都可能造成误判。
`state_forward` 仅去掉后向模式消息；它仍使用含 FAI 的离线观测，绝不是实时/前缀检测对照。

局部响应依赖本次原生激活，所以 u_t 是内生变量；条件拟合不是外生输入的因果识别。
跨 token 状态是统计近似，不等于完整 KV 的 Markov 状态。低维投影会丢信息，
Gaussian 也不保证能描述所有多峰语义。当前代码允许真实检验这些假设，未证明必然提高 AUROC。

## 8. 计算方案与成本

总成本 = 原生前向采集 + 敏感度/传递观测 + 小状态模型拟合 + 离线推断 + 评价。
只报告最后一项状态推断很快，会严重低估费用。

| 方案 | 成本判断 |
|---|---|
| 已缓存特征上的小 MLP | 通常最简单；本方案不承诺比它快 |
| 每个 token 多个完整 LLM VJP/JVP | 很可能更慢；预算与显存必须实测 |
| rank-r 局部 FFN JVP | 不构造 D×D Jacobian，但约增加 r 个方向的 FFN 矩阵运算，不是免费 |
| 固定低维观测上的 Gaussian/模式推断 | 对回答长度近似线性；小维度下廉价，离线后向增加一遍递推 |
| 原生导数到低秩代理的摊销 | 可能降低大规模评分成本；需另行训练和留出来源的导数误差验证，尚未实现 |

初始预算建议：先在既有具体样本中取少量完整回答，用 1--4 个 choice 方向核对原生敏感度；
固定状态方向的采集预算单独报告，不能把 choice 方向当作固定 latent 坐标。
reference 捕获精确 teacher-forced 单目标梯度；不得对整答 loss 求和后把受未来目标影响的梯度
当作当前 token 的梯度。即使离线允许未来，它们依然是两种不同观测，不能混名。

query 复制为独立 batch 可同时求多个单目标导数，但复制 KV 会增显存；当前 query chunk=8
只是前向分块大小，不能直接承诺 8 个目标梯度免费并行。先给出单卡 24GB 实测峰值再选 batch。
head/source 分组的梯度内积在 GPU 上批量完成，不做每边一次 backward，不输出完整 Jacobian。

## 9. 缓存与首次实验

已有 forward 缓存可复用 attention、逐边能量、单一 margin 投影、token 对齐和熵。
不能从它恢复通用消息方向、FFN Jacobian、完整来源传递或缺失的 prompt 行。
已有 native audit 的 margin_sensitivity 只能按其单方向、detached-KV 范围使用。

第一次自然试验先固定同一批回答和来源划分：

1. 复用原采集器，补采缺失的少量原生响应观测，记录总时间/峰值显存/覆盖率。
2. 保存 per-token/layer/head/source 的原始与敏感度分布、固定坐标向量及来源映射。
3. 训练来源拟合联合状态模型；全回答平滑，保存 forward/smoothed 两套分数。
4. 所有分数落盘后才读取 annotations。保持 R、A、entropy 及 position 同样本对照；
   `route_offline_mean` 使用前 7 行、当前行和后 8 行的均值，检验收益是否仅来自便宜的离线平滑。
5. 报 all_error/onset/first/continuation、AP、within-answer/source-balanced、前后半段。
   离线模式报告首错附近提前/滞后扩散和正常片段误报，不把后向平滑效果叫实时提前预警。

复用 32 答可以做开发检查，但它已被反复使用；冻结设计后另取未参与调参的来源确认。
不在本轮启动全量 GPU 重跑。旧四答的平滑优势不能作为新设计已经有效的依据。

## 10. 已实施及未实施

已实施：`main.py dynamics` 的 prepare/capture/fit/score/evaluate/run；
`native_response.py` 的逐query、多方向原生导数；逐头观测与离线复用；共同坐标拟合；
完整协方差状态模型训练；来源不相交检查；断点续跑；统一 AUROC/AP、同答/来源等权、首错/延续、前后半段；
同样本增益表与 HTML 曲线/层头热图。训练/评分使用数字 NPZ，不反序列化任意 pickle。
此前 `analysis/ffn_transport.py` 和 `dynamics_core.py` 继续作为数学参考与模式递推。
测试覆盖头间分工与头内分散、相同 entropy 不同方向协方差、无来源缺测、Gaussian 条件化、
未来信息确实影响离线后验、持续模式可以退出、FFN 导数、反向写入与有效读出并存、坐标变换不变性。

实际验证还包括：真实小型 Llama 的完整采集—训练—评分—评价；分批导数等于逐方向导数；
detached-KV 的当前query导数与完整前缀参考一致；原生 BF16 分批导数可运行；训练不读标注；标签变化不影响分数；
重复评分和完整断点续跑不加载 LLM。小模型随机权重只用于验证接口，不是自然检测结果。

未完成：8B/24GB GPU 运行、自然 RAGTruth AUROC、新机制的自然普适性证明、完整跨token导数。
目前没有本地 GPU/用户模型和自然缓存，不将软件测试的 AUROC 当方法效果。

现有 reference/test：

```bash
python -u main.py dynamics --stage run \
  --reference-output outputs/native_support_validation32/reference \
  --output outputs/native_support_validation32/test --resume
```

一次补采后可用 `--stage score` 重算，不加载 LLM。完整官方数据准备入口与一键脚本见
`experiments/native_support/README.md`。新结果保存在 `state_dynamics/`，原缓存和基线不覆盖。

局部检查命令（不加载 8B）：

```bash
PYTHONPATH=teaching/state_audit/src python -m pytest -q tests/test_native_dynamics.py tests/test_dynamics_pipeline.py
```

## 参考

- Information Flow: https://github.com/rxu0112/RAG-information-flow/blob/main/Information_Flow_Reveals_When_to_Trust_Language_Models.pdf
- 官方实现: https://github.com/rxu0112/RAG-information-flow/tree/263bec3c82a835707c124c17dc2d7c350fc522c1/proposed
- Step-Saliency: https://arxiv.org/html/2604.06695v1 （§3.1、§6、附录 A.1/E.3）
- Preplan-and-anchor: https://arxiv.org/html/2510.13554v1
- 输入驱动系统: https://pmc.ncbi.nlm.nih.gov/articles/PMC7839411/
- rSLDS: https://proceedings.mlr.press/v54/linderman17a.html
