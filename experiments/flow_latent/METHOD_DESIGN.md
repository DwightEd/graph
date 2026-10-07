# Flow latent：无监督逐 token 图条件因子模型

2026-10-07。方法设计先于代码与新评分。代码独立子目录 graph/experiments/flow_latent；复用 token_backtrace 原生捕获适配器。旧实验和默认 risk 保留。

## 可实施第一阶段与上轮公式的关系

上轮完整混合成员节点/连续 Dirichlet attention 行联合似然仍是后续扩展，第一阶段不声称已实现该完整模型。本阶段先实现可检验的子模型：attention 是观测条件图，拟合由它实际输运的内容和节点向量的条件概率模型。使用 Dirichlet **全局混合权重先验**，不把 token 后验冒称 Dirichlet 后验；不声称学习了 attention 的生成分布。可先证明/否定真实边的检测增量，避免先实现不可识别的复杂边生成模型。

## 观测、节点、连接

P 为 prompt 长度，回答 t 从0开始。节点 x_t 为输入该回答词后的原生 head readout，位置 P+t，维度32层×32头×128坐标。预测条件图来自位置 P+t−1 的 attention：所有来源、其它 prompt 和严格过去回答 key 均保留。query 自己的 key 是过去回答词，保留；目标回答词及未来 key 不可见。

每 head 的来源、历史和其它 prompt 内容分别为 S_t=Σ_source A_tj V_j、H_t=Σ_answerpast A_tj V_j、O_t=Σ_otherprompt A_tj V_j。这三个向量严格来自同一次前向与同一模板。保存全维 x、全 key V、完整 attention、token IDs 和物理层头轴；重构 S+H+O 对应 predictor 原生 head 的误差必须<2e−4。

匹配乱边：在每个 head/query 内，仅在相同 key token ID、相同来源/历史/其它类别和同一二进制 lag 桶内置换端点，保持权重及总质量。lag1固定。若可交换质量过小，阴性结果无法排除图信息，必须汇报可交换质量。

## 路径与谱

回答边 W_tj=A_(P+t−1,P+j)，j<t；严格下三角，因此 W 的 T 次幂为0（T为回答长度），邻接全部特征值为0。v2 使用来源根路径内容 R=S+γWR，γ=.5，即有限路径算子 (I−γW)^−1 S。逐层逐头递推，保留原幅度，既不按 sink 条件归一，也不全 head 平均。它是固定观测图上的内容传播分析，非模型的真实跨层守恒分解，非 FlowTracer 原论文的完整目标 sink/Doob 条件流。矩阵分解针对内容协方差及有限路径算子，不能借零邻接谱声称稳定性。

## 降维与概率模型

全坐标原始数据保存；建模先用 seed42 固定128→4正交坐标投影，保持1024个 head 的独立身份与符号/幅度，再仅在无标签 fit 来源上拟合节点和条件各自的 PCA。节点32维、条件48维，做 fit-only whitening；必须汇报压缩与保留方差，不能说模型使用了所有原坐标的完整信息。

全局 π~Dir(1.2,…,1.2)，z_t~Cat(π)，u_t~N(0,I_4)：

x_t | z_t=k,u_t,c_t ~ N(C_k [1,c_t]+U_k u_t, σ²_k I).

积分掉 u 后 covariance=U_k U_kᵀ+σ²_k I。K=4。E-step 为每 token 的精确 categorical posterior；M-step 为加权 ridge 回归、残差协方差特征分解的 PPCA 更新、Dirichlet MAP 权重更新。不是变分EM的非共轭 MMSB实现；这是可解析边缘的 EM 子问题。记录目标与有效簇权重。一致 MAP：matrix-normal 系数先验 λ=.01×N_fit（固定）；协方差包含系数二次项并使用 N_k+d_c 分母，见 MAP_CORRECTION.md，σ²下限=.05。无自然真假标签训练，无簇真假命名。

主风险是−logΣ_k π_k p(x_t|c_t,k)，每 token 独立数值。异常分数不等于幻觉概率；参考可能包含错误。阈值为来源隔离 reference 的分任务95百分位，只表示参考预算，不代表正常回答5% FPR。

## 冻结迭代顺序

v1：条件 [S,H,O]，node-only 同K=4/PPCA rank4（无回归条件，参数量较少）。
v2：条件 [R,H,O]，专门检验多跳路径。
v3：条件 [R,O]，删除历史解释，检验错误历史易预测是否掩盖异常。仅此一个明确修改；不加 HMM、组共分、真假 classifier，不按样例标签挑簇/头/方向。

三个版本都预先规定，后续展示所有失败和收益，不自动宣布最高 AUC 为最终方法。每版与用相应匹配乱边训练/评分的模型比较，保持模型容量与流程；乱图使用同一 token/node 数据。另保留实际 observer NLL。若真实图对 node、rewire 两项增量来源 bootstrap CI 均不为正，主线未成立；若数据扰动无效先承认对照可辨性不足。

## 一手依据

FlowTracer：https://proceedings.mlr.press/v306/dong26f.html
PPCA：https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis/
Mixture PPCA：https://www.microsoft.com/en-us/research/publication/mixtures-of-probabilistic-principal-component-analyzers/
完整模型推导见共享研究目录 refine-logs/flow_latent_unsupervised_20261007/DERIVATION_PACKAGE.md。以上文献支持分析工具；本实验的风险方向、条件回归及边增量都是额外假设，尚待实测。


## 优化修正版

初轮正则化交替更新并非严格MAP，保留原始输出。公式核对后追加同目标EM修正，详见 MAP_CORRECTION.md；修正版默认输出 flow_latent_20261007_map_v*，完整簇与因子后验均保存。无完整attention行生成模型。

## 审计后容量与保真度限定

node-only仅混合簇/因子rank一致，回归参数量较少；native与rewired模型完全同容量。追加同容量lag_pair对照见CAPACITY_CONTROL.md，使用前两个原生节点，无显式图属性。高保真最终节点子空间保留原方差：fit45.18%、reference29.61%、regression31.01%；消息方差保留率未测，不将节点比例推广到消息。固定30步EM保证目标非降，不保证达到收敛。
