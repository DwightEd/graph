# 来源锚定的相邻 token 联合状态：实现契约

本方法尝试读出信息来源的区分信号，不假设所有幻觉都符合同一模式。代码已运行；有效性见 RESULTS.md，不能以实现完成代替检测成功。

## 运输与采纳的测量

`measure.py` 保留原 token 序列与位置，做两个完整世界：原生世界；每一层阻止非来源 query 读取来源 key 的世界。来源 query 可以计算，但不能向其他位置输出来源信息。来源坐标为 blocked logp 减 native logp，方向继承旧来源对比，不按词频定义。

对每个实际输出 token，自动取原生最高的非原词作为固定 rival，目标为 actual logit 减 rival logit。逐 query 重算当前 self K/V、所有下游层及 FFN，过去 native K/V 固定。计算每层每头 128 维 source/history/other 消息和输出目标对消息的梯度，signed adoption 为内积。不同层不能相加解释为独立守恒贡献，正负也不是事实真假。全部 32×32 head 身份保留；原始 channel 消息以 float16 存档，梯度计算 FP32。

生成 token t 的 predictor 在 P+t−1；post-token self 状态在 P+t；邻居 sender j<t 的 key 在 P+j。二者明确区分。边描述符覆盖前 8 个输出 token 的 native signed message 与 attention mass，包含 predictor 自己的 key，即上一词。不是 PageRank 影响力分数。

## 联合推断邻居状态

每个 token 有 r 维隐状态，邻居参与状态方程：

\[
z_t=\sum_{j=\max(0,t-8)}^{t-1}F(e_{jt})z_j+\eta_t,
\qquad F(e)=\sum_{b=1}^{4}e_b T_b,\qquad \eta_t\sim N(0,Q).
\]

每个转移矩阵可带正负方向，不使用“相连 token 必须同真值”的约束。独立 node 无转移；chain 有相同维数/参数数的 lag-group 转移；native 用真实消息描述符；rewired 只交换描述符的 sender；edge_node 将同一局部边描述符作为当前节点观测，但没有隐状态邻居。

观测模型是 \(o_t=Cz_t+\epsilon_t\)，固定第一行 \(C_0=(1,0,\ldots)\)。评分为 \(E[z_{t,0}\mid O,e]\) 换回来源尺度，不读出负似然、密度、后验方差或罕见度。Q/R 对角正定，噪声 floor .02。

v1/v2 使用 source_gap、3072 维 signed 采纳和 1024 维 self diagonal；v3/v4 去掉缺乏条件增量的 signed 节点观测，保留 1024 self 状态，并恢复旧缓存 local/full token 来源差的均值。旧 anchor 的模板不同于新 native 消息，二者只能称统计融合，不能称同世界因果中介。

## 最终带权后验及 EM

1024 个 head 观测容易压过来源 anchor。v3/v4 使用 source weight=1、其余 observation weight=1/1024：

\[
p(Z\mid O,e)\propto p(Z\mid e)
\prod_{t,p}N(o_{tp};C_pz_t,R_p)^{w_p}.
\]

这是 Gaussian likelihood 的 powered factors / generalized Bayes。其归一化后验仍然是精确 Gaussian；目标为 weighted factors 的积分加 MAP priors，不能称原观测的一般生成分布边际似然。矩阵精度为

\[
\Lambda=(I-F)^T Q^{-1}(I-F)+I_T\otimes C^T\operatorname{diag}(w/R)C.
\]

`gaussian.py` 用 banded Cholesky 求均值、Takahashi selected inverse 求相关协方差；EM 的 sufficient statistics 含邻居交叉协方差。loading/transition 使用各 row 噪声条件的 matrix-normal prior，系数 .01×fit token 数。带权 loading prior 与 observation factor 同权重；R/Q 更新包含 coefficient penalty 和 determinant 项，anchor 行固定且不加 loading prior。每步检查同一个 MAP objective 不下降，收敛 tolerance=1e−5 相对 objective，停止不读取检测指标。

v3 独立遮蔽无标签 validation 中固定每 5 个 token 的 1 个来源观测，保留相邻 token 的来源坐标；只在 fit 来源内部划分 validation，按遮蔽来源预测选 r∈{2,4}。v4 固定 r=2，从 v3 冻结参数继续到收敛。v1/v2 遮蔽全部来源坐标，其验证不能直接测邻居来源继承，记录保留。

## 边基与控制

v1 使用 signed/mass 各 2 维 fit-only SVD；v2–v4 的 signed 基通过 fit-only donor/receiver 来源观测回归得到，mass 仍为 2 维 SVD。这是无自然标签的 proxy 自监督，不是真假语义监督。全层头原始坐标仍保存在缓存，边转移仅使用压缩的 4 维描述符。

rewired 在每 query、每 head、lag {3,4}/{5,6,7,8} 内交换消息/mass，lag1/2 固定。不做 BPE 匹配；只改统计模型的端点映射，不物理修改 downstream computation。25% adopted mass 重分配门槛并非所有回答通过，完整统计见报告。native/chain/rewired 转移参数数相同；edge_node 增加 32 个 loading/noise rows，作为输入/容量近似对照，不声称与图完全等参数。

## 评价边界

fit/参考/回归来源互斥，fit 阶段不读自然标签。所有分数先冻结，再按 RAGTruth 官方字符 span 对齐评价；监督 logistic probe 是独立诊断，权重不迁移。完整回答 smoothing、局部居中均值及单元均值均允许未来观测。经验 reference percentile 只是尺度，不是事实概率或正常 FPR 保证。

全部当前来源历史已暴露，四轮仅为探索回归。没有新来源确认；没有图必要性或自然 hallucination 信息流的因果识别结论。
