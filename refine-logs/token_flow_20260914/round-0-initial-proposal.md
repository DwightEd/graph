# 无监督 token 图：条件创新与信息流分析

## 问题锚点

- 底线问题：在一个完整 prompt＋response 样本图上，无幻觉标签地给 response token 排异常分数，并检验其是否能定位幻觉。
- 必须解决：直接自重构的复制捷径、未来拓扑泄漏、图统计与事实正确性混淆；错误起点和顺畅延续须分开检验。
- 非目标：不把 attention 当真实因果消息，不以图罕见性证明事实错误，不复用 S10 监督权重。
- 约束：沿用冻结 observer 缓存，来源隔离；标签仅在全部分数冻结后加入。高维节点和实际连接保留；CPU 优先。
- 成功条件：工程上无标签泄漏且索引/稀疏数值正确；科学上固定自然来源评价胜过同容量无邻居和连接置换对照，并改善回答内定位。

## 取舍

旧 GraphAutoencoder 接收目标节点自身输入再重构它，低损失不代表上游能解释该节点；outgoing 特征和全回答归一位置还使用未来。
路线 A 是无目标输入的条件预测，路线 B 是跨层 V/O 消息预测。选择 A 作为当前可执行的可证伪基线；B 需要真实逐层 hidden/V/O 缓存，不能用层平均 attention 代替。
核心贡献候选只有一个：**相对于同来源采样规则下无标签参考分布的上游条件创新**。岭回归、Gaussian 工作模型、图信号能量均是已有工具，组合本身不声明新颖性或顶会就绪。

## 图和时间语义

节点是实际 token 位置 i；边 j→i 对应 attention query i 对 key j，j<i。本版分数是看到 token i 后的 observer 诊断，不能用于提前预测 token i。
只有预测 query=P+t−1 的矩形缓存不可默认为 token t 自身状态；本版拒绝这种缺少完整节点属性的输入。
节点属性默认保留所有 layer/head 的 attention diagonal，也可显式选用完整 [N,D] frozen hidden。两种特征不能混用一个模型；diagonal 只是拓扑属性，不是语义表示。
所有通道中严格大于 floor 的边保留并按通道总数平均；不对稀疏保留质量重归一化，不建立 N×N 中间数组。上游未保存质量单列 unknown。层平均边只是 token 依赖代理，不是跨 Transformer 层的 attention rollout。

## 数学机制

参考来源等权，每来源内回答等权，每回答内 response token 等权，权重 w_i 的总和为1。
随机种子固定的线性投影最多64维（原始节点属性保留），参考集标准化得到 z_i。
令 A_ij 为严格历史边；s_i、h_i 分别为 prompt/history 保留质量。

    m_i^P = Σ_{j<P} A_ij z_j
    m_i^R = Σ_{P≤j<i} A_ij z_j
    c_i = [1, log(1+P), log(1+i−P), s_i, h_i]
    u_i = [c_i, m_i^P, m_i^R]
    B = argmin_B Σ_i w_i ||z_i − u_i B||² + λ ||B_nonintercept||²
    e_i = z_i − u_i B
    a_i = (1/d) Σ_k e_ik² / v_k

v_k 是参考残差的加权二阶矩，下界固定0.05；λ=0.01。不使用目标自身 z_i 作为预测输入，不使用未来 out-degree/总回答长度。
Gaussian 工作模型的每维负对数密度为 a_i/2 + Σ_k log(2πv_k)/(2d)。这是在所选坐标中的条件编码代价，**不是估计出的真假似然比或幻觉概率**。无标签混合参考可能包含系统性错误，错误若可预测会漏检。
无邻居对照只用 c_i，保持相同预测器/损失；它仍有局部质量统计，准确称为 no_neighbors。
置换对照在每 query、prompt/history 分段内置换已保留边的权重，保持端点集合、边数、总质量；只检验权重分配，不能声称检验全部拓扑必要性。全部等权边时对照可能恒等，必须报告。

## 信息论和图统计分别能说明什么

1. 路由熵 H(J|i)=−Σ_j A_ij log A_ij−u_i log u_i（含 self 及 unknown 桶）：量化读向集中程度；不是 token 输出熵 H(Y_i|prefix)。unknown 聚桶会低估真实细分路由熵。
2. JS([prompt,history,self,unknown]_i || 同参考分布) 可测路由分布异常；它只比较路由类别，不表示语义冲突，本版不另加到主分数。
3. 局部 Dirichlet 量 E_i=Σ_j A_ij||z_i−z_j||²/d 衡量相邻表征不平滑；语言中的正常转折也可能很高。条件残差比直接假定相邻相同更灵活。
4. 条件互信息 I(Z_i;M_i^P|M_i^R,C_i)=E log[p(Z_i|M_i^P,M_i^R,C_i)/p(Z_i|M_i^R,C_i)] 需要跨样本联合分布。单个节点上的 attention 值不能直接当 MI。两个拟合密度的分差含模型误设偏差，本版不报告 CMI 数值。
5. 数据处理不等式只约束定义明确的随机变量 Markov 链；Transformer 有残差、并行头和输入依赖权重，不能从层平均 token 图推出事实信息单调损失。
6. 异常延续可以在后续研究中用 r=b+ρAr 的严格历史 DAG 路径和分析。但 attention 大可能是引用后纠正，且旧 P2 已败给链对照；本轮不把传播堆到主分数来掩盖入口失败。

## 统计校准与可识别性

fit、calibration、test 的 source_id 严格不相交。校准来源 b 的统计量 M_b 是其全部回答 response token 的最大 a_i。
测试 token 的保守来源尾部值 p_i=(1+Σ_b 1[M_b≥a_i])/(B+1)，alarm 定义 p_i≤α；并列值保守处理。
阈值取 k=ceil((B+1)(1−α)) 的有序来源最大值；k>B 时阈值为∞，不能用普通 quantile 伪造小样本5%预算。
在冻结评分器及新旧来源分组（包括回答数/生成器/长度采样规则）可交换的条件下，新来源任一 token 报警的概率≤α。它控制混合无标签来源总体报警率，**不保证正常来源条件误报率，也不保证 token FDR**；分布移位时保证不成立。
不可识别反例：两条真假相反的回答如果观测图及属性相同，任何仅依赖这些观测的检测器分数都相同。因此无监督图异常与幻觉等价需要额外的、可检验的区分假设，不能靠数学换名解决。

## 最小验证（冻结后执行）

1. 工程：密集/CSR等价、已知线性条件生成图、目标自输入排除、因果前缀不变、并列和小样本校准、冻结篡改拒绝。合成异常只验证机制，不证明自然幻觉效果。
2. 自然：预固定来源 roster，reference/calibration/test；主分数、no_neighbors、权重置换、原熵/NLL（若缓存有）同一 token 支持集。报告全token、每答第一次错误之前及含首错、严格首错后、回答内/来源等权、覆盖及来源 bootstrap。已用于研究选择的旧自然集合明确标为开发复用。
3. 删除判据：真实连接未胜 no_neighbors/置换，或仅 pooled 而无回答内提升，则不保留“图改善检测”的贡献；不能在同一 test 上调参后宣布新结果。

## 原始文献与边界

- [Abnar & Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385/)：跨层 attention rollout/flow 是相关先例；这里的层平均上游预测不能冒称该算法。
- [Hsieh et al., EMNLP 2024, Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/)：context/generation attention ratio＋分类器，说明质量比不是新贡献；其标签训练与本版不同。
- [Binkowski et al., 2025, LapEigvals](https://arxiv.org/abs/2502.17598)：attention 图谱特征＋探针，意味着谱/图表征本身不能声明首次。
- [Noël, 2025, Graph Signal Processing Framework](https://arxiv.org/abs/2510.19117)：已有 Dirichlet/spectral entropy 路线；其效果主张未在本项目复现。
- [Angelopoulos & Bates, 2021](https://arxiv.org/abs/2107.07511)：有限样本 conformal 分位数与可交换性背景；本方案校准单位是来源块。

## 状态

方法规格已固定供实现和评审；自然有效性尚未验证。预期工程工作只用CPU，不重新采集GPU、不读取自然幻觉标签。本轮方法审查与代码验证记录后续补充。
