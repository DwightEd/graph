# 来源—中继信息分解：实施计划

基准：agent/graph-structure-audit / 54c12437f5a50ec18e3ab0c6a44c8b810bd4a7b7。
本轮顺序：先提交计划与接口骨架，再填充实现和回归测试。旧图自编码器保留为 baseline，不把它的重构误差称为机制证据。

## 收敛问题

一个历史 token 被读取，不代表它仍然承载 prompt 来源。先在逐 layer/head 的 token DAG 上追踪来源，再比较直接 prompt 路径与 response 中继路径。只建模 attention 路由代理；不声称恢复 WV/WO 消息或事实语义，也不声称已经证明新颖性或检测增益。

## 数学对象

对每个物理通道分别构造 source→query 加权 DAG。prompt 是边界来源；缺失的 prompt 内部边不补零冒充已知。query=q 与预测位置=q+1 分开保存。

对 response query q，来源分布 F_q 满足：
F_q = sum_{i<p} a[q,i] delta_i + sum_{p<=j<q} a[q,j] F_j
      + a[q,q] delta_self + (1-sum_j a[q,j]) delta_unknown。
self 是 token 图在同位置处的终止边界，不表示事实错误；unknown 是截断或未观测来源，不重分配给已知边。这里只沿同一 layer/head 的 token DAG 递推，是诊断性路径模型，不是跨层真实 Transformer 消息传播。

核心分解：H(Z)=H(Z|C)+I(Z;C)，C 为最后一跳 carrier，Z 为追溯来源。
比较直接与中继两条 prompt 可达分支：
D=direct/direct_mass，R=relay/relay_prompt_mass，alpha=direct_mass/(direct_mass+relay_prompt_mass)。
source_mismatch_bits = H(alpha D+(1-alpha)R)-alpha H(D)-(1-alpha)H(R)。
这是条件于 prompt 可达的分支—来源互信息（加权 JS），不是幻觉概率。任一分支为空时返回不可用，不用零伪造一致。另报 prompt_reach、self_terminal、unknown，不能把缺失质量当幻觉。

可选 root_bins 是固定 prompt 地址合并；默认精确 token 根。相同合并映射下 JS 不会增大，但会失去细粒度差异，必须记录。不是无损压缩。

## 模块

- data.py：保持标签盲读取；数据集按文件逐样本迭代。
- channels.py：统一 canonical CSR、方阵、显式 query 位置的紧凑矩阵；逐通道交付 CSR。
- information.py：熵、JS、来源传播和边际保持的历史端点对照。
- reanchor.py：逐头统计；固定旧坐标集合的质量增加；只用过去的事件阈值；FAI 独立离线输出。
- reanchor_run.py：流式构图/测量/保存；不运行 LLM，不读取标签。
- evaluate.py / reanchor_evaluate.py：正确处理 ties、缺失覆盖和 token 对齐；标签只在评分保存后读取。

## 必修错误

取消预先 head/layer 平均、全回答分位数、FAI 混入在线特征、窗口移动假事件、切片原地归一化、空条件分布伪造、二值直方图退化、log-loss 广播、CSR query 偏一位。lookback 命名迁移为 reanchor；保留历史结果不改。

## 检验与消融

合成回归：CSR/稠密等价；prefix 不变；head 交换可见；同一连接跨窗口不触发；零质量无伪 JS；二值分离不塌箱；输入不变；ties 正确；缺失来源守恒；同根不同 carrier 一致；多跳换根改变分数；单跳和按 lag 分组重排历史端点对照。
真实评估：官方 train/test 独立，缓存文件不带 offsets/身份时可以分析，但不能冒充完成 RAGTruth 评价。先看 source_mismatch_bits 对正常换主题/纠正的误报、覆盖率及 source bootstrap；不在 test 上选择方向或融合权重。本轮不训练真假分类器，不组合一堆指标成未验证总分。

## 依据与边界

FlowTracer: https://arxiv.org/abs/2606.10646 （目标相关路径和守恒，不是本实现的算法代码）。
Abnar & Zuidema: https://arxiv.org/abs/2005.00928 （attention flow 是近似解释）。
Mixture entropy: https://arxiv.org/abs/1805.11257 （混合熵与加权分量熵之差）。
SciPy entropy/relative entropy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html 。
上述信息论恒等式是已有数学；待验证的创新点是来源—中继分解及缺失感知的 RAG 路由诊断，不是创造了新的熵或证明 attention 等于语义信息。
