# 原生分布响应与条件状态检测 v1

实现入口：`python main.py transport-observable`。
本文件区分外部研究依据、我们自己的检测设计和已完成的软件验证。
没有自然数据上的新 AUROC/AP；不把候选自动替换为主基线。

## 1. 外部依据及其适用范围

Fernando & Guitchounts, *Dynamics of the Transformer Residual Stream: Coupling
Spectral Geometry to Network Topology*, arXiv:2605.14258v1：
https://arxiv.org/html/2605.14258v1

本次采用的思想是：残差更新的局部 Jacobian、去掉 skip 后的更新算子、沿深度复合的传递，
以及输入/输出方向之间的几何关系。原文并未验证幻觉检测。
附录 A.2 将观测到的末词激活送入无 KV cache 的独立单 token block；
该算子不是当前 RAG 完整上下文的注意力路由导数。
附录 A.4 的图节点是残差坐标，不是 token 或 attention head。
附录 A.6 的代码链接仍为 `[redacted for review]`；这里是独立实现，没有声称复现作者代码。

因此不搬用论文的层分界、社区或谱阈值；不把其平均 Jacobian 的结论当成每个生成位置的事实。
本实现保留完整上下文，通过原生 VJP 计算累计算子在输出可观测方向上的作用。
不做 Schur 干预，不构造 4096×4096 的逐 token 全谱。
旋转与非正规性不是同一性质；以下 Gram/夹角也不是 Henrici departure 的替代估计。

此前方法仅量化消息幅度或原词优势的正负，未把“注入后会经历什么变换”作为检测状态。
本次补上这个缺口。原词更易预测不作为正确性定义。

## 2. 测量对象：输出分布的局部可观测性

固定原始回答前缀，在 query = P+t-1 读取原生分布 p_t。设消息 m 注入某层 attention
后的残差 u；从 u 到最终 logits 的导数为 D_t。它包含后续层、RMS、Q/K 和 SwiGLU
的原生导数。针对当前 query 注入，后续因果层的当前位置沿层传播，历史输入仍是条件。

类别分布的 Fisher 矩阵为 F_t = diag(p_t) - p_t p_t^T。局部有：

    KL(p(z) || p(z + epsilon D_t m))
      = epsilon^2 / 2 * m^T D_t^T F_t D_t m + o(epsilon^2).

这里的 m^T D_t^T F_t D_t m 是消息扰动对分布的局部影响强度。
它不依赖原词 y_t 的梯度正负，既不等于事实支持，也不声称执行有限幅度删除消息。

不显式构造 F 或 D。生成 K 个固定种子的 Rademacher 向量 r/sqrt(K)，令：

    xi = sqrt(p) * r - p * sum(sqrt(p) * r)
    sum E[xi xi^T] = F
    b_k = D_t^T xi_k
    v(m)_k = m^T b_k

于是 E[v(m)^T v(n)] = m^T D_t^T F_t D_t n。
K 默认 8，是随机估计精度预算，不是学到的低维语义、不保证足够精确。
概率作为当前点的度量固定，不能再对 probe 的 p 反传而改变目标。
随机符号在 CPU 按种子产生后搬到设备，保证设备间 probe 定义一致。

## 3. FFN 分开保留，交互不能用正负标签替代

    v = u + F(N(u))
    B_before = B_after + J_(F o N)^T B_after

针对同一来源消息分别保留：

    response_residual = m^T B_after
    response_ffn      = m^T (B_before - B_after)
    response_total   = response_residual + response_ffn

各项是 K 维响应。任意单个坐标可随随机 probe 符号变化，不具有事实极性。
响应的相对方向、大小和交叉内积描述增益、抵消、转向；都不能直接命名为语义矛盾。
FFN 的 actual write 与 FFN Jacobian 对已有消息的变换是两个对象；此处测后者。

逐层保存 B_before B_before^T、B_after B_after^T 和交叉 Gram，及对应能量与夹角。
`operators.csv` 的 `sketch_effective_rank` 从小 Gram 的非负特征值计算，最多 K；
它是输出加权随机观测的谱，不是完整累计 Jacobian 的谱、条件数或真实有效秩。
没有实现残差坐标社区检测，也没有据此宣称所有头协同或中间层必然最重要。

## 4. 检测状态：保留多头响应的联合结构

原始缓存保留 layer × physical_head × automatic_source_block × probe。
来源块沿用已有输入划分，不需要为每个样本标实体、关系、限定条件。
predictor self 单独成组；它对应已经输入的上一个回答 token，不是当前 target。

跨回答的块编号没有语义对应。用于参考比较时，块被按 source/history/self/other 四个角色
做相干求和，residual/FFN 两通道保持为不同的行。每层形成：

    A_t: [2 channels × H physical heads × 4 roles, K probes]
    G_t = A_t A_t^T / ||A_t A_t^T||_F.

保留全部头的行身份和交叉项。不同头可有不同方向，不先将所有头平均。
总尺度另存 `response_scale`；主核使用归一化图，不能检出所有来源一起等比例收缩。
块内相反消息仍可能抵消；原始块响应保留供审计，不声称无损语义表示。

同时保存每层每头的四角色读取分布 q_t，平方根嵌入用于 Hellinger 距离：

    d_G(t,s) = mean_layer ||G_t-G_s||_F^2 / 2
    d_A(t,s) = mean_layer,head ||sqrt(q_t)-sqrt(q_s)||^2 / 2
    d_X(t,s) = (d_G+d_A)/2.

这是两种归一化结构距离的固定等权直和，不是把熵、路由、正负值加成一个风险。
数值实现使用 ||A_t^T A_s||_F^2 的恒等式，避免显式存储大 head×head 矩阵。
该图是消息的输出响应 Gram 图，不是论文的激活相关社区图。

## 5. 条件参考与历史保持

对待测来源排除其全部回答。其余来源是无标签参考，可能混入错误，不能叫全正常集。
至少需要两个参考来源，即输入 cohort 至少三个来源；也可指定独立 `--reference`。
外部参考必须同模型、probe rank/seed、dtype 与角色协议。

上下文 C_t 为 log1p(熵)、熵变化、log1p(回答位置)、log1p(prompt长度)、log1p(来源块数)。
仅在参考上拟合中心和尺度。熵是条件变量，不按熵高直接加风险；这些变量并未识别语义阶段。
按 C_t 从每个参考来源选最多 16 个邻居。首 token 仅匹配首 token，其余匹配有前驱的行。

结构带宽 h 由参考中跨来源随机配对距离的正值中位数得到，与标签、目标来源无关。
相似参考权重为：

    w_i ∝ exp(-||C_t-C_i||^2/(2 dim(C))) / selected_count(source_i)
          * exp(-d_X(t-1,i-1)/h).

首位置省略前驱项。当前状态只在被评分项出现，不进入邻居预选。
候选分数：

    transport_conditional(t) = -log [sum_i w_i exp(-d_X(t,i)/h) / sum_i w_i].

这是条件核相似度的负对数，不是归一化概率密度或幻觉概率。
`transport_context` 去掉前驱条件，作为明确的比较项。
此前状态影响“当前什么样的变化更常见”，不把此前原词的正向质量当作当前事实支持。
高风险不会机械地传染给后词，reanchor 也不触发人为清零。
持续低来源依赖若常见于参考集，仍可能漏检；这也是无标签正常/异常不可辨识的限制。

`transitions.csv` 独立保存来源读取、集中变化、响应结构变化及分数。
它们能表达“集中回看但响应未更新”等观测组合，却不能自动认定证据是否适用。
句法正常延续是否与错误延续分开，必须由实际 token AUROC/AP 和高分正常词检查。

## 6. 运行、输出与成本

在原 research 环境、项目根目录运行：

```bash
git pull --ff-only origin main &&
python -m pip install -r requirements.txt &&
bash experiments/native_support/run_observable.sh
```

默认只运行已有四答 manifest，不扩展数据。每 16 个 query 复用一个完整原生前向，
每个 query 默认 8 个独立 VJP；没有完整 d×d Jacobian，也没有独立候选生成或判决。
GPU 执行模型，CPU 默认完成结构评分；`--score-device cuda:0` 可显式指定 GPU 评分。
这一采集仍比只算 attention 昂贵，不能称为零成本或保证在24GB下任意长度都可跑。
旧标量梯度缓存缺少这些响应方向，首次须补采集；后续评分不加载模型：

```bash
python -u main.py transport-observable --stage score \
  --output outputs/native_support_ragtruth4/observable_transport_v1 --resume
```

改 neighbors/reference 等评分参数须用新的运行目录；原始缓存不改。
需要独立参考时，在新的 output 首次 run 加 `--reference 已采集的observable目录`。
不为改方法自动重采集整个 RAGTruth。

输出：
- `summary.json`、`evaluation.json`、`comparisons.csv`：AUROC/AP、首错/起点/延续、
  前后半段、source-balanced、within-answer；标注缺失时明确不可评价。
- `operators.csv`、`transitions.csv`、`coverage.json`：可直接分析的数据。
- `responses/*/token_*.npz`：完整来源块响应、读取及 pullback Gram。
- `responses/*/state.npz`：实际进入结构核的矩阵、读取和条件变量。
- `responses/*/neighbors.json`：参考回答/位置、权重、距离、有效邻居数及上下文距离。
- `responses/*/trajectory.png`：读取、状态变化与检测分数曲线。
- 自动生成旁边的 `observable_transport_v1_review.zip`，包含以上数据和标注快照。

raw_route/raw_attention/entropy 从同一新采集重新测量；与旧缓存可能有浮点差异。
独立报告 observable_route、原始路由、相同离线窗口均值，默认 risk 仍为 raw_route。
全部分数保存后才读取标签。无拟合真假分类器，无自动选择最佳头/符号/方法。
参考使用其他完整回答，因此 cohort 模式属于来源隔离的传导式评价，不是独立部署评估。

## 7. 软件验证与未完成的自然评价

小型随机 Llama/GQA 验证：Fisher 协方差恒等式、logit 整体平移不变性、
来源消息响应与完整原生有限差分相符、residual+FFN 恒等式、独立 query 的分块/前缀不变性、
资源恢复、结构核与显式 Gram 相符、物理头身份保留、同来源多回答整体排除、
完整 capture→score→evaluate→打包和改变标签不改变分数。
既有 v3 FFN 小模型测试同时回归。

尚未运行本候选的真实8B/RAGTruth；不能给出提高了AUROC、已解耦语义或已证明因果机制的结论。
研究上的增量是“沿原生计算路径定义消息的输出可观测性，再对多头读取/传递状态作条件比较”。
Fisher、随机估计和条件核本身都是数学工具，不应单独包装成新算法贡献。
