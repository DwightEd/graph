# 位置／符号行为先验 v2：审计其能否帮助去噪

v2 修复普通 key 零质量误删。必须用新输出重跑 prepare/profile/fit/score，不能续用 v1 的 observations 或 bank。
质量加权熵定义为 m H(a/m)，m=0 时为 0，并同时保留普通质量和条件分布是否有定义的标记。
局部 Q/K 角色探测仍需要条件概率，不将没有质量的交换块认作角色证据。
观测公式与边界见 [优化说明](../head_geometry/OPTIMIZATION_V2.md)。

对应 Urrutia 等人的 [Decoupling Positional and Symbolic Attention Behavior in Transformers](https://arxiv.org/html/2511.11579v1)，第 3 节、附录 A.3。

论文以内容块交换后的注意力检验位置不变性与内容等变性。对交换前块均值 `v=(d_i,d_j)`、交换后同槽位均值 `v'=(d'_i,d'_j)`，计算：

```
alpha_swap = softmax_swap(abs(d_i - d_j) / temperature)
positional = sum_swap alpha_swap * cosine(v', v)
symbolic   = sum_swap alpha_swap * cosine(v', reverse(v))
```

它们是给定输入上的 routing 行为，不能直接命名为“事实头”“无用头”。近均匀输入可能两项都高。论文还分析 RoPE 频段；本实现不复现其频率限制训练和 canonical-task 性能实验，也不直接按高／低频删头。

## 本次明确检验的假设

**H1：** 原缓存中的部分变化主要由位置偏好头承载；按独立无标签探测冻结这些头后，减少它们的输入是否优于同层等数量随机去头？

**H2：** 局部路由分类是否随输入、查询位置而改变？天然 QA 的局部 Q/K 探测，与可保持事实内容的完整输入重排分别记录，不能混成一个“永久角色”。

**H3：** 去头是在去干扰还是丢判别信息？无标签 kNN 与明确标记的监督 layer-block LDA 都报告；LDA 仅作诊断，不参与先验、主分数或阈值。

## 一键运行

在现有 research 环境、仓库根目录运行。主模型必须与 attention cache 的**观察模型**一致；它不一定是 RAGTruth metadata 中原回答的生成器。

```bash
git pull --ff-only origin main &&
bash experiments/unsupervised_token_graph/head_roles/run_all.sh
```

默认 QA、Llama-3.1-8B-Instruct、本项目既有 TRAIN/TEST 路径，输出 `outputs/head_roles_v2`。包含：原缓存读取 → TRAIN 局部探测 → 冻结去头对照 → TEST AUROC/AP → 监督诊断 → 完整输入 binding 复核。无标签主检测和监督诊断有不同输出，不混报。需要 torch、transformers、numpy、scipy、sklearn、tqdm、matplotlib。

只需主检测时传 `--no-binding-check --no-diagnostic-lda`。模型迁移时显式给出同一 checkpoint 的 `--model /path --tokenizer /path`。CPU 小模型软件验证可设 `--device cpu --dtype float32`，不是推荐在 CPU 运行真实 8B。

只重新评价：

```bash
python -m experiments.unsupervised_token_graph.head_roles --phase evaluate
```

已有冻结预测后单独跑监督诊断或完整输入复核：

```bash
python -m experiments.unsupervised_token_graph.head_roles --phase diagnostic
python -m experiments.unsupervised_token_graph.head_roles --phase binding --resume
```

修改科学参数须使用新输出目录。`--resume` 保存并复用逐答 observations、逐探测 NPZ、参照库和预测；不覆写原 cache。一次只在一个进程使用同一输出目录。

## 实验 A：自然 TRAIN 上的局部 key 内容交换

从官方 TRAIN 按 source 拆参考／校准，比例约 80/20，seed=17。角色探测仅使用参考来源：默认 8 个 source、每个 4 个均匀分布的已覆盖回答位置。它不按幻觉、熵或起点选 token。

预测回答 token `t` 的 query 是 `q=P+t-1`。在该 query 之前，取连续普通 key 的等长 8-token 块，随机选择至多 8 对交换。特殊 token、query 自身、未来位置不参与交换；短前缀不足两块记未测。

每个真实 Llama 层保持原来的 query 和全部上下文化表示，交换 **RoPE 前的 key 内容**，随后在原目的位置应用该 checkpoint 实际输出的 cos/sin。保留 GQA 的 Q-head / KV-head 对应关系。HF Llama 的旋转平面是前后半维配对，不是相邻维配对。新 attention 仅用于本层探测，不注入真实模型后续层。

这实现论文定义的局部置换检验及其块评分形式；不是将整个文本重排后再次执行模型。Q/K 表示可能已经包含上游位置和关系信息，因此分数描述当前路由算子的敏感性，不表示获得了纯语义子空间。

原始行在排除特殊 key 后归一化。重构的基线 attention 必须与原生 eager 行一致；默认最大绝对误差容限 .02，用于 BF16，保存每层/head/query 的实测误差。需要精度复核时用新目录运行 float32 和更小容限。真实行普通 key 质量下溢会报错，不补零。

使用原文 softmax 权重公式；temperature=.05 是本实验冻结的选择，并非声称来自作者默认值。默认只把满足下述条件的 head 作为可去除候选：

- 至少 2 个参考 source 有效；按 source 等权聚合，75% source 的 gap 方向一致。
- 被交换块平均质量 ≥ .01；块对相对差 `abs(d_i-d_j)/(d_i+d_j)` ≥ .1。
- `gap=symbolic-positional` 小于 −.05 为位置偏好，大于 .05 为符号偏好；其余不分类。
- 每层最多去掉 25% 的 heads，按相应偏好强度排序。上述数值是预设实验条件，不是验证过的语义边界。

两个 cosine 都高不自动表示多功能。均匀／低质量／低辨识／跨 source 不稳定的 head 保留。若无可去除头，各方法相同是合法阴性结果，summary 明确输出 removed=0，不强行凑数量。

## 实验 B：去头对照与自然幻觉评价

完整读取原缓存每个 head 的已保存 causal attention，排除特殊 key，提取 7 个通道：self、prompt、前 10 个历史位置、更早历史、质量加权熵、最大单 key 原质量、普通 key 总质量。各路由保留原子质量，不重新归一化；已保存行的普通质量为零是合法零观测，缺行才记 NaN。self 是前一输入词自身；prompt 不能叫“适用证据”。保留物理 head 身份。稀疏 cache 的统计仅针对已保存普通质量，`retained_mass` 一起保存，未保存尾部不冒充真实零。

分别使用当前原向量 raw、同层保留 heads 的相对向量 contrast。**先去头，再计算保留头的同层共同分量**，防止被删除头从中心化步骤泄漏回来。没有时间窗口、EWMA 或新神经网络。

每种表示运行 9 个冻结版本：

| 版本 | 改变 |
|---|---|
| all | 全部 heads |
| drop_positional | 去掉可靠位置偏好候选，主方法为 contrast__drop_positional |
| drop_symbolic | 去掉可靠符号偏好候选，反向对照 |
| random_0 / 1 / 2 | 每层去头数量与 drop_positional 完全相同，3 个固定随机种子 |
| symbolic_random_0 / 1 / 2 | 每层去头数量与 drop_symbolic 完全相同，单独的 3 组随机对照 |

去头只作用于检测器输入，不改变 LLM。drop_symbolic 的实际数量可能不同，不能当完全同维对照；两类各自的随机组才严格匹配维数；evaluation.json 的 symbolic_minus_random 单独报告符号去头减匹配随机去头。

所有版本共用参考／校准 source 划分和参考 token 位置。每来源最多 16 个 token、库大小最多 2048；列中位数/IQR 标准化、5-NN 欧氏距离除以维数平方根；混合无标签校准的 95% 分位数报警。该阈值不保证正常样本 FPR=5%。表示变化没有解决“罕见是否等于幻觉”的问题，不预设结果改善。

默认主比较使用相同的七通道输入；不能将其与旧 self-only 成绩作纯先验增量比较。要单独检验原来的 self 信号：

```bash
OUTPUT=outputs/head_roles_self_v2 bash experiments/unsupervised_token_graph/head_roles/run_all.sh --features self
```

全部 TEST 分数冻结后才读取自然标签。报告 AUROC、AP、覆盖、报警 recall/FPR、首错、span onset、continuation、span 前半／后半、此前错误条件、检测延迟及覆盖；主方法减各对照的差在共同 token 上按 source bootstrap。前后半以原标注 token 区间二分，奇数多出来的词归前半；不是语义决策点。既有 TEST 已多次分析，只作探索性验证。

## 实验 C：去头是否损失可读出的信息

`diagnostic/supervised_lda.csv` 显式使用自然 TRAIN 标签，在同一参考 token 集上拟合每层独立协方差的 LDA，固定 .1 对角收缩，TEST 只评价。所有 masks/bases 均可比较。未同时抽到两个类别时报告缺测，不补一个分类器。它不是原 0.8427 的完整协方差 LDA，也不是无监督成绩。

- kNN 提高且超过同维随机组，才支持这里的先验比任意删维更有用。
- LDA 判别力明显降低，说明位置偏好头中也有判别信息，不能宣布这些头是噪声。
- LDA 有效、kNN 无改善，说明读出假设仍不合适，不把负结果归咎于没有真假差异。
- 提高仅在 continuation 或后半段，必须同时检查 onset 与正常恢复位置，不能称为提前预警。

## 实验 D：完整输入交换及迁移

`binding` 构造 8 个明确的姓名—颜色事实，对第 1/4/8 块提问，交换整个事实块并重新运行 LLM。事实绑定保持；各块 token 数可以不同，按新边界计算块平均。使用原 tokenizer/chat template，记录所有层/head 的论文评分。

此时 query 与全部上下文化状态也会变化。它与实验 A 的局部干预分开保存，通过 `binding/transfer.csv` 比较 head 偏好是否保持。不同任务、全输入与局部干预的差异都可能改变分类；不以此重新挑选自然 TEST 的 heads，也不声称两个实验测的是完全相同的量。

## 看哪些结果

- `profile/heads.csv`、`head_roles.png`：实际每个 head 的位置／符号分数、可辨识性、去头名单。
- `profile/*.npz`：精确 query、交换端点、每次交换前后块均值和原生重构误差。
- `profile/summary.json`：独立来源数、实际去除数、数值误差。
- `predictions/metrics.csv`：18 个无标签检测对照及位置基线的 AUROC/AP。
- `predictions/evaluation.json`：按来源的配对差、区间、延迟、覆盖及 availability 窗口统计。
- `observations/manifest.json.coverage_summary`：每答缺行、观测零质量与有效位置统计。
- `diagnostic/supervised_lda.csv`：单独标记的有监督信息保留检查。
- `binding/heads.csv`、`binding/transfer.csv`：完整事实块交换与自然局部先验的对照。

软件测试使用真实随机初始化微型 Llama（含 GQA）及合成缓存，检验 RoPE 配对、原生重构、未来不变性、特殊 token、均匀不可辨识、同层随机数量匹配、去除头不能经中心化泄漏、标签／TEST 不影响拟合以及端到端冻结。没有真实 RAGTruth cache 和 8B 权重的环境中不产生自然检测成绩。

测试命令：

```bash
python -m pytest -q experiments/unsupervised_token_graph/head_roles/tests experiments/unsupervised_token_graph/head_geometry/tests
```
