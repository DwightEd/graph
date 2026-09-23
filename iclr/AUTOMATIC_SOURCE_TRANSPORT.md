> 历史研究记录：对应评分实现已退出当前主线；需要复现时查看 Git `9d4ba17`。当前实现见 [VALUE_PATH_TRANSPORT.md](VALUE_PATH_TRANSPORT.md)。

# 无需人工证据类型的来源状态模型

2026-09-23。实现入口：`python main.py transport`。

本轮将上一轮研究设计中可由现有缓存直接测量的部分落地：自动来源身份、逐层逐头响应、原生历史连接和对称状态保持。没有逐样本证据类型表，没有语义真假教师，没有 SVD，没有自然幻觉标签拟合。尚未实现完整跨 token JVP 或自动语义约束识别，不将当前图称为完整因果信息流。

## 1. 不标类型，仍然能保留来源身份

直接复用 `state_dynamics/capture/NNNN/sources.json` 的自动分块。它由 prompt 来源区间、标点与长度上限产生；没有来源区间时明确使用普通 prompt 变体。每块保留 ID、token 位置和原文，`semantic_type="unassigned"`。

不要求用户逐个填写“实体、数量、否定、范围”。这些是上一轮提出的可选语义解释，不能成为主检测的运行前提。自动块只表示信息在哪里，不能等同正确证据，也不能把无标签聚类后得到的状态强行命名为相容/冲突。

输出的全部 token 都有候选路由分数；未判断语义类型不等于该 token 被判断正常或错误。若以后加入自动关系解析，必须另报覆盖与误差，并与解析器单独读出比较。

## 2. 一个明确的计算模型

流程：原生缓存 → 每头来源状态相似性 → 真实历史连接加权 → 图上的来源预算状态 → 固定路由读出 → AUROC/AP。

### 原生观测

| 数组 | 轴 | 含义 |
|---|---|---|
| `group_effect` | token × layer × head × group × rank | 同一组固定 native hidden 方向的有符号终端响应 |
| `group_attention` | token × layer × head × group | 原生访问分配 |
| `ffn_state` | token × layer × rank | FFN 写入对相同固定方向的响应 |
| `history_attention` | query-row × layer × head × answer-key | 每头实际读取的历史位置 |
| `choice_contrast` | token × layer × head × group × competitor | 同一 query 内，实际词响应减竞争词响应 |
| `routing_budget` | token × layer × head × source/role | 原生逐边消息范数的来源预算 |

固定 native rank 仍来自旧采集的随机共同坐标，默认 8；本轮没有再进行 SVD，也不能恢复此前未采集的方向。动态候选的 token ID 一起保存，候选槽位不在不同 token 之间直接比较。

### 每个头先比较，再沿该头的真实连接汇总

每个头构造状态向量：

```
signature = concat(
    source_effect / sum_source(norm(source_effect)),
    sqrt(group_attention),
    ffn_state / norm(ffn_state)
)
```

来源轴、方向、符号、层头身份保留。零响应保留为零，不当作真假证据。拼接后的平方距离相当于三部分固定等权的距离和，这是公开的建模假设，不是按标签挑权重，也不是已证明的最优度量。

每头带宽取整答相邻状态的非零平方距离中位数。全为零时核为常数；首尾缺测标志不进入距离。允许离线完整回答，不能将这条路径称为实时检测。

```
kernel[l,h,t,s] = exp(-distance(signature[t], signature[s]) / bandwidth[l,h])
W[t,s] = mean_layer_head(kernel[l,h,t,s] * actual_reuse[l,h,t,s])
```

先按头计算相似性并乘该头的连接，最后才汇总；不是先平均所有头。FFN 正负只改变状态相似性，不产生“负 FFN = 幻觉”的惩罚。FFN 响应没有与头的终端响应相加为所谓总因果贡献。

### 索引必须按查询状态对齐

目标 `t` 的 query 是 `P+t-1`。回答 key `P+j` 对应的查询状态行是 `j+1`，因此历史边从 `t` 接到 `j+1`，不是 `j`。query 自己的 key 对应自环，去掉后不参与保持约束。

这是一张查询状态之间的观测关联图，不是“某生成词犯错就向后传播真假”的图。节点 0 不对应任何回答 key，因此在这张仅用回答历史的图中必定孤立，首回答 token 的风险不变。目标 j 的 key 接到状态 j+1，不能声称把目标 j 的首错风险直接沿其 key 传播。旧采集 detach 了 past KV，终端 `Jm` 不能再逐层相乘来伪造跨 token 归因。

### 状态是向量预算，不是正确/错误隐状态

令 M 是原生逐头来源预算；对每个预算坐标使用同一图，估计 X：

```
min_X  0.5 * ||X-M||² + strength/2 * sum_{t>s} W[t,s] ||X[t]-X[s]||²
X = (I + strength * Laplacian(W + W.T))^(-1) M
```

这是观测条件下的 Gaussian 场均值，使用 Cholesky 求解，不显式求逆。输出方差是单位观测精度下的工作模型方差，不是真值置信度。

来源、层和头的预算坐标不混合，但各坐标共享同一个最终聚合的 token 图；不能称为每个头独有的传递算子或已识别多头协同。图可让来源访问状态相近且实际相连的位置共同约束估计。保持对正常与错误完全对称，不把保持时间、预测残差或历史模式后验当幻觉风险。

在非负边权下，解对每个预算坐标是输入的凸组合：常数预算不变、非负预算保持非负、不会凭空制造超出原坐标范围的值。零 strength 恢复原预算；重新按组求和可能产生浮点舍入量级差异。

### 固定风险方向

```
risk[t] = mean_layer(
    sum_head(history_budget-source_budget) / sum_head_group(all_budget)
)
```

`transport_route` 在状态预算上读出，`raw_route` 使用现有 `route_scores` 的原始数值；原始分母与 self/special 规则不变。另保留离线 16 行均值、attention 和 entropy 作为比较。

**这是向量响应决定连接、预算状态经图估计后的路由候选。** 最终方向仍来自已有效果证据的路由基线，不是声称无标签地识别了语义冲突。它可能降低噪声，也可能平滑错误；是否提高效果由同样本 AUROC/AP 决定。常数图或不合适的度量可能退化成普通平滑，因此必须保留普通均值比较。不能将这个版本当作上一轮完整约束传递理论已经得到验证。

## 3. 缺测、原始预算与缓存

首行没有来源变化差分；最后两词没有严格未来 query。两者使用独立掩码，未进入 kernel 或似然；末尾没有未来边，不表示零值未来证据。最后两行仍正常评分，不删除或强制低分。

原 B+4 分组将 self、special 与历史拆开，不能直接拿严格 history 桶复刻 R。本轮独立构造 B+3 个路由桶：B 个原来源块、一个未归块来源桶、精确 history、剩余质量。官方来源模式的 history 包含 answer self/special，首个 prompt self 不属于 source；普通 prompt 变体遵循其原定义。

原始 `token_*.npz` 必须存在。仅有 `audit_data.zip` 的压缩表征不能恢复来源向量。score 读取缓存，不加载模型、不读取参考集、不训练；run 复用 teaching 原生采集，只在缺少 token 缓存时调用 LLM。

计算保留全头全来源，因而比标量平滑贵。图按 head batch 批量计算，可在 GPU；Gaussian 场在 CPU 使用稠密求解，成本约 O(T³+T²LHG)。原始缓存按回答加载，不同时堆叠全部回答。没有 8B/24GB 新运行时间，不能承诺具体速度。

## 4. 文件职责

| 文件 | 职责 |
|---|---|
| `transport_observations.py` | 读取原生缓存、来源文本映射、预算与缺测协议 |
| `transport_graph.py` | 每头向量状态与实际历史连接 |
| `transport_state.py` | Gaussian 场求解与固定预算读出 |
| `transport.py` | 主控：采集/评分/保存/调用既有评价 |
| `transport_report.py` | 简短 HTML 比较 |

没有添加证据类型标注器、通用模型工厂、真假分类器或合成错接训练。旧 `support` 基线、`dynamics` 结果与缓存保留。

## 5. 运行

已有 32 答的原生 dynamics 缓存：

```bash
git pull --ff-only origin main
python -u main.py transport --stage score \
  --output outputs/native_support_validation32/test --score-device cuda:0
```

CPU 评分省略 `--score-device cuda:0`。没有新的 reference/fit 步骤。自动读取当前目录的 `annotations.json`，只在全部分数保存后评价；缺标签时仍保存分数，不伪造 AUROC。

首次从已有 settings/manifest 补采：

```bash
python -u main.py transport --stage run \
  --output outputs/native_support_validation32/test --device cuda:0 --resume
```

capture 的 rank/choices/block-tokens/seed 必须与已有 capture_settings 一致；score 自动读原采集配置。输出在 `source_transport/`：

- `summary.json`、`evaluation.json`、`comparisons.csv`：AUROC/AP、首错/延续、前后半段、同答/来源等权和基线差值。
- `tokens.csv`、`high_risk_normals.csv`、`onsets.csv`、`recovery.csv`：全部分数与诊断分组。
- `responses/NNNN/state.npz`：原预算、推断预算、实际/加权连接、带宽、方差、未来缺测掩码。
- `responses/NNNN/choices.npz`、`sources.json`：当前候选作用和自动来源文本；完整原生响应继续保存在原采集目录。
- `report.html`、`timing.csv`：比较表、token 分数和逐答耗时。

## 6. 本轮验证范围

小型真实 Llama 验证原生采集到评分评价；不需要 reference 或人工证据表。针对性检查来源/头身份、候选对齐、历史键偏移、原路由数值、首尾缺测、零强度、常数/非负保持、断边隔离；改变标签不改变分数；完整缓存评分不加载模型。

未运行自然 RAGTruth 新成绩、未验证 GPU 峰值显存、未证明语义约束识别或新机制。本版应先在原 32 答缓存上与旧 R、均值和旧 dynamics 同样本比较，不启动全量重跑。

## 7. 32 答结果与缓存审计

用户随后提供同一 32 答、7176 token、604 错误的自然结果：transport AUROC/AP 为
0.718068/0.191248，raw 为 0.711559/0.185660，离线均值为 0.731726/0.186802。
transport 相对均值的回答前半段 AP 差为 -0.083156。上述结果来自用户运行，不能当作本地重跑。
新版仅增加审核入口，不改检测公式、参数或已保存分数：

```bash
python -u main.py transport --stage audit \
  --output outputs/native_support_validation32/test
```

输入只需现有 `settings.json`、`annotations.json`、`source_transport/scoring_protocol.json`、
`evaluation.json` 和逐答 `scores.npz` / `state.npz` / `sources.json`。
不读取原生 capture、不加载 tokenizer/LLM、不重新构图、不重新求解状态，不需要证据类型标注。
官方幻觉标签只在审核时用于分组、评价和图端点分类；标签不反馈给检测。
可用 `--annotations` 指定同 token 对齐的其他标注；若与旧评价不同，会在核对表中显示。

输出目录 `source_transport/audit/`：

| 文件 | 内容 |
|---|---|
| `tokens.csv` | token、上下文、首错/起点/延续、回答前后半、所有分数、预算/来源份额变化、图度数与邻接标签质量 |
| `metrics.csv` | 全体及每条回答的绝对 AUROC/AP；含后续 span 起点、同答与来源等权 AUROC |
| `ranking_changes.csv` | 对 raw/均值的分数变化、完整并列排名区间、同答排名改善、正常词排在前面的数量和 AP 信用变化 |
| `ap_attribution.csv` | 按回答/阶段/位置分解全部、前半、后半各自的 AP 差；各分组贡献之和严格恢复该范围 AP 差 |
| `top_budget.csv` | 同等 top-k 数量下的预期错误数、正常数、precision/recall；并列边界给出上下界 |
| `budget_changes.csv` | 哪些 token 的 top-k 纳入权重增加或减少，附标签与位置；按回答/target 连接 tokens 表可取上下文 |
| `auc_half_pairs.csv` | 错误位置半段 × 正常位置半段的四种比较，解释总体 AUC 与分组 AUC 差异 |
| `score_similarity.csv` | transport 对 raw/均值的 Pearson、Spearman 及绝对分数改变量分布 |
| `state_groups.csv` | 按阶段、标签×回答半段、回答统计度数、保持系数、预算改变量和风险改变量 |
| `graph_edges.csv` | 逐答正常/错误/未知端点的实际复用质量与条件化后质量，分别按查询目标标签和历史 key 标签对齐 |
| `state_checks.csv` | 预算读出、原路由重分组舍入、度数、保持系数和三类预算坐标的状态方程残差 |
| `evaluation_checks.csv` | 与旧评价逐项核对 token 数、正负数、AUROC/AP |
| `bootstrap.csv` / `.npz` | 按完整 source 成对重采样的差值、95% 区间、有效重采样数及全部重复结果 |
| `responses/NNNN.npz` | 完整 token/query 对齐、有效掩码、标签、分数、两种边矩阵及逐 token 状态统计；不复制大型预算张量 |

自动打包到 **`source_transport/audit_data.zip`**，只包含上述审核数据与 summary，保留原始结果。
默认 `--bootstrap-replicates 1000 --seed 37`；`--bootstrap-replicates 0` 跳过区间估计。
CPU 即可，保留逐答和重采样进度条。

### 解释约定

- `front_half/back_half` 按完整回答 token 位置切半，不是幻觉 span 的前后半。
  `later_onset_vs_normal` 精确取 onset 且非 first；summary 另报首错不属于 onset 的数量。
- AP 信用由完整同分阈值的 precision 分配给正例，分区之和等于该范围 AP；这是排名账本，
  不表示某回答因果地造成其他回答的损失，也不同于各回答单独算 AP 后再平均。
- top-k 边界同分按均匀随机顺序给出纳入概率；`selection_change` 是该概率的净变化，
  不是任意打破同分后给出的唯一告警集合。没有校准阈值，不称为部署误报率。
- receiver t 对应目标词 t，donor 查询状态 s 对应实际 key 词 s-1。查询目标标签审核状态
  平滑连接，key 标签描述实际读取的历史词；无效标签记 -1。两种表均不证明因果错误传播。
- 邻接错误比例、跨类边质量是缓存图与标签的关联；保持系数是条件邻居系数，方差是工作
  模型方差。图状态和风险没有被标注修正。状态分组为 token 加权描述统计。
- 同一 source 的所有回答/token 构成一个重采样簇；方法和阶段共享每次 source 抽样。
  区间是 pooled 指标差的探索性 percentile 区间。无正例或单类重复分别记 AP/AUC 缺测，
  并报有效次数，不补零。只有一个 source 时不报告置信区间。
- 状态方程只检查最终读出所用 source/history/other 的逐层线性汇总坐标，
  不声称逐个验证完整层头来源预算，也不会为检查而重新运行原生模型。

软件验证使用人工构造缓存及小 Llama 的既有集成夹具；自然审核数字必须由用户缓存产生。
