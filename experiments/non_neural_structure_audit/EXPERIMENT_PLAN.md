# 非神经结构审计实验计划

> **状态：目标协议，尚未全部实现。** 当前代码只完成 A0a artifact binding 和若干 exploratory/pilot 统计；A0b gold alignment 与 A0c full-pipeline label permutation 缺失，因此总 A0 未通过，正式 A1–A10 全部必须 `BLOCKED_BY_A0`。本文后续 PASS 阈值和 499/2,000 次参数描述的是补齐未实现项后的目标，不是当前一键脚本已经能产生的论文级结论。

## 1. 目标与核心主张

本项目先回答“哪些结构关系确实值得建模”，再决定神经模型应当学习什么。核心主张是：

> 如果某种 token、head、layer、路径或时间结构，在匹配控制和结构特异 null 下仍能稳定解释幻觉差异，并且相对简单基线带来可复现的增量，那么才有理由把它写进后续图模型；否则应保留更简单的统计模型。

本计划不把原始 attention 直接解释为信息贡献或因果来源。当前可从 attention cache 得到的量统一称为 **attention-routing proxy（注意力路由代理）**。只有 A10 的成对模型干预通过后，才允许使用“因果机制”“因果路径”一类表述。

方法设计参考了 attention rollout/flow、贡献分解、因果追踪、activation patching、显著性 sanity check、幻觉早检和 RAG 幻觉检测的成熟做法：[Attention Flow](https://aclanthology.org/2020.acl-main.385/)、[ALTI](https://aclanthology.org/2022.emnlp-main.595/)、[ROME](https://proceedings.neurips.cc/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)、[Activation Patching Best Practices](https://openreview.net/forum?id=Hf17y6u9BC)、[Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)、[Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/)、[Snowballing Hallucinations](https://proceedings.mlr.press/v235/zhang24ay.html) 和 [CHARM](https://openreview.net/forum?id=4twbqwV4br)。

本计划的阈值是本项目准备预注册的决策标准，不是上述论文共同给出的通用常数。正式 confirmation 开始前，可以根据 discovery 的功效分析一次性冻结阈值；打开 confirmation 标签后不得再调整。

## 2. Claim ladder

| 层级 | 可支持的主张 | 最低条件 | 不允许越界的表述 |
|---|---|---|---|
| C0 管线有效 | token、标签、attention 行和 null 的实现对齐 | A0 通过 | 不能据此声称存在结构信号 |
| C1 统计关联 | 某个预注册结构量与幻觉 token 相关 | 对应 A1–A8 gate 通过 | 不能称为信息来源或机制 |
| C2 增量结构 | 该关系超越 nuisance/Lookback，并被结构特异 null 消除 | A2–A8 至少一个结构 gate 通过 | 不能从预测增益推出因果性 |
| C3 联合形式 | 多类关系的非线性交互有必要 | A9 通过 | 只能授权联合模型，不能声称其学到机制 |
| C4 因果机制 | 对目标路径的模型内部干预产生特异、可复现的输出变化 | A10 通过 | 仍需限定在模型、任务和干预分布内 |

论文的最强 claim 必须停在实际通过的最高层级。A1–A9 即使全部通过，也最多支持“attention-routing proxy 包含增量结构信号”。

## 3. Discovery / confirmation 协议

### 3.1 划分单位

- 以完整 QA/sample/context ID 为最小划分单位，绝不按 token 随机拆分。
- 同一问题、同一证据、同一回答的改写或近重复样本必须进入同一 split。
- 当前 canonical 数据已经提供独立 train/test。正式运行固定为：完整 train 仅拟合无标签 reference；test 在读标签前按完整 `source_id` 确定性分成 50% discovery、50% confirmation。
- 当前实现只按 label-free `source_id` 确定性划分，不按“是否含幻觉 span”分层。若未来做分层，只能使用在打开 discovery/confirmation 标签前合法可得的 task、data source、回答长度等变量，且不能让同源样本跨 split。

### 3.2 三个 split 的职责

| Split | 是否读取幻觉标签 | 用途 | 可冻结的对象 |
|---|---:|---|---|
| Reference | 否 | 拟合无标签基准分布，计算 task/position/token-class 条件下的 median、MAD 和分位点 | 归一化参数、匹配分箱、稀疏 attention 已知质量分布 |
| Discovery | 是 | 选择每个 audit 的方向、主统计量、时间窗、交互项和阈值 | 唯一 primary statistic、方向、窗口、模型超参数、决策阈值 |
| Confirmation | 最后一次性读取 | 对冻结假设作一次确认 | 不再选择或调参 |

任何使用 discovery 标签选出的结果都只算 discovery 证据。所有预处理器、分箱、特征筛选器和分类器必须只在对应训练 split 上拟合，避免数据泄漏；这与 scikit-learn 的[官方数据泄漏指南](https://scikit-learn.org/stable/common_pitfalls.html)一致。

### 3.3 token 处理原则

- 构图和路径传播保留全部 token。标点、功能词可能是路径中介，不能在传播前删除。
- primary outcome 只在 content/entity/number token 上报告。
- function token、punctuation、special token 分层单独报告，不能与 content token 混成一个主结果。
- span 标签先通过 tokenizer offset 映射到 token；边界 token 单列为 `boundary`，不能悄悄并入正例或负例。
- 同一回答中的 token 不是独立样本。置信区间和置换均以 QA/回答为聚类单位。

RAGTruth 提供自然生成回答的细粒度 hallucination 标注，数据定义以其[ACL 2024 论文](https://aclanthology.org/2024.acl-long.585/)为准。

## 4. 统一统计协议

### 4.1 主指标

- **AUPRC**：token 级不平衡任务的主预测指标，同时报告正例 prevalence。
- **AUROC**：次指标，不单独用于通过 gate。
- **matched \(d_z\)**：先在匹配 strata 内计算 hallucinated 与 correct token 的差，再用 reference 的稳健尺度标准化；按 QA 等权聚合。
- **增量 \(\Delta\mathrm{AUPRC}\)**：候选结构相对预注册基线，在同一 grouped folds/confirmation 集上的差值。
- **sample-cluster bootstrap CI**：以 QA 为单位重采样，正式运行 2,000 次。

### 4.2 多重检验和方向稳定性

- 每个 audit 只指定一个 primary hypothesis；其余统计量标记为 secondary/exploratory。
- A1–A9 中具有有效随机化/交换性依据的 primary p 值统一做 Benjamini–Hochberg 校正，要求 `q < 0.05`；当前启发式 endpoint pilot 只报告 exceedance rate，不作为 p/q 值。
- grouped CV 要求至少 4/5 outer folds 与预注册方向一致。
- 置换检验以 sample/span 为单位，正式运行至少 499 次。标签置换检验的设计依据 [Ojala & Garriga, JMLR 2010](https://www.jmlr.org/beta/papers/v11/ojala10a.html)。

### 4.3 通用 gate

除非某个 audit 写了更严格门槛，正式 PASS 同时要求：

1. confirmation 效应方向与 discovery 冻结方向一致；
2. sample-cluster bootstrap 95% CI 不跨 0；
3. primary test 经 BH 后 `q < 0.05`；
4. 匹配效应达到 `|d_z| >= 0.20`；
5. 局部组件的增量至少 `Delta AUPRC >= 0.01`，完整结构族或联合形式至少 `>= 0.02`；
6. 若使用 grouped CV，至少 4/5 folds 同方向；
7. 所用 null 的不变量检查通过。

状态只有三种：

- `PASS`：通用条件和 audit-specific 条件全部满足；
- `FAIL`：实现/null 有效且功效足够，但至少一个冻结条件不满足；
- `INCONCLUSIVE`：null 无效、alignment 未确认、有效样本不足或 CI 过宽。

confirmation 中含幻觉的回答少于 50 条时，所有 hallucination-specific gate 标为 `INCONCLUSIVE_LOW_POWER`，不能按 FAIL 或 PASS 解读。

## 5. A0–A10 审计矩阵

| Audit | 问题 | Primary statistic | 核心 null/control | 通过后授权的模块 |
|---|---|---|---|---|
| A0 | token、标签和 attention query 是否精确对齐，管线是否泄漏 | exact alignment + null invariant errors | sample/span label permutation | 仅授权继续 A1–A10 |
| A1 | 幻觉是否首先只是“少看 prompt/evidence、多看 response” | matched Lookback/role-mass \(d_z\) 与 AUPRC | 位置、token 类和已知质量匹配 | Lookback/logistic baseline |
| A2 | 精确 source endpoint 身份是否有额外信息 | real-vs-rewired \(Delta\)AUPRC | 约束 endpoint swap | exact token graph |
| A3 | head 间分工/分裂是否超越 head mean | head-fracture incremental AUPRC | 匹配 token 间 head-profile permutation | head-resolved/set aggregator |
| A4 | layer 顺序是否不可交换 | ordered-vs-shuffled \(Delta\)AUPRC | 递归中的 layer-order shuffle | GRU/SSM/跨层递归 |
| A5 | 多跳路径是否超越直接/一跳量 | full/2-hop-vs-1-hop \(Delta\)AUPRC | 分层 endpoint rewire/path order shuffle | 多跳 message passing |
| A6 | sum 是否足够，还是需要 max/集中度/top-k | candidate-vs-sum \(Delta\)AUPRC | 固定 topology 的 weight shuffle | 与通过统计量一致的 aggregator |
| A7 | 错误后是否存在 response-history lock-in | post-onset persistence \(d_z\) | all-correct pseudo-onset/span shift | gated/hysteretic temporal update |
| A8 | 结构变化能否在首个错误前形成 change point | lead、hit rate、FPR | matched pseudo-onset/circular shift | 在线 change-point detector |
| A9 | 多类关系是否需要非线性联合 | interaction-vs-additive \(Delta\)AUPRC | grouped label/restricted feature permutation | 联合 GNN/GRU 类模型 |
| A10 | 路径分数是否对应模型内因果效应 | patch recovery 与结构分数相关性 | matched random patch/path | 有限定的因果机制 claim |

## 6. 各 audit 的可执行定义

### A0：Alignment、泄漏与 null 不变量

**假设**：缓存中的 query row、生成 token 和 RAGTruth span 存在唯一、可验证的映射；任何下游分数都没有利用 confirmation 标签拟合。

**检查项**：

1. 构造一个带已知 token/字符边界的小型 trace，逐 token 核对 query `t` 预测的是 response token `t` 还是 `t+1`；两种 shift 必须由模型生成语义而不是相关性选择。
2. 检查 prompt offset、BOS/EOS、assistant 前缀、padding 和截断对 response index 的影响。
3. 将 span 的字符区间映射到 token offset，并输出 unmatched/boundary 比例。
4. 在完整 pipeline 上做 sample-level label permutation；分类性能应回到置换 null。
5. 每种 graph null 都输出其不变量误差。

**Primary statistic**：toy trace 的 exact mapping 是否 100% 正确，以及 null invariant 最大误差。

**Null**：按 QA 或完整 hallucination span 置换标签，绝不逐 token 独立打乱。

**PASS 门槛**：

- 已知 trace 对齐 100%；
- `row_mass_max_error <= 1e-6`、`role_mass_max_error <= 1e-6`；
- 无 causal-order violation 和 duplicate edge；
- label-permutation 的真实分数处于置换分布预期范围内，不显著高于 null；
- confirmation 标签未参与 reference/discovery 拟合。

**授权**：只授权继续运行 A1–A10。A0 未通过时，所有科学结果无效。

### A1：直接 evidence/prompt access

**假设**：幻觉 content token 的 evidence/prompt attention mass 更低，response-history mass 更高；该差异不能完全由 token 位置、类型或稀疏缓存覆盖率解释。

**统计量**：

- evidence、question/instruction、response、special/unresolved 的 row mass；
- `Lookback = prompt_mass / (prompt_mass + response_mass)`；
- response takeover、position-conditioned matched \(d_z\)；
- nuisance-only 与 nuisance+role-mass 的 grouped AUPRC。

**Null/control**：在同一 QA 内，按 causal position、content/entity/number 类、回答阶段和 known attention mass 匹配 correct token；另做完整 span circular shift。

**PASS 门槛**：通用 gate，且 nuisance+role-mass 相对 prevalence/nuisance baseline `Delta AUPRC >= 0.02`。

**授权**：只授权 Lookback/role-mass logistic baseline。若只有 A1 通过而 A2–A9 均不通过，结论应是“简单来源比例已经足够”，停止开发 GNN。

### A2：精确 endpoint identity

**假设**：在保持每个 target 的总 attention、source role 和粗粒度 lag 后，具体看向哪个 token 仍然重要。

**统计量**：source concentration、top-1 source reuse、evidence anchor agreement、exact-endpoint feature 相对 A1 baseline 的 `Delta AUPRC`。

**目标约束 endpoint-swap null**：对同一 sample×layer×head×source-role×lag/position-bin×weight-bin 内两条不同 target 的边交换 source，权重留在原 target row。只接受满足 `source < target`、不产生重复边的交换。输出：

- `changed_fraction`；
- `row_mass_max_error`；
- `role_mass_max_error`；
- source degree/strength error；
- `(layer, head, lag-bin, source)` 分层计数误差；
- `causal_violations`；
- `coarse_lag_violations`；
- `duplicate_edges`。

当前 pilot 实现只在 response-history 边内匹配 `(layer, head, coarse log2-lag bin)`，保持 row mass、response role 和非加权 source degree，并逐 replicate 验证 coarse lag 与分层 source count；它不保持 weighted source strength，也没有 mixing/均匀性证明。因此当前输出只把 ensemble rank 写成 `endpoint_null_exceedance_rate`，不参与 BH 或 gate。最新两条 QA engineering smoke 的 coverage 约为 0.139、0.162，并不是总体估计，低于待预注册的计划阈值 0.70。实际顶层 A2 decision 因总 A0 未完成而是 `BLOCKED_BY_A0`；若只检查 pilot null 质量则为 `INCONCLUSIVE_NULL_INVALID`。正式实现应另行设计 degree/weighted-strength preserving double-edge-swap/MCMC，并报告 burn-in、mixing、coverage 与不变量校准，不能把当前 pilot 冒充本段目标 null。

**PASS 门槛**：`changed_fraction >= 0.70`，质量不变量通过，exact-endpoint 特征相对 A1 `Delta AUPRC >= 0.01` 且 CI 全部大于 0。

**授权**：exact token graph。若失败，只保留 role/position/lag 聚合图，不保留精确 token-to-token edge 身份。

### A3：Head fracture

**假设**：幻觉前，不同 heads 对 evidence/response/source endpoint 的分工或分裂程度包含超越 head mean 的信息。

**统计量**：head 间 typed-route pairwise Jensen–Shannon/Hellinger distance、head dispersion、exact-anchor agreement，以及这些量相对 head-mean baseline 的增量。

**Null**：在 task、position、token class 和 mean role-mass 匹配的 token 之间置换完整 head-residual profile。不能只在单个 token 内重排 head 标签，因为对称的 pairwise disagreement 对这种重排严格不变，构不成有效 null。

**PASS 门槛**：在首个幻觉 onset 之前仍有 `|d_z| >= 0.20`，并且相对 mean role-mass baseline `Delta AUPRC >= 0.01`、CI 大于 0。

**授权**：head-resolved 或 permutation-invariant head-set aggregator。若失败，后续模型先对 heads 求均值，避免无依据地增加参数。

### A4：Layer order

**假设**：结构信号取决于 layer 的有序演化，而非最终层、各层均值或无序集合。

**统计量**：有序 slope、trajectory area、first threshold crossing，以及 ordered sequence 相对 final-layer/layer-mean 的 `Delta AUPRC`。

**Null**：对同一个样本的 layer matrices 随机排列后，从头重新执行相同递归；不能只打乱已经汇总完成的 layer feature。另设 layer-collapsed baseline。

**PASS 门槛**：ordered sequence 相对最强无序/汇总 baseline `Delta AUPRC >= 0.01`，CI 大于 0，并在至少 4/5 grouped folds 同方向。

**授权**：GRU、SSM 或显式跨层递归更新。若失败，只用 layer pooling；hidden=16/32/96 等跨层隐状态没有结构依据。

### A5：Multi-hop lineage proxy

**假设**：当前 response token 对 earlier response 的依赖，经过多跳路由后能否追溯到 evidence，包含超越直接 attention 或单跳邻居的信号。

**统计量**：在固定的 attention-only 动态规划中分别计算 direct、1-hop、2-hop 和 full-depth evidence lineage，以及经 response 中转仍无法追溯至 evidence 的 detached-via-response mass。

传播时必须同时保存 known mass 与 unresolved mass。由于当前 cache 没有 value/output/residual 信息，不把 attention 对角线当作 residual，也不把路径乘积称为真实贡献。Attention rollout/flow 的动机来自 [Abnar & Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385/)，但本项目只能验证 attention-routing 版本。

**Null/control**：每层独立做 A2 的约束 endpoint rewire；另做 path-layer order shuffle，并与 direct Lookback、1-hop 路径比较。

**PASS 门槛**：2-hop 或 full-depth 相对 1-hop `Delta AUPRC >= 0.02`、CI 大于 0；真实增量在 topology null 下显著消失；最短通过深度在 discovery 冻结并在 confirmation 复现。

**授权**：多跳 message passing，传播深度取“最短通过深度”。若失败，不使用多层 GNN；最多保留 direct/1-hop 特征。

### A6：Aggregation semantics

**假设**：邻边求和并不一定是正确聚合；极强单一 source、来源集中度或少数 top-k 路径可能更重要。

**候选统计量**：

- typed sum；
- normalized mean；
- max；
- top-k mass；
- Herfindahl–Hirschman index（HHI）；
- effective source count；
- evidence/response 分组后的上述量。

**Null/control**：固定 topology 后在匹配 edge strata 内打乱 weights；以及固定 target row mass 后打乱 source assignment。每个候选均相对 typed sum 比较。

**PASS 门槛**：候选聚合相对 typed sum `Delta AUPRC >= 0.01`、CI 大于 0，并在 weight/topology 对应 null 下失效。

**授权规则**：

- 只有 sum 通过：使用 typed sum；
- max/集中度/top-k 通过：使用 `sum +` 对应通过量；
- 所有候选均无增量：不使用 learned message aggregator；
- 不能因为神经网络可表达某函数，就在没有 gate 的情况下全部加入。

### A7：Response-history lock-in

**假设**：首个幻觉发生后，模型对错误 response history 的路由依赖会持续增强，形成可测量的 lock-in，而非回答自然变长造成的假象。

**统计量**：post-onset response mass、unsupported-lineage proxy、exact-source reuse run length、source concentration 和 persistence length。

**Null/control**：

- 为长度和任务匹配的 all-correct response 采样 pseudo-onset；
- 在同一回答内 circular-shift hallucination span；
- 条件化总 response mass，单独检验 exact reuse/concentration；
- 使用 A2 endpoint rewire 检验身份特异性。

**PASS 门槛**：post-onset matched `d_z >= 0.30`，效应至少持续 3 个 content tokens，显著强于 all-correct pseudo-onset，且 `q < 0.05`。

**授权**：带 hysteresis/gate 的 temporal state update。若仅总 response mass 变化而身份/持续性不通过，只保留 position-conditioned scalar feature。

### A8：Onset / change point

**假设**：由 A1–A7 discovery 阶段冻结的单一结构分数，在第一个 hallucinated content token 之前出现可检测变化。

**统计量**：对冻结分数运行单 change-point 或 CUSUM；报告 `lead = first_hallucination_position - estimated_change_position`、response-level hit rate、all-correct FPR 和 detection coverage。change-point 算法可用 [ruptures 官方文档](https://centre-borelli.github.io/ruptures-docs/)中明确的离线算法实现，但算法、cost 和 penalty 都必须在 discovery 冻结。

**Null/control**：长度/任务匹配的 all-correct pseudo-onset、同回答 span circular shift，以及冻结分数的 stationary matched null。

**PASS 门槛**：confirmation 上 all-correct FPR `<= 10%`；至少 60% 的 hallucination-containing responses 在区间 `[-4, 0]` 个 content tokens 内检测到 change point；median lead `>= 1` 个 content token；优于 matched null。

**授权**：在线 change-point detector。若变化只发生在错误 token 当下或之后，只能声称 concurrent/post-error detection，不能声称 early warning。

### A9：非线性联合形式 screening

**假设**：通过前述 gate 的关系之间存在稳定交互，不能由加性非神经模型充分表达。

**模型阶梯**：

1. nuisance-only logistic；
2. 最强单一审计族；
3. 所有通过审计族的 main-effect additive logistic；
4. discovery 预注册的 interaction logistic；
5. 可选 GAM 作为平滑但可解释的非神经上界。

只允许使用 A1–A8 已通过的 feature family。interaction 候选必须在 discovery 内通过 grouped nested CV 选择；confirmation 只评估一个冻结模型。

**Null/control**：sample-level label permutation；在 task、token class 和 causal-position strata 内做 restricted feature permutation，以区分真实交互和共同 nuisance。

**PASS 门槛**：additive 相对最强单一族 `Delta AUPRC >= 0.02`；interaction 相对 additive `>= 0.01` 且 CI 大于 0；入选交互方向在至少 80% outer folds 一致。

**授权**：只有 A9 通过才授权联合 GNN/GRU 学习通过关系的联合形式。若 A9 失败，最终方法应优先采用 logistic/GAM，而不是用神经模型追求不可解释的小幅拟合增益。

### A10：Base-LLM causal validation

**假设**：非神经结构分数排名靠前的 layer/head/source-target 路径，确实承载影响事实输出的模型内部状态。

**设计**：对同一 QA 构造 clean/corrupt pair。优先使用语法和语义分布尽量匹配的 evidence/entity replacement；在 corrupt run 上 patch clean activation，并比较事实 token 的 logit difference。设计遵循 causal tracing/activation patching 的基本框架，参考 [ROME](https://proceedings.neurips.cc/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) 与 [Activation Patching Best Practices](https://openreview.net/forum?id=Hf17y6u9BC)。

**统计量**：

- normalized recovery `R = (patched - corrupt) / (clean - corrupt)`，只对 clean/corrupt manipulation 成功改变目标事实 logit 的 pair 定义；
- 结构分数与 patch effect 的 sample-level Spearman correlation；
- targeted patch 与 matched-random patch 的差。

**Null/control**：同 layer、head、position、source type、lag 和 patch norm 的随机路径；并分别报告 mean、zero、resample ablation，防止把 out-of-distribution 破坏误当作机制证据。

**PASS 门槛**：Spearman `rho >= 0.30` 且 CI 大于 0；targeted recovery 比 matched random 高至少 0.05，且均值约为其 1.5 倍以上；非目标事实/logit 的 specificity control 不出现同量级泛化扰动。

**授权**：只对通过的关系、模型和任务使用有限定的 causal/mechanistic claim。A10 只在 A2 或 A5 通过后运行。

## 7. 30 条 smoke 与目标正式全量参数

### 7.1 参数对照

| 项目 | 30 条 smoke | 目标正式 discovery/confirmation（待 A0 完成） |
|---|---|---|
| 目的 | 验证读取、对齐、内存、null 能运行、结果文件完整 | 作科学决策 |
| 样本 | 30 个 QA；尽量 15 个含幻觉、15 个全正确 | 全部符合纳入标准的 QA |
| Split | 100 条 train reference + 前 30 条 test，仅走通 smoke | 完整 train reference；test source groups 50% discovery / 50% confirmation |
| Bootstrap | 20–50 次 | 2,000 次 sample-cluster bootstrap |
| Null repetitions | 10–20 次 | 至少 499 次 |
| 多重检验 | 不执行 | BH-FDR `q < 0.05` |
| CV | 可跳过或 2 folds，仅测代码 | discovery 内 grouped 5×5 nested CV；confirmation 一次 |
| A10 | 关闭 | 仅 A2/A5 通过后，对约 100 个有效 paired QA 运行 |
| Gate 输出 | 全部 `NOT_EVALUATED_SMOKE` | `PASS` / `FAIL` / `INCONCLUSIVE` |
| 可发表性 | 无 | 满足本计划和功效要求后才可引用 |

smoke 中“有正确有错误”只用于确保两个代码路径都被覆盖，不是代表性抽样。不得用 30 条结果选择最终 claim、宣布 gate 通过或修改 confirmation 阈值。

### 7.2 目标正式运行的冻结顺序

当前应在第 3 步停止；未完成 A0b/A0c 前，不应启动 499-replicate confirmation。

1. 冻结样本纳入/排除标准和 grouped split IDs。
2. 在 reference 上拟合无标签归一化、matching bins 和 known-mass strata。
3. 运行 A0；失败则停止。
4. 在 discovery 上依次运行 A1–A8，冻结每个 audit 的方向、主统计量、窗口、null 和阈值。
5. 只让已经 PASS 的 feature family 进入 A9；冻结唯一联合模型。
6. 锁定配置、代码 commit、数据 manifest 和随机种子。
7. 一次性打开 confirmation，产生 gate table；不回到 discovery 改规则。
8. 仅当 A2/A5 通过时启动 A10。
9. 若 confirmation 失败，报告失败或低功效；不能把 confirmation 重新命名为 discovery 后继续试到成功。

## 8. 目标结果表与可审计输出

论文级完成稿要求每个 audit 至少输出一行机器可读记录。当前实现仍把 decision、relation、temporal 和 CV 分散在多个表中，尚未满足这一输出契约：

| 字段 | 含义 |
|---|---|
| `audit_id` | A0–A10 |
| `split` | smoke/reference/discovery/confirmation |
| `primary_metric` | 冻结的唯一主指标 |
| `effect` / `ci_low` / `ci_high` | sample-cluster 效应与 95% CI |
| `delta_auprc` / `prevalence` | 增量性能与基准率 |
| `p_value` / `q_value` | 原始与 BH 校正值 |
| `null_name` / `null_repetitions` | null 及次数 |
| `null_valid` | 不变量是否满足 |
| `fold_direction_count` | 同方向 folds 数 |
| `gate_status` | PASS/FAIL/INCONCLUSIVE/NOT_EVALUATED_SMOKE |
| `authorized_module` | 本 gate 实际授权的模块；失败时为 none/简化版本 |

图表以 QA 为统计单位展示 CI；token 数只能作为描述性样本量。必须同时报告 micro token 指标、QA-macro 指标和 position/token-class matched 指标，避免长回答主导结论。

## 9. 从 gate 到模型的停止规则

| 已通过的最高结构 gate | 后续模型上限 |
|---|---|
| 仅 A1 | role-mass/Lookback logistic |
| A2 | exact endpoint feature 或浅层 token graph |
| A3 | head-resolved/set aggregation |
| A4 | 有序跨层 GRU/SSM；否则 layer pooling |
| A5 | 最短通过深度的 multi-hop message passing |
| A6 | 只加入通过的 sum/max/HHI/top-k 聚合 |
| A7 | gated/hysteretic response-history state |
| A8 | 独立在线 change-point 模块 |
| A9 | 联合神经模型学习已通过关系的交互 |
| A10 | 对通过路径作有限定的因果解释 |

该表是复杂度预算，不是模块购物清单。某个 gate 未通过，就从模型中删除对应自由度。

## 10. 论文级尚未实现或尚未验证的项目

以下内容在形成论文级结论前必须补齐。即使仓库中已有原型，也应在正式 run manifest 中标为“已实现且通过验证”后才能移出本清单。

- [ ] evidence / question / instruction 的可靠 prompt parser；否则只能分析粗粒度 `prompt` 来源。
- [ ] tokenizer offset、RAGTruth span、generation query shift 的 gold toy trace 测试。
- [ ] 完整 pipeline 的 sample/span-level label-permutation leakage sanity；现有 circular-shift association test 不能替代 A0c。
- [ ] value projection、output projection、residual stream 和 FFN activation 的采集。
- [ ] residual-aware rollout 或 ALTI 类贡献权重；当前 raw-attention 路由不能替代它。[ALTI](https://aclanthology.org/2022.emnlp-main.595/)明确强调从 attention 权重转向 token contribution。
- [ ] 对 sparse/top-k cache 的 unresolved mass、上下界和 sparsification-threshold 敏感性分析。
- [ ] 更强的 degree/weighted-strength preserving null ensemble，而不只是一种近似 endpoint swap。
- [ ] content/function/punctuation/entity/number 的可复现语义分层与实体/数字 span 对齐。
- [x] reference/test manifest、source groups、score 文件、代码、tokenizer 与 evaluation config 的冻结机制；confirmation 必须使用预冻结 plan。
- [ ] A1–A9 primary hypotheses 的统一多重检验与低功效处理。
- [ ] QA-macro、matched-pair `d_z` 与 token-micro 三套预注册 effect 报告；当前代码只有 token-micro + source-group bootstrap。
- [ ] A9 grouped nested-CV interaction screen 和冻结的 confirmation evaluator。
- [ ] A10 的 paired clean/corrupt 生成、activation/path patching 和 matched-random 干预。
- [ ] 跨随机种子、跨模型规模、跨生成模型、跨 RAG 任务的 replication。
- [ ] 对 attention 保存精度、head/layer 子采样、最大回答长度和压缩参数的鲁棒性分析。
- [ ] 正式功效分析与最小可检测效应；本计划阈值须在 confirmation 前固定。
- [ ] 论文主表、null sanity figure、onset 时间图、失败 gate 和负结果的完整报告。

## 11. 必须公开的限制

1. Raw attention 不是信息贡献，更不是因果解释；[Attention Flow](https://aclanthology.org/2020.acl-main.385/)只提供跨层路由近似，ALTI 类方法需要更丰富的内部量。
2. 若 cache 缺少 value/output/residual/FFN，只能报告 attention-routing proxy。
3. Attention matrix 的 diagonal 不等于 residual connection，不能直接当作“保留自身信息”。
4. 若 prompt 未可靠拆成 evidence/question/instruction，只能声称 prompt provenance，不能声称 evidence grounding。
5. 稀疏 cache 会产生 censoring；必须报告 unresolved mass，不能把未保存权重当作 0 后忽略。
6. 字符 span 到 subword token 的映射存在边界歧义，boundary token 必须单列。
7. 同一回答内 token 强相关；逐 token IID 检验会夸大显著性。
8. pooled token 指标会被长回答主导，因此必须同时报告 QA-macro 和 matched 结果。
9. 与首个错误 token 同时发生的变化可能是错误结果而非错误前因；只有正 lead 才能支持 early warning。
10. 约束 endpoint swap 只能近似保持 weighted source strength；null 不变量和 changed fraction 必须公开。
11. discovery 使用标签选方向和交互，因此 discovery 分数不能作为最终确认结果。
12. 单一模型、任务、数据集或 sparsification threshold 上的结果不能自动泛化。
13. A9 的预测性交互不等于模型机制；只有 A10 能提高到有限定的因果主张。
14. 30 条 smoke 只验证工程路径，不提供任何科学证据。

## 12. 主要一手来源

- Abnar, S. & Zuidema, W. (2020). [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/). ACL.
- Ferrando, J. et al. (2022). [Towards Opening the Black Box of Neural Machine Translation: Source and Target Interpretations of the Transformer](https://aclanthology.org/2022.emnlp-main.595/). EMNLP（ALTI）。
- Meng, K. et al. (2022). [Locating and Editing Factual Associations in GPT](https://proceedings.neurips.cc/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html). NeurIPS（ROME）。
- Zhang, F. & Nanda, N. (2024). [Towards Best Practices of Activation Patching in Language Models](https://openreview.net/forum?id=Hf17y6u9BC). ICLR.
- Adebayo, J. et al. (2018). [Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html). NeurIPS.
- Chuang, Y.-S. et al. (2024). [Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps](https://aclanthology.org/2024.emnlp-main.84/). EMNLP.
- Zhang, M. et al. (2024). [How Language Model Hallucinations Can Snowball](https://proceedings.mlr.press/v235/zhang24ay.html). ICML.
- Wu, Y. et al. (2026). [CHARM: Hallucination Detection and Mitigation through a Cross-Layer Attention-Guided Reasoning Mechanism](https://openreview.net/forum?id=4twbqwV4br). ICLR.
- Niu, C. et al. (2024). [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://aclanthology.org/2024.acl-long.585/). ACL.
- Ojala, M. & Garriga, G. C. (2010). [Permutation Tests for Studying Classifier Performance](https://www.jmlr.org/beta/papers/v11/ojala10a.html). JMLR.
