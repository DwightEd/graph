# Round 1 Independent Method Review

## 替代审查方式

Codex MCP 当前不可用；本文件作为 `research-refine` Phase 2 的替代审查，由当前独立 GPT-5.5-xhigh-style reviewer 按同一 7 维 rubric 完成。审查只写入本文件，未更新 `REFINE_STATE.json`、manifest 或其它日志文件，以符合本次任务的“只读并写该审查文件”约束。

审查输入：

- `refine-logs/round-0-initial-proposal.md`
- `docs/EXPERIMENT_HISTORY.md`
- 参考核对：CHARM、TOHA、Geometric Scattering、Attention Rollout、factual recall / non-factual hallucination 机制论文的主页面。

## 总体判断

方案没有把目标漂移成可视化，也基本保留了真实来源/路径 × 候选 × 层 × 通道结构；它也遵守了三条硬约束：不训练新神经检测器、不用幻觉标签构图或调分、不把模型消融当成建图前提。最有价值的想法是把“某条边看向哪里”与“那个端点对某个候选是否可读”相乘，再用角色条件零模型扣掉角色质量和 self mass。

但是当前版本还没有到 READY。主要问题不是工程，而是统计对象尚未钉死：当前零模型只保留角色质量和 self，未保留 lag/source/cardinality，因此 residual 可能仍是位置、源分布或端点数量代理；候选状态 `X` 的定义也不足够精确，容易退化成 top-1 margin / uncertainty 的换名。若这两点不修，论文贡献会被审稿人读成“attention rollout + logit lens + role permutation null”，而不是新的候选条件路径残差。

Verdict: **REVISE**

Overall score: **7.1 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8.5 | 紧贴 anchor，明确不训练检测器、不用标签选图、不靠消融建图；唯一轻微风险是 candidate top-K 与真实 factual commitment 的对应关系不稳定。 |
| Method Specificity | 7.0 | 公式和数据流已经可实现，但 `X` 的候选 margin、`d=2` 路径方向、mask 插入位置、缺失角色/空 bin 处理还不够精确。 |
| Contribution Quality | 6.5 | 核心机制有潜力，但目前与 attention rollout、TOHA/attention topology、logit-lens residual 的边界还靠叙述而不是强零模型和强对照来建立。 |
| Frontier Leverage | 7.5 | 不强行加入 LLM/VLM/RL 是正确选择；使用冻结 LLM computational trace 是现代且合适的。无需新增神经训练。 |
| Feasibility | 6.5 | 本地冻结模型 + attention/hidden states 抽取可做，但全层全头全 token 的内存、source-disjoint reference 稀疏性、candidate omission 会成为真实瓶颈。 |
| Validation Focus | 8.0 | 三个验证块聚焦、可证伪，且吸取了历史负结果；需要把“成功/失败门槛”写得更硬。 |
| Venue Readiness | 6.0 | 若 residual 在强零模型下胜出，会有 paper 形状；当前版本还不足以抵挡“这只是另一种手工 trace feature”的质疑。 |

## 最关键修订

### 1. 把贡献句改成可被证明或击穿的单句

当前“候选条件的路径残差”听起来对，但还不够锋利。建议改成：

> 在冻结生成轨迹中，候选可读状态沿真实 attention 端点传播后，相对于保留 role/self/lag/source 条件的端点置换零模型所产生的 signed residual，能提供超越角色质量、位置、margin、entropy 和 topology-only 分数的无监督异常信号。

这句话把新意放在三个点上：候选条件、端点条件零模型、相同读出下的增量检测。不要把主贡献写成“路径图可视化”或“多跳 attention graph”，因为这些已经被 TOHA、CHARM、attention rollout 和 graph scattering 相关工作挤压得很窄。

Priority: **CRITICAL**

### 2. 零模型需要从 role-only 升级成分层条件零模型

当前 `N` 精确保留每行的角色质量、自注意力和因果 mask，这是好的第一步，但不够。它没有保留：

- source group / passage identity；
- log-lag 或局部/远程历史距离；
- role 内候选端点数；
- special/BOS 附近的异常集中；
- source/history 内部的 message 边界。

因此 `observed - null` 可能主要是在说“真实 attention 更偏向近端/某个 source/某类 source token”，而不是“边端点与候选状态异常对齐”。建议预注册三个零模型，按强度递进：

1. `N_role`: 保留 role mass + self mass + causal mask。
2. `N_role_lag`: 在 `role × log2(i-j)` bin 内均匀置换端点。
3. `N_source_lag`: 在 `role × source_group × log2(i-j)` bin 内置换端点；bin 太小则按预声明规则合并或标记 missing。

主 claim 只能基于 residual 在 `N_role_lag` 或 `N_source_lag` 下仍然优于 observed/null/signal/position/margin。`N_role` 可作为解释性弱零模型，但不能承担论文主证据。

Priority: **CRITICAL**

### 3. 明确候选和 `X`，否则检测对象会滑成 uncertainty

“使用冻结模型 final norm 和输出 embedding 方向构造 top-1 对其余候选的余弦差 X”还不够可复现。需要写成一个不可歧义的张量定义，例如：

```text
C_q = top-K logits at predictor position q before token t is sampled
x_l(i,c) = cosine(LN(h_l[i]), W_U[c]) - mean_{c' in C_q \ {c}} cosine(LN(h_l[i]), W_U[c'])
```

或者使用 unembedding logit margin，但必须二选一并固定。还要记录：

- generated token 是否在 `C_q`；
- factual answer token / source answer alias 是否在 `C_q`，如不可得则只报告 coverage，不回填；
- `K=2/4/8` 的 coverage 与分数稳定性；
- candidate rank 的 signed residual，不提前把正负号或 rank 合并。

若正确候选经常不在 top-K，主方法最多能检测“模型自身候选竞争异常”，不能声称检测事实约束使用失败。

Priority: **CRITICAL**

### 4. 把 `d=1` 和 `d=2` 的算子写成完整方向公式

建议把路径定义改成显式版本，避免实现时把 mask 放反：

```text
R1(l,h,r,c) =
  e_q^T (P_lh - Q_lh) M_r x_l(:,c)

R2(l,h,r->s,c) =
  e_q^T P_lh M_s P_{l-1,mean} M_r x_{l-1}(:,c)
  - e_q^T Q_lh M_s Q_{l-1,mean} M_r x_{l-1}(:,c)
```

对 `source -> history -> query`，`r=source, s=history`，并显式排除 `q` 自身和 future positions。这样 `d=2` 不是“多乘一次矩阵”的任意复杂化，而是一个可撤回的中继 claim。若 `R2` 不稳定优于 `R1`，中继贡献应撤回，保留端点对齐 residual。

Priority: **IMPORTANT**

### 5. 降低无监督读出的自由度

当前 layer × head × path × candidate 的维度很容易让 kNN/MAD reference 吸收 source-specific quirks。建议主读出只用一个预声明的固定向量：

- path: `source`, `history`, `source->history`；
- candidate rank: top-1, top-2, maybe top-4 aggregate；
- layer bands: early/mid/late 或 all-layer robust average，不能按标签选层；
- head: 保留 head 维度用于诊断，但主分数使用固定 aggregation，或报告 “all-head” 与 “copy-head prior” 两个 label-free 版本。

如果坚持 kNN，必须同时报告一个低自由度 robust-z score：

```text
score = mean_j z_j(residual_feature)^2
```

并证明 kNN 的增益不是高维 reference artifact。

Priority: **IMPORTANT**

## 是否只是换名字？

当前版本处于危险边缘，但不是必然失败。它区别于已有工作的潜在新意是“候选状态 × 具体端点连接”的条件 residual，而不是 attention topology 本身。

主要邻近工作压力如下：

- CHARM 已把 token 节点、attention 边、activation features 组织成 supervised graph learning；因此“attention graph + activations 有用”不是新贡献。
- TOHA 已做 attention graph topology divergence，并且有少标注/无标注选头变体；因此“无监督 attention 拓扑检测”也不是新贡献。
- Attention rollout/flow 已做跨层 attention 乘积；因此 `P_l ... P_{l-d}` 不是单独的新意。
- Graph scattering 已覆盖固定图滤波/多尺度传播的一般思想；因此“固定算子 + kNN”不能单独成为强创新。

要避免换名，方案必须证明：在同一候选状态、同一 role/source/lag 条件、同一无监督读出下，真实端点连接提供了 observed/null/signal/position/margin 之外的增量信息。

## 最致命可证伪点

1. **强零模型击穿**：如果 residual 只在 `N_role` 下有效，而在 `N_role_lag` 或 `N_source_lag` 下消失，则方法不是连接端点创新，只是 recency/source allocation 代理。
2. **候选遗漏击穿**：如果 factual-relevant candidate 大量不在 top-K，且分数主要跟 top-1 margin 或 entropy 相关，则方法检测的是不确定性，不是来源约束使用。
3. **路径阶数击穿**：如果 `d=2 source->history` 不优于 `d=1`，必须撤回中继机制，不能用“多跳路径”包装结果。
4. **位置混杂击穿**：若 residual 与 absolute/relative position 的 Spearman 仍接近历史 rupture 分数水平，或 source bootstrap 下对 position/margin 没有稳定增量，则主 claim 失败。
5. **source-disjoint 稀疏击穿**：若 reference split 中每个 task/source/layer 条件的样本不足，MAD/kNN 会不稳定；此时必须报告 failure/coverage，而不是跨任务回退。

这些失败都不要求训练新模型或做模型消融，可以直接在冻结 trace 上验证。

## 实现可行性判断

可以实现，但需要先做 extraction contract，而不是先重构大系统。最小可实现路线：

1. 固定 predictor position `q = response_start + t - 1`，验证 t=0、future leak、生成 token coverage。
2. 只抽取 scoring positions 的 attention rows 和 layer-input states，避免默认保存全 `T×T×L×H`。
3. 生成 `C_q`、`x_l(i,c)`、role/source/lag bins、`P/Q` residual features。
4. 序列化 frozen score table，再 join labels。
5. 报告 residual vs observed/null/signal/position/margin/entropy 的同 split 对照。

内存风险是真实的，但不是阻断项；阻断项是候选和零模型不够严。

## Simplification Opportunities

1. 删除“谱特征值”作为主要论证，只保留一句数学动机：下三角 attention 的原始 eigenvalue 不足以表达端点路径，因此读 operator action。不要把它写成对所有谱/topology 方法的否定。
2. 主实验先只做 `d=1` 和一个严格定义的 `source->history` `d=2`，不要引入更多路径阶数。
3. 主分数先用 robust-z fixed feature；kNN 放 secondary。这样能更清楚地说明 residual 本身是否有价值。

## Modernization Opportunities

NONE as new trainable components. 本方案不需要再加 GNN、autoencoder、VLM/RL 或 LLM judge。现代性来自冻结 foundation-model trace 的候选条件读出。若要借鉴 frontier-era practice，最多加入 label-free prior，例如 copy-head prior 作为预注册 secondary variant，但不能用标签选头。

## Drift Warning

NONE, as written. 方案仍然解决 anchor：从自然生成轨迹读取连接关系结构用于无监督幻觉检测。

需要避免两种未来漂移：

- 漂成 `docs/METHOD.md` 中的 factorial counterfactual graph 复活版；那是另一路线，可作历史基线或后续 causal validation，不应成为当前建图前提。
- 漂成 attention/path visualization；主输出必须是 frozen score table 和 source-disjoint detection result。

## Primary-source spot check

- CHARM: https://arxiv.org/html/2509.24770v2 。主张是 attributed attention graphs + GNN hallucination detection，支持“图结构有用”，但不是无监督路径 residual。
- TOHA: https://arxiv.org/html/2504.10063 。主张是 attention graph topological divergence，并包含少标注/无标注 head 选择变体；它压缩了“无监督 attention topology”作为新颖性的空间。
- Geometric Scattering: https://proceedings.mlr.press/v97/gao19e.html 。固定图散射/多尺度图特征早已有一般框架。
- Attention rollout/flow: https://aclanthology.org/2020.acl-main.385/ 。跨层 attention 乘积作为信息流近似已有先例。
- Factual recall: https://aclanthology.org/2023.emnlp-main.751/ 和 https://aclanthology.org/2024.findings-emnlp.466/ 。支持把候选可读状态、attention extraction 和 factual failure 区分开；也提醒本方法不能把 logit-lens proxy 直接宣称为 causal knowledge use。

## Action items before Round 1 refinement

1. 重写方法 thesis：主贡献必须是 candidate-conditioned endpoint residual under strong conditional null。
2. 增加 `N_role_lag` / `N_source_lag`，并声明主 claim 以强零模型为准。
3. 给 `X`、`C_q`、candidate coverage、rank/sign 处理写精确定义。
4. 给 `R1/R2` 写方向明确的公式，尤其是 `source->history->query`。
5. 预注册低自由度主分数和 kNN secondary 分数。
6. 把失败门槛写进方案：强零模型、candidate omission、position correlation、`d=2 <= d=1` 时分别撤回哪些 claim。

Final verdict: **REVISE**, not RETHINK. 核心想法值得继续，但 READY 需要先把零模型和候选定义收紧；否则最强反驳会非常直接。
