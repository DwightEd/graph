# Round 2 Independent Method Review

## 替代审查方式

Codex MCP 当前不可用；本文件继续作为 `research-refine` 的独立 GPT-5.5-xhigh-style 复审替代。审查输入为 `refine-logs/round-1-refinement.md`、`refine-logs/round-1-review.md` 与 `docs/EXPERIMENT_HISTORY.md`。本轮只写入 `round-2-review.md`，未修改其它文件。

## 总体判断

第二轮修订实质性解决了第一轮的核心风险。方案现在明确承认“无自然性能实验证据”是当前状态，没有把 tiny Llama / 软件测试包装成检测有效性；也没有漂移成可视化、factorial counterfactual 或新神经检测器。强零模型从 role-only 收紧为 `role × source_unit × loglag`，候选 cosine contrast 有明确公式，`R1/R2/R2relay` 有显式方向，主读出改成 robust-z，kNN 降为补充。这些都是正确修订。

仍不能给 READY。剩余 blocker 不是“需要先跑自然结果”，而是两处数学/实现规格在进代码前必须钉住：一是 `R2relay = observed product - null product` 不能单独证明 relay 交互，可能只是两段一阶 residual 的叠加；二是 `mean(z²)` 的 MAD-only scale 在常数/近常数特征上会制造数值异常，需要明确剔除或 fallback。其余问题多属于研究待证：强零模型下是否仍有增量、top-K 候选覆盖是否足够、最终层 160 维是否会被 reference 稀疏性限制。

Verdict: **REVISE**

Overall score: **8.1 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 9.0 | Anchor 保持得很好：无新训练、无标签构图、不靠模型消融、保留真实路径/候选/层/head 结构；也明确软件通过不等于研究成功。 |
| Method Specificity | 8.3 | 张量、零模型、路径算子、读出、数据边界都比上一轮清楚；还缺 R2 relay 增量定义、常数特征 scale 策略和若干序列化不变量。 |
| Contribution Quality | 7.8 | 已从“attention + logit lens 换名”推进到 candidate-conditioned endpoint residual under strong null；但只有自然数据和 hybrid relay control 才能支撑真正贡献。 |
| Frontier Leverage | 8.0 | 使用冻结 foundation-model trace 合理，不需要再引入 GNN/AE/LLM judge。 |
| Feasibility | 7.4 | 实现路径可行，最终层 160 维也比全层读出克制；稠密 attention、source-disjoint reference 稀疏和 singleton groups 仍是工程/适用性风险。 |
| Validation Focus | 8.5 | 验证块现在非常清楚，且失败时撤回哪些 claim 写得较诚实。需要补一个 relay hybrid control 和 constant-feature diagnostic。 |
| Venue Readiness | 7.0 | 方法定义已经像一篇可验证论文，但没有自然性能证据，且 relay/scale 两个规格缺口未关。 |

## 接受的修订与反驳

第一轮说“role 内 cardinality 未保留”不准确。当前修订说明组内节点集合/数量本来由条件组保留，`N` 只在非空组内均匀重分配原质量；singleton 组 `N=A`。这个反驳成立。

第一轮关于 lag/source 混杂的批评已经被主要吸收。新的零模型保留 source unit、log-lag、role、自注意力和因果 mask，因此 residual 的目标已从“source allocation 是否异常”收窄为“在同 source/loglag/role 条件内，真实端点与候选状态的对齐是否异常”。这更窄，但更干净。

## 代码前必须修的数学/规格问题

### 1. `R2relay` 不能直接解释为 relay 交互

当前：

```text
R2relay =
  e_q^T P_lh M_Hminus P_(l-1,mean) M_S x
  - e_q^T Q_lh M_Hminus Q_(l-1,mean) M_S x
```

这是两步 observed product 与两步 null product 的差。它会同时吸收：

- query -> history 这一步的一阶 endpoint residual；
- history -> source 这一步的一阶 endpoint residual；
- 两步共同偏离 null 的交互项。

因此，如果 `R2relay` 有增益，不能直接宣称发现了 `source -> history -> query` 中继机制。进代码前建议至少保存两个 hybrid terms：

```text
H_query = e_q^T P_lh M_Hminus Q_(l-1,mean) M_S x
H_source = e_q^T Q_lh M_Hminus P_(l-1,mean) M_S x
```

并定义可选的 relay interaction：

```text
I_relay =
  P_lh M_Hminus P_(l-1,mean)
  - P_lh M_Hminus Q_(l-1,mean)
  - Q_lh M_Hminus P_(l-1,mean)
  + Q_lh M_Hminus Q_(l-1,mean)
```

主实现可以仍然报告当前 `R2relay`，但命名应是 “two-step source-through-history residual”。只有 `I_relay` 或相对 hybrid controls 的增量稳定时，才能写“relay interaction”。这是数学 claim 边界，不是自然数据待证项。

Priority: **CRITICAL**

### 2. robust-z 的常数/近常数特征规则不够安全

当前主分数用：

```text
z = (feature - median) / max(1.4826 MAD, 1e-6)
score = mean(z^2)
```

这会在结构性零特征、singleton-dominated features、或 float 噪声很小但非零的特征上产生伪异常。进代码前需要明确规则：

- 若 fit 中某 feature 的 nonzero rate 或 unique count 低于阈值，标记 inactive，不进入 `mean(z²)`；
- 或使用 `MAD -> IQR -> std -> inactive` 的 fallback，而不是只有 `1e-6` floor；
- 报告每个条件被 inactive 的 feature 比例；
- `observed/null/residual/signal` 四种表征必须用相同 active mask，避免 residual 因 feature selection 获得优势。

这是实现规格缺口，会直接影响分数，不应留到自然评估后再修。

Priority: **CRITICAL**

### 3. source-unit 强零模型收窄了主 claim，需要在代码和文档中同步命名

由于 null 保留 `source_unit × loglag × role` 的质量，主 residual 不再衡量“模型是否看向正确 source unit”。它衡量的是：在同一个 source unit 和 lag bucket 内，具体端点与候选状态是否异常对齐。

这是合理的设计，但 claim 必须对应收窄。否则审稿人会问：你已经把 source allocation 固定住了，为什么还能宣称发现 source constraint control？建议命名为：

> conditional endpoint-candidate alignment residual

而不是 source-control residual。source-control 只能作为 secondary 解释，或由 weaker null / separate factorial experiment 支撑。

Priority: **IMPORTANT**

### 4. feature order 与 shape 需要成为不可变 contract

默认 160 维为 `5 × heads × (K-1)`，但“5”对应哪些通道必须机器可验证。建议在 score table 中保存：

```text
feature_name = representation/path/depth/layer/head/candidate_rank/null_spec
```

并保存 feature digest。否则 observed/null/residual/signal、R1/R2/R2relay、source/history 通道在后续评估中很容易错位。这是工程规格，但它保护的是数学对象。

Priority: **IMPORTANT**

## 研究待证项，不应在代码前过度修

这些问题需要自然数据结果判断，不能靠继续改定义解决：

1. residual 是否在强零模型下稳定胜过 null/signal/observed、position、entropy、negative margin。
2. `K=2` 的 candidate coverage 是否足够；若不足，`K=4/8` 是否保留信号而不是稀释。
3. 最终层 160 维是否比 layer bands 更好；当前默认最终层是合理的复杂度控制，不必在跑前改成全层。
4. paragraph/空行 source unit 是否足够接近事实证据单元；这是数据适用性问题，先报告 coverage 和 singleton 率。
5. probe model 与原 generator 不同导致的 logit/candidate mismatch 是否削弱检测；这是实验解释边界，不是方法定义错误。

## role-null 与 confound 检查

当前 strong null 已经保留了主要 role-null confound：role mass、source-unit mass、log-lag mass、self mass、causal mask、组内 cardinality。剩余 confound 是诚实可接受的，但要报告：

- exact lag within bucket：`floor(log2(i-j))` 仍可能留下 bucket 内距离差异；
- source unit segmentation：段落规则可能把一个真实证据单元切得过粗或过细；
- special/BOS：special role 虽固定，但历史上已有 BOS confound，需单列 mass/feature dominance；
- singleton rate：singleton 使 `N=A`，会让 residual 局部归零并改变有效特征维度；
- final-layer only：可能混入 late decoding competition，而非中层 factual state。

这些不是 blocker，但必须进 manifest/diagnostics。

## 创新 claim 复审

修订后不再像简单换名。与 CHARM/TOHA/Attention Rollout/Geometric Scattering 的边界已经更清楚：

- 不是 supervised graph classifier；
- 不是 topology-only divergence；
- 不是单纯跨层 attention rollout；
- 不是固定图滤波替代 AE；
- 不是把 logit lens 当因果知识读出。

真正的新颖性仍取决于一个实证事实：在强条件零模型和同一无监督读出下，candidate-conditioned endpoint residual 是否提供增量。如果没有增量，这仍会被归类为一个严谨但负面的 trace analysis。

## Drift Warning

NONE。修订稿保持了原 anchor，也明确不把自然性能缺失粉饰成成功。

需要继续避免：

- 把 `R2relay` 增益自动解释为中继机制；
- 把 source-unit-preserving residual 写成 source selection/control；
- 用 tiny Llama 软件测试暗示真实检测有效；
- 因 kNN secondary 好看而弱化 robust-z 主分数。

## Simplification Opportunities

1. 不要再加入多种弱零模型作为默认实现。保留当前强零模型，最多把 weaker null 用于 appendix/diagnostic。
2. `R2relay` 若短期不加 hybrid control，就只作为 two-step residual，不主张 relay interaction。
3. kNN 继续保持 supplement；主叙事围绕 robust-z residual 是否有增量。

## Modernization Opportunities

NONE。不要加 GNN、autoencoder、LLM judge、VLM 或训练式 detector。当前冻结 trace + 强零模型已经是更合适的 frontier-era primitive。

## Verdict Rule Interpretation

本轮不是因为缺自然性能证据而要求停工；缺自然性能证据是当前诚实状态。真正阻止 READY 的是两个可在代码前修的规格问题：relay interaction claim 与 constant-feature scaling。一旦这两项修掉，方案可以进入自然数据评估；但 scientific READY 仍要等 source-disjoint 自然结果。

Final verdict: **REVISE**。
