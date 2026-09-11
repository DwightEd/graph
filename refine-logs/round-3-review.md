# Round 3 Independent Method Review

## 替代审查方式

Codex MCP 当前不可用；本文件继续作为 `research-refine` 的独立 GPT-5.5-xhigh-style 复审替代。审查输入为 `refine-logs/round-2-refinement.md`、`refine-logs/round-2-review.md` 与前一轮修订稿。只写入本文件，未修改其它文件。

## 总体判断

第 3 轮已经把第 2 轮的两个代码前 blocker 收住了。二步量现在明确命名为 **two-step source-through-history residual**，并承认它包含两段一阶偏离及交叉项，不再宣称 pure relay interaction；这避免了为追求“二阶创新”而增加 hybrid 系统。标准化也从 MAD-only 改成每 view 中位数 center、四 view 参考 std 最大值作共享 scale、同一 active mask、常数坐标忽略；整 stratum 无 active 坐标时显式 `reference_active_features=0` 并给 0 分。这是保守但一致的规则，适合进入自然数据评估。

本轮我没有发现新的数学 blocker。实现规格已经基本具备，可以进入 source-disjoint 自然实验。论文有效性仍未 READY：还没有真实预训练权重上的检测增益，主贡献仍需证明 residual 在强零模型下胜过 null/signal/observed、position、entropy 和 negative-margin。这个缺口不能靠审查分数弥补。

Verdict: **REVISE**

Implementation status: **READY FOR NATURAL EVALUATION**

Scientific / paper status: **NOT READY UNTIL NATURAL RESULTS**

Overall score: **8.5 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 9.3 | Anchor 保持稳定；没有训练新检测器、没有标签构图、没有模型消融前提，也没有把目标降为可视化。 |
| Method Specificity | 8.9 | 零模型、候选信号、路径算子、二步命名、active-mask 标准化和失败标记都足够具体；只剩 manifest/diagnostic 级实现断言。 |
| Contribution Quality | 8.1 | 现在确实是 candidate-conditioned endpoint residual under strong null，不再明显像已有 attention/topology/logit-lens 工作换名；贡献仍要靠自然增量成立。 |
| Frontier Leverage | 8.2 | 冻结 LLM trace + label-free conditional null 是合适的现代工具；不需要再加 GNN/AE/LLM judge。 |
| Feasibility | 7.9 | 默认最终层 160 维、robust-z 主分数、kNN 补充都可实现；稠密 attention、reference 稀疏和 probe-model mismatch 仍是实际风险。 |
| Validation Focus | 8.9 | 验证块清楚，失败时撤回哪些 claim 写得明确；自然实验会直接击穿或支持主假设。 |
| Venue Readiness | 7.5 | 方法定义像一篇可评估论文，但没有主结果前不能宣称 venue-ready。 |

## 已解决的问题

### 1. 二步量的 claim 边界已修正

不新增 hybrid terms 是可接受的，因为方案现在不再把 `R2relay` 叫 pure interaction。当前表述足够诚实：

- 它是 source-through-history two-step residual；
- 它可能包含后一段、前一段及交叉项；
- 用 `d=1` 对照检验二步读取是否增加检测信息；
- 若 `d=2` 无收益，撤回中继贡献。

这不会阻止实现。后续论文里要避免写 “isolates relay interaction” 或 “pure second-order mechanism”。

### 2. 标准化规则已足够保守

共享 scale 取四 view std 最大值，会牺牲敏感性，但避免某个 view 因近零方差被放大。共同 active mask 也避免 residual view 通过选择坐标获得不公平优势。常数坐标被忽略的代价已经记录：测试中才出现的新变化无法被该坐标检测。这个取舍合理，适合先跑自然评估。

## 仍需在实现中断言/报告的规格

这些不是方法 blocker，但建议在长跑前作为 manifest 或测试断言固定下来：

1. `feature_schema` 必须显式保存每一维的 `view/path/depth/layer/head/candidate_rank/null_spec`，不能只保存 digest。
2. 每个 stratum 必须报告 `reference_active_features` 的分布；自然评估至少报告 all-token 结果和 `active_features > 0` 覆盖率。
3. scale 和 active mask 必须只由 reference split 计算，不能在测试 token 上更新。
4. 四个 view 必须使用同一 active mask；若实现为了 kNN 使用不同坐标，应作为 secondary score 单独命名。
5. singleton 组比例、special/BOS dominance、source-unit 数量、每 stratum 参考来源数应随 score table 一起输出。

## paper 有效性待自然实验

以下问题不是继续改规格能解决的，必须靠 source-disjoint 自然数据判断：

1. residual 是否稳定胜过 observed/null/signal、position、entropy、negative margin。
2. 强 source-unit×loglag×role null 是否过强，导致真实 source allocation 信号被完全扣掉。
3. `K=2` 候选覆盖是否足够；`K=4/8` 是否保留方向而不是稀释。
4. two-step residual 是否相对 `d=1` 有增量；若没有，应撤回中继贡献。
5. probe Llama 与原 generator 不一致是否使 candidate/top-logit 信号偏离真实生成机制。
6. paragraph/空行 source unit 是否足够作为事实证据单元。

这些待证项会决定 paper 是否成立，但不阻止当前实现进入评估。

## role-null 与 confound 复查

当前 role-null 已经保留 role mass、source-unit mass、loglag mass、self mass、causal mask 和组内 cardinality。剩余 confound 已被正确降级为 diagnostics：exact lag within bucket、source segmentation、BOS/special、singleton rate、final-layer-only late decoding competition。这里没有新的数学错误。

需要注意 claim 的收窄：强零模型保留 source-unit allocation 后，主结果只能说“同 source/loglag/role 条件内，具体端点与候选状态的条件对齐有异常信号”。不要把它写成“发现模型选择了错误 source unit”。source selection/control 只能由 weaker diagnostic 或其它实验支持。

## 创新判断

修订后的新意足够清楚：它不是 supervised graph classifier、不是 topology-only divergence、不是 attention rollout 本身、不是 graph scattering，也不是把 logit-lens 当 causal knowledge。它的可发表性取决于一个窄而干净的 claim：

> 在强条件零模型下，冻结轨迹中的候选状态-端点对齐 residual 能提供可重复的无监督异常增量。

若自然结果为负，这仍是一个清楚的负结果；若为正，才具备论文主贡献。

## Drift Warning

NONE。当前方案仍解决原 anchor，没有漂成可视化、factorial recapture、神经 detector 或标签调参。

继续避免三件事：

- 用软件/TDD/tiny Llama 执行成功暗示检测有效；
- 把 two-step residual 写成 pure interaction；
- 在自然结果出来前把 method refinement score 当成 paper readiness。

## Verdict Rule Interpretation

这轮不给 READY 不是因为实现规格不具备，而是因为 `research-refine` 的 READY 需要方法贡献接近论文可辩护状态。当前更准确的状态是：

- **实现规格：具备，可以跑自然评估。**
- **研究 claim：待证，不能写成成功。**
- **论文 readiness：不足 9 分，需真实结果。**

Final verdict: **REVISE**。
