# ETCC validation plan

## 当前状态

核心计算链已经实现：配对世界、semantic source units、attention/message 双 backend、
target-specific path gradient、residual-aware throughput、root 双向 patch、carrier patch/block、
corridor cut/patch/block 和 restoration positive control。

真实数据的 native subset pilot 也已实现。它从 formal cache 自动选择 source-diverse cohort，
固定 observed-token/native-runner target，用原生 attention 或真实 message norm 构图，再以
source Value-message cut、corridor patch 和 carrier block 验证。它不要求手工 pair，也不承担
correct-vs-false factual claim。

schema-v8 detector 冻结为 baseline。最新 held-out 指标为：

| Task | token AUROC | token AUPRC | onset AUROC | onset AUPRC |
|---|---:|---:|---:|---:|
| QA | 0.5804 | 0.1027 | 0.7431 | 0.0068 |
| Summary | 0.5915 | 0.0632 | 0.6684 | 0.0077 |
| Data2txt | 0.6198 | 0.0985 | 0.5622 | 0.0111 |
| ALL | 0.5928 | 0.0906 | 0.6267 | 0.0077 |

这些数值只说明旧的 role-share/peak 特征不能稳定识别 onset。它们不用于设置 ETCC
edge coverage、root 数或因果阈值。

## Phase 0：native subset 可行性

先分别固定 message 与 attention 两个完全相同的 cohort，默认每任务 1 个样本、每样本 1 个
target；随后扩到每任务 5 个样本、每样本 3 个 target。先使用 `carrier_scope=response`，只有
恢复率和内存稳定后才运行 `all` 以寻找 prompt carrier。

必须先满足：

1. native 与 root-cut 两个 base world 的 corridor restoration validity 均为 100%；
2. artifact 中 `a` 等于 observed token，`b` 在所有干预前冻结且永不重算；
3. attention/message 只改变 transport ranking，functional `grad·message` 与干预算子相同；
4. source root、corridor、carrier 的正结果分别报告，不能以其中一个代替完整链；
5. label 只在全部 target 完成后由 `subset-evaluate` 加入。

停止规则：出现任何 restoration invalid 时先检查 persistent root-cut base gate 与 edge code；
不得通过放宽因果阈值掩盖。attention/message 的相同 coverage 不保证相同 edge count；主比较
必须使用 matched cohort，并按实际 corridor edge count 分层或另做固定 edge-budget 对照。若
message graph 在该对照下没有更高的 confirmed corridor/full-chain rate，则不能声称 message
transport 提供增量。
若 pilot 中 hallucinated target 太少，只增加预先固定的样本数/target 数，不能用 test label
回头挑选 target。

## Phase 1：受控 pair 正确性

先构造少量可人工核验、token 长度对齐的 clean/corrupt pairs：

1. QA：每次只替换一个 supporting passage 或精确 fact span；
2. Summary：替换一个 source sentence 中的实体/数值；
3. Data2txt：替换一个 leaf field，保持序列坐标；
4. 固定 correct/competing candidate 和 `contrast_origin`，不从内部图反推 target；
5. response 在两个世界完全相同，以 teacher forcing 审计同一 target。

必须报告每个 pair 的 tokenizer、变化 token、candidate unit、target pair effect。pair effect
过小的样本用于负控制，不进入“决定 target”的阳性集合。

停止规则：若 same-world delete-and-restore 不能在 dtype tolerance 内稳定恢复，停止所有
corridor 解释并修复 intervention operator。

## Phase 2：backend 与路径对照

在完全相同 pairs/targets 上运行：

- message backend（主方法）；
- raw attention backend；
- layer order shuffle；
- head 与 `W_O` block 错配；
- source-unit permutation；
- response-only carrier scope。

主比较不是 detector AUROC，而是：confirmed-root precision、corridor sufficiency、mediated
sufficiency、carrier mediated rescue、coverage/edge-count trade-off 与 restoration validity。

停止规则：如果 message backend 在显式固定的 matched edge budget 下不优于 raw attention，不能声称
真实消息或 target-specific sign 提供增量；如果 `all` 与 `response` 无差异，prompt carrier
不是必要创新点。

## Phase 3：completeness 与 interaction

对 edge coverage `0.80/0.90/0.95/0.99/1.00` 画 causal completeness curve：

\[
\text{completeness}(k)=
\frac{\text{corridor sufficiency}(k)}{F^+-F^-}.
\]

同时分别报告 necessity，不能假设 clean-cut 与 corrupt-patch 对称。多个 candidate units
共同变化时，代码会在 screening 后自动重跑 isolated-root world；后续再比较单 unit、联合
patch 与 factorial interaction。若交互很强，不把 joint screening effect 归给单个 unit。

停止规则：若 effect 只在 100% dense edge set 出现，稀疏 corridor 范式未得到支持；若
root 排名对 corruption baseline 极不稳定，则先改进 paired data，而不是调图权重。

## Phase 4：RAGTruth 外部评价

ETCC 对 RAGTruth 的使用顺序固定：

1. 在不打开 hallucination labels 时构造 source units、冻结 pairs/targets 和所有参数；
2. 完成 root/corridor/carrier capture；
3. 最后才将已确认的 mechanism mode 与 hallucination onset 对齐；
4. bootstrap unit 使用 `source_id`，QA/Summary/Data2txt 分别报告；
5. 不用 test label 选择 head、layer、coverage 或 root limit。

RAGTruth 没有 corrected target 和精确 supporting fact 的样本，只能报告 coarse-context
route，不进入“准确信息被采纳”的主结论。

## Phase 5：统一范式是否成立

只有以下三条同时通过，才把 ETCC 写成新研究范式：

1. 同一数学对象在三类 task 上工作，不需要按 task 重写 head/layer 权重；
2. attention 只能给 route candidate，而 message + target function + causal block 能稳定淘汰
   high-attention false paths；
3. failure 可以定位为 root selection、transport/content、carrier integration 或 terminal
   adoption，而不是重新合成一个难解释的 hallucination score。

任何阶段的负结果都保留在 artifact 中。没有通过 restoration、双向 root patch 或 block 的
对象统一命名为 candidate，不通过后验阈值把它升级为 mechanism。
