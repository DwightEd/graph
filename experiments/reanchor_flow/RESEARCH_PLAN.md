# ETCC validation plan

## 当前状态

核心计算链已经实现：配对世界、semantic source units、attention/message 双 backend、
target-specific path gradient、residual-aware throughput、root 双向 patch、carrier patch/block、
corridor cut/patch/block 和 restoration positive control。

schema-v8 detector 冻结为 baseline。最新 held-out 指标为：

| Task | token AUROC | token AUPRC | onset AUROC | onset AUPRC |
|---|---:|---:|---:|---:|
| QA | 0.5804 | 0.1027 | 0.7431 | 0.0068 |
| Summary | 0.5915 | 0.0632 | 0.6684 | 0.0077 |
| Data2txt | 0.6198 | 0.0985 | 0.5622 | 0.0111 |
| ALL | 0.5928 | 0.0906 | 0.6267 | 0.0077 |

这些数值只说明旧的 role-share/peak 特征不能稳定识别 onset。它们不用于设置 ETCC
edge coverage、root 数或因果阈值。

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

停止规则：如果 message backend 在 matched edge budget 下不优于 raw attention，不能声称
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
