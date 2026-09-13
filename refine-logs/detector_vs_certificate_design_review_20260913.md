# Detector vs certificate 核心取舍审查

日期：2026-09-13  
模式：`research-refine` 有界方法规格审查；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。范围：比较 A exact-A/B certificate-first auditor 与 B full-coverage soft graph detector + certificate overlay。未训练、未启 GPU、未改代码。

## 结论

是的，当前把“检测分数”和“高置信因果证书”绑得过死，是 v1-v3 全弃权的结构原因之一。A 是严格审计器，不是完整检测器。它要求每个风险分数先通过 exact A/B、条件核验、matched controls 和 native 干预，导致 sourceQA/contrast 任一环节失败时 native 近零调用。这个设计适合发少量高精度机制证书，不适合满足用户要的“准确、可评估的自动检测/定位方法”。

下一版主线应改成 B：**全覆盖软图检测器先输出可评估风险分数和定位排序，A/B 因果证书只作为子集 overlay**。这不是放宽成任意相似候选，也不是把 dependence 改名成 truth；它是把 prediction 和 proof 分开。检测器可以失败、可以被 RAGTruth 评价，但不应因为没有形式证书而拒绝输出预测。

B 可行，但只在一个有限意义上可识别：它能定义一个无幻觉标签训练的、参数来源明确的 structured risk posterior；它不能无监督识别“真实 hallucination posterior”。如果冻结 Qwen finite verifier 错、source graph 漏、或 native dependence 只反映格式/topic，B 会错。这个风险必须通过同 Qwen 无图基线和人工标签评价暴露，而不是在方法名里隐藏。

## A 与 B 的角色分工

### A：exact-A/B certificate-first auditor

优点：

- 方向性清楚：`F = log P(B) - log P(A)`，能说原事件相对来源替代/withholding 的偏好。
- 可做 matched controls、origin mediation、pairedR、MLP interaction，证书语义强。
- 不容易把弱 semantic reader 的判断直接包装成 native truth。

缺点：

- 需要 exact source answer、完整条件、可编辑 A/B、共同 prefix、native controls 全部同时成立。
- 对多角色错、derived support、Data2txt scalar/null/range、未抽条件、same-value wrong-event 极易全弃权。
- 把“检测/排序”推迟到证书之后，导致没有证书就没有检测结果。

A 应保留为 `certificate_overlay`，不应再作为主检测入口。

### B：soft graph detector + certificate overlay

B 的目标不是证明每个分数，而是给每个 claim/event 输出：

- `risk_score`：当前 claim 不受来源支持或沿错误历史传播的模型化风险；
- `localization_distribution`：source/history/role/span 的候选责任分布；
- `confidence/unknown_mass`：有多少来自有效关系、候选覆盖和 native specificity；
- `certificate_ids`：若 A/B 子集通过，附上高置信机制证书。

B 直接满足“可评估检测/定位”：全 claim 都有 score/ranking，可以用 RAGTruth 或新人工标签评估 AUROC/AP/span IoU；证书覆盖单独报告。

## 可识别的最小联合推断目标

可行的最小目标是一个**固定参数因子图**，不是训练 free heads。变量和输入如下。

### 变量

对每个 response event/claim `i`：

```text
Y_i ∈ {supported, unsupported, unknown}
Z_ir ∈ source candidate IDs ∪ {NULL, UNKNOWN}       # role ownership / source assignment
H_ji ∈ {reuse, correction, elaboration, new_topic, unknown} for j < i
M_ig ∈ {pushes_B, suppresses_B, no_specific_effect, unmeasured} for native group g
```

`Y_i` 是预测状态，不是证书。`Z_ir` 由 event matcher 或 source-owner pointer 供给；`H_ji` 由 finite option history relation verifier 供给；`M_ig` 由 blind claim logP native dependence 给出。

### 输入势函数

1. **Ownership / candidate potential** `π_i(z)`  
   来自 event-local matcher 或 source self-supervised owner pointer。若用训练头，训练目标只能是 source 自身 mask-value → original source position；不能用 RAGTruth、synthetic hallucination binary labels、reader E/C/N 或 native Δ。

2. **Finite semantic relation logits** `q_i(l | z)`  
   冻结 Qwen 只在 frozen source/response IDs 上选有限标签：`support / conflict / not_stated_estimate / unrelated / condition_failed / unknown`。它不生成 source answer 或 quote。`not_stated_estimate` 只有 full raw source absence check 通过才可转为 `N_valid`，否则并入 unknown。

3. **Blind native dependence** `a_i(g)`  
   对原始 claim B 的完整 logP 做盲 native 搜索：

```text
Δ_text(g) = log P_theta(B | source,prefix) - log P_theta,gated(B | source,prefix)
```

正值表示 group g 推动 B；负值表示 group g 抑制 B。用于检测器时必须先做 signed matched-control subtraction：

```text
δ_i(g) = Δ_text(g) - median_{c in matched_controls(g)} Δ_text(c)
a_i(g) = tanh(δ_i(g) / τ_native)
```

`τ_native` 固定为预注册数值，例如沿用证书阈值的 0.5 nats；不能用 RAGTruth 调。没有 matched controls 时 `M_ig=unmeasured_or_uncontrolled`，只降低 confidence，不贡献 truth。

4. **Directional history potential** `h_ji`  
   由 previous/current frozen IDs 的有限关系标签和 history native dependence 共同给出。只能按响应顺序从前到后传播，不能用未来信息。

### Claim-level energy

先聚合候选关系：

```text
pE_i = Σ_z π_i(z) q_i(support | z)
pC_i = Σ_z π_i(z) q_i(conflict | z)
pN_i = Σ_z π_i(z) q_i(N_valid | z)
pU_i = Σ_z π_i(z) q_i(unknown/condition_failed/unrelated | z) + π_i(NULL/UNKNOWN)
```

基础 semantic log-odds：

```text
L_sem_i = log((pC_i + pN_i + ε) / (pE_i + ε))
```

Graph/native adoption 项只调制这个语义风险，不直接制造 truth：

```text
L_src_i = Σ_z π_i(z) a_i(g(z)) * log((q_i(conflict|z)+q_i(N_valid|z)+ε) / (q_i(support|z)+ε))
```

历史传播项：

```text
L_hist_i = Σ_{j<i} a_ji * [ q(reuse|j,i) * L_i_from_prev(j)
                            - q(correction|j,i) * L_i_from_prev(j) ]
```

其中 `L_i_from_prev(j)` 使用前一 claim 的 frozen risk logit，并只在 previous slot/event relation verified 时启用；否则进入 unknown/confidence penalty。

最终：

```text
L_i = L_sem_i + λ_src L_src_i + λ_hist L_hist_i - λ_unk pU_i
risk_i = (1 - u_i) * sigmoid(L_i) + u_i * 0.5
confidence_i = 1 - u_i
```

`λ_src, λ_hist, λ_unk` 必须固定且预注册。最小选择可以是 `λ_src=1, λ_hist=1, λ_unk=1`，因为 relation logits 和 native Δ 都在 log/纳特尺度，经 `tanh` 有界；若调参，只能用 source self-supervised / unlabeled stability，不用 hallucination labels。

这给出一个可评估概率式分数，但它是**模型内部 risk posterior**，不是已校准事实概率。论文/报告应同时给 AUROC/AP 和 calibration；不准按结果重调权重。

## 图如何真正影响检测分数

B 不是把 Qwen 判断贴一个解释。图至少在三处进入 score：

1. `π_i(z)`：source/response event topology、高维 span features、field-path/record provenance 决定哪些 source candidates 被纳入以及权重。
2. `a_i(g)`：blind native dependence 决定一个语义 relation 是否被模型实际用于生成 B，还是只是外部 verifier 发现的 unused evidence。
3. `H_ji`：历史关系和 native history dependence 把前文 risk 有方向地传到当前 claim，形成 continuous span，而不是按相邻句平滑。

同 Qwen 无图基线必须固定为：只用同一个 finite verifier 的 source-claim relation logits，不用 matcher topology、native dependence、history edge。第二个消融是 Qwen + matcher，但去掉 native/history。若 B 相对这两个基线没有提升，只能说图没有改善检测，不能靠证书案例写成功。

## 为什么这不等于把 dependence 当 truth

`a_i(g)` 从不单独进入 `Y_i`。它只乘在 relation log-odds 上：source group 推动 B 且 relation says conflict/missing，会增加 unsupported risk；source group 推动 B 且 relation says support，会降低 risk；relation unknown 时，native dependence 只给 localization/confidence，不给 truth。这样避免“模型依赖某位置，因此该位置就是正确/错误证据”的偷换。

若 relation verifier 失败，输出应是：

```json
{
  "risk_score": 0.5,
  "confidence": "low_relation_unknown",
  "localization_distribution": [...],
  "scope": "target_dependence_only",
  "certificate_ids": []
}
```

这能提供回看位置，但不能提供 route-derived hallucination claim。

## Data2txt 与复杂自然 case 的处理

- Data2txt scalar：field path/value/record 进入 `π_i(z)` 和 relation candidates；`None` 只贡献 unknown/null-literal，不自动变 N；bool 必须经 polarity verifier；range 用 typed overlap 状态，不能硬替换成点值。
- 多角色同时错：B 可给 event-level risk，因为 `π_i(z)` 和 `q_i(l|z)` 可聚合多个 role；但 A/B 证书只有未编辑 identity_basis 时才附加。
- 同义改写无字面 identity：可由 matcher/topology 给 candidate 和 soft score；没有 finite relation basis 时 confidence 低，不发证书。
- 句中未抽条件：condition residue 进入 `pU_i` 或 event-level risk；不能 silent drop，也不能让 residue 全部阻断 target-dependence。
- 同值不同事件：在 `π_i(z)` 中作为 ambiguity 分布保留；若 graph/native不能区分，risk 可输出但 localization confidence 低，A/B certificate 禁止。

## 可评估目标与分母

B 的主输出必须覆盖所有 claim/event：

```json
{
  "claim_id": "...",
  "risk_score": 0.0,
  "confidence": 0.0,
  "unknown_mass": 0.0,
  "top_source_assignments": [...],
  "top_history_edges": [...],
  "native_dependence_edges": [...],
  "scope": "soft_detection|target_dependence_only|certificate_overlay",
  "certificate_ids": [...],
  "failure_reasons": [...]
}
```

报告分母：

- `claims_total`、`events_total`、`roles_total`；
- `soft_scores_emitted`、`low_confidence_scores`、`relation_unknown_mass`；
- `matcher_candidate_supplied`、`same_value_ambiguous`、`identity_unresolved`；
- `native_dependence_measured`、`native_controls_available`、`target_dependence_only`；
- `A/B_contrast_valid`、`certificate_attached`；
- `continuation_soft_edges`、`continuation_certified_edges`。

自然标签只用于评价：answer/claim AUROC、AP、span IoU/F1、calibration、supported false positive、continuous span boundary。不能用于训练、weight tuning、threshold selection 或 cherry-pick。

## 必要取舍

推荐把主线改为：

```text
B: full-coverage soft graph detector
A: exact-A/B causal certificate overlay for a subset
```

这不是降低科学标准，而是把两个不同问题分开：检测器需要全覆盖、可排序、可评价；证书需要高精度、可解释、低覆盖也可以。v1-v3 的问题正是把证书当作检测器入口，导致没有证书就没有预测。

B 不应承诺：

- 无监督真值识别；
- 100% 定位；
- native dependence 自身判真；
- Qwen finite relation 是 GT；
- certificate 覆盖等于 detector 覆盖。

B 可以承诺并验证：

- 不用 hallucination labels 训练主方法；
- 每个 claim 输出固定因子图 risk/ranking；
- 图因素相对同 Qwen 无图基线是否提高检测/定位；
- A/B 子集是否提供高置信机制证书。

## Required before adopting B

1. 文档中显式拆分 `prediction_score` 与 `causal_certificate` 两套状态机。
2. 固定 factor energy、变量、输入势、权重来源和 forbidden inputs；不得留自由训练头。
3. 将 blind F_text/native dependence 定义为 signed edge potential，只在 relation-known 时影响 risk；relation unknown 时只输出 localization/dependence。
4. 同 Qwen 无图 baseline 和 Qwen+matcher-no-native baseline 预注册；RAGTruth 只评价不调参。
5. 输出 `risk_score` 和 `confidence/unknown_mass`，避免把 unknown 当 unsupported。
6. A/B contrast 继续作为 certificate overlay；没有 A/B 不阻止预测，但阻止机制 claim。
7. continuous span 由 directional history factor 产生；无 previous relation + native history mediation 时只能是 soft edge，不能叫错误传播证书。
8. 若 B 提升只来自 Qwen finite logits，图主张失败；若 B 提升来自 native/history但 relation unknown，高层 route-hallucination claim 仍失败。

## 最终判断

B 是更合适的主线目标；A 应降为证书层。不存在一个无需 hallucination-label 训练又能直接识别真实 hallucination probability 的完全可识别目标，但存在一个足够具体、可实现、可评估的无标签固定因子图检测器。它不会把 dependence 当 truth，也不会要求每个概率都有形式证书。它让图真正进入分数，同时保留 A/B 证书作为强机制证据。

若根代理继续只优化 A，下一轮很可能仍是“更少弃权的审计器”，不是用户要的完整检测/定位方法。若采用 B，必须接受论文/报告语言变成：**soft graph detector predicts; exact native auditor certifies a subset**。
