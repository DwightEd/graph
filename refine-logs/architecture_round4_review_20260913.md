# Architecture Round 4 Method Review

日期：2026-09-13

## 审查方式

Codex MCP 不可用。本文件是协作代理的 GPT-5.5 xhigh 风格备用方法审查，按 `research-refine` 的同一 7 维加权 rubric 执行；它不是缺失 MCP 的外部 verdict，也不是实验审计。未跑 GPU，未改代码、冻结 RAGTruth 结果、状态文件或 manifest。审查输入为：

- `refine-logs/architecture_round3_20260913.md`
- `refine-logs/architecture_round3_review_20260913.md`
- `refine-logs/grounding_architecture_lit_20260913.md`
- `refine-logs/attribution_architecture_lit_20260913.md`
- `refine-logs/unsupervised_detector_lit_20260913.md`
- `route_graph/causal_contrast.py`
- `tests/test_causal_contrast.py`

`causal_contrast.py` 与 CPU 测试只说明有限事件/E-V 内核已有局部工程定义；它们不是自然数据结果，不证明方法有效或覆盖足够。

## 总体判断

修订3实质性补上了第三轮指出的主要结构缺口。现在的结构不是“QA 后附解释图”那么薄：它加入了 evidence-candidate-blind native 候选流、source+history 跨角色 R gate、source input 到指定 V-message 再到 recipient `F_G` 的嵌套介导、适用消息与 MLP 的真实 2x2 交互，以及按 forward 数记账的 256/Claim 机制教师预算。更重要的是，文稿不再声称纯内部事实归属已经解决，也不再把 reader 当独立真值。

我接受一点 scope 反驳：原用户锚点是“不以幻觉标签训练”，并未禁止使用来源文本或冻结语义模型。当前方法满足这个字面锚点。不能再把“使用 Qwen reader”本身当作硬性否决。真正的边界应写成：这是 hallucination-label-free、frozen-reader-anchored、native-mechanism-certified audit；它尚未解决 pure internal evidence attribution。

仍不能 READY。阻断项已经从“方法对象未定义”变成“若干证书的可识别性和控制还差最后一层”：broad R 对整 role 缺 matched control，只能是 effect-only；嵌套介导是清楚的 V-message path-specific effect，但不能升级为全部 constraint-information origin；MLP 2x2 判据自洽，但预算和优先级必须确保它不会挤掉 route/origin 证书；256/Claim 是机制教师成本，不是可部署检测器。方法可以进入实现规格冻结前的最后修订，但还不到 9 分 READY。

Verdict: **REVISE**

Overall score: **7.8 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8.5 | 符合原始锚点：不用自然幻觉标签训练，保留语义锚点与 native 采用测量，明确不宣称纯内部归属已解。N/withholding 保留了“来源缺失当前阶段时长却生成具体承诺”的核心案例。 |
| Method Specificity | 8.1 | `F_G`、E/V/R/M、nested mediation、2x2 interaction、预算、coverage 和降级规则都已具体。未闭合点主要是 broad R 的选择性控制、paired R 的正式定义，以及 origin/interaction 证书的优先级。 |
| Contribution Quality | 7.2 | 贡献已从 QA pipeline 转为 route-certified mechanism audit，有更清楚的新意。仍受限于 reader anchor 和复杂证书 taxonomy，必须靠 witness 覆盖率证明不是 sparse case-study tool。 |
| Frontier Leverage | 8.3 | 冻结 reader、冻结 observer、有限事件干预、嵌套 path-specific mediation、同目标 controls 都是合适的现代机制工具；没有在教师成立前训练 GNN 是正确选择。 |
| Feasibility | 7.0 | 无新训练组件，256/Claim 的机制教师预算可在小规模自然 dev 上测，但对单 4090 和大量样本很贵；Qwen 调用、donor/recipient replay、MLP 2x2 和层带验证的实现复杂度真实存在。 |
| Validation Focus | 8.0 | 验证目标已对准语义锚点 vs native 证书、coverage、route/history/correction/interaction 分母，不再做 trivial relation experiments。 |
| Venue Readiness | 6.9 | 已有强论文雏形，但还缺自然链条覆盖证据和 paired R/control 的最终规格。作为方法计划接近可执行，作为 READY top-venue claim 仍早。 |

Weighted calculation: `8.5*.15 + 8.1*.25 + 7.2*.25 + 8.3*.15 + 7.0*.10 + 8.0*.05 + 6.9*.05 = 7.79`.

## 已解决或接受的关键问题

### 1. Scope 处理现在合理

当前文稿说清楚了两层命题：

```text
Solved by this architecture:
  hallucination-label-free audit using frozen-reader evidence anchors
  plus frozen-generator native mechanism certificates.

Not solved:
  pure internal generator-only factual evidence attribution.
```

这符合原锚点，不应再因为使用 reader 而降成 RETHINK。需要继续禁止的是把 reader 事实判断伪装成 native 图事实归属。

### 2. Native candidate-blind 流是实质增强

候选提议器不接收来源适用性分数、reader 引文或 reader 选择，只用 source/history 覆盖树、原生计算和冻结 `F_G` 提出候选，然后再让 reader 判语义关系。这解决了上一轮“Qwen 同时决定哪里该看和哪里被采用”的问题。

仍需在实现中强制两条流分开落盘：

```text
reader_reference_candidates
native_blind_candidates
native_candidate_semantic_labels
```

如果最后只用 reader candidates 通过，不能算 native independent adoption discovery。

### 3. 嵌套介导定义基本自洽

修订3没有犯“inner/outer 各显著就等于 mediation”的错误。它定义的是 donor input span `S` 改变指定 key message `K` 的 V 表示，再把这个 donor V 注入 recipient 中已选的 `A_ij V_j` 消息位置，测同一个 `F_G`。这确实是一个 path-specific V-message-mediated input effect。

可接受的结论名称是：

```text
input_origin_mediated_effect through selected V-message channel
```

不可接受的升级是：

```text
full constraint information origin
complete source-to-output causal path
natural direct/indirect effect
```

因为它固定 recipient attention，绕过 QK 路由变化，也不排除同一个 K 同时携带其它 prompt 信息。文稿的 `origin_nonexclusive=true` 是必要字段。

### 4. 适用消息 × MLP 的 2x2 交互是合理补齐

当前公式方向是对的：

```text
Delta_E = F11 - F01 <= -0.5
Delta_M = F11 - F10 >=  0.5
Omega   = (F11 - F10) - (F01 - F00) >= 0.5
```

在 `F` 是 bad-over-alternative log-odds 时，`Delta_E < 0` 表示适用消息反对坏事件，`Delta_M > 0` 表示 MLP 推动坏事件，`Omega > 0` 表示 MLP 的坏偏好效应依赖该适用消息存在。称为 `evidence_MLP_antagonism` 或 interaction witness 是合适的。它不能排除同时存在 route error，也不能称唯一成因；文稿已经写到这一点。

## 仍阻断 READY 的问题

### 1. Broad R 只有 allocation effect，不能给选择性证书

修订3里的 R gate 把重分配域扩为 source+history，保持两者合质量和其它 role 不动。这能测跨角色 allocation effect。但对“整个 source role”或“整个 history role”没有自然的 matched unrelated role control。文稿承认 whole-role R 没匹配控制时只能 effect-only，这是正确的，但这意味着 cross-role routing 证书还没完全闭合。

你提出的 paired R 是更合理的最小修复，应正式写入下一版：

```text
Given:
  H = native-positive history group
  S = native-negative or applicable source group
  C = matched unrelated source group

pairedR(H -> S):
  remove a fixed mass from H;
  move only that mass to S by S's original proportions;
  keep the same query/layer/head and total source+history mass.

control pairedR(H -> C):
  keep H, removed mass, query/layer/head identical;
  change only destination from S to matched unrelated source C.

specificity =
  Delta_pairedR(H -> S) - max_C |Delta_pairedR(H -> C)|
```

This tests whether moving allocation away from a harmful history group toward the applicable source group specifically changes the same `F_G`. It is cleaner than broad role R for selective cross-role claims. Broad R can remain as diagnostic `cross_role_allocation_effect`.

Priority: **CRITICAL**

### 2. Nested mediation needs a sign/coverage contract

The mediation estimand is coherent, but the acceptance rule still needs two small guardrails.

First, distinguish three outcomes:

```text
position_effect_only:
  K -> F_G passes, S -> K -> F_G does not.

origin_mediated_effect:
  K -> F_G passes and S -> K -> F_G passes with compatible sign.

origin_mismatch:
  K -> F_G passes but the cited S has opposite/unstable/no mediated effect.
```

Second, report mediated coverage:

```text
mediated_coverage =
  #position witnesses upgraded to origin_mediated_effect
  / #position witnesses tested for source-origin
```

Without this, a few successful origin examples can hide that most route witnesses are only endpoint-position effects.

Priority: **IMPORTANT**

### 3. The 256/Claim budget is self-consistent but fragile

The table sums correctly and now counts branch/donor calls as real forwards. That is a major improvement. The issue is that 256/Claim must cover candidate-blind screening, position/role validation, source mediation, MLP interaction, layer bands, template sensitivity, invalid gates, controls, and repeats. It is plausible for a mechanism teacher on a small dev batch, but not enough to guarantee all certificate types per Claim.

Concrete fix: add a fixed priority policy:

```text
Priority 1: certify strongest native-blind position effect.
Priority 2: if cross-role pattern is present, run pairedR.
Priority 3: if applicable evidence has negative V/origin effect and M has positive effect, run 2x2 MLP.
Priority 4: run layer-band refinement only after one mechanism certificate exists.
Priority 5: run N template sensitivity only for N-class certificate candidates.
```

If budget ends before a priority is reached, output the corresponding unresolved status. Do not spend reserve to search for a different easier success after a verification failure.

Priority: **IMPORTANT**

### 4. The MLP interaction can be expensive and non-local

Scaling an MLP branch at selected query/layer positions is implementable, but it is a broad intervention. It can change syntax, discourse state, and future branch likelihood beyond the factual slot. Matched V controls help, but do not fully isolate a semantic transformation.

Keep the current narrow name:

```text
evidence_MLP_antagonism
```

and add:

```text
collateral_delta =
  effect on non-target content tokens inside the same B/A event,
  reported but not used to accept the witness.
```

If collateral dominates target-slot logp changes, downgrade to `broad_mlp_effect`.

Priority: **IMPORTANT**

### 5. QA anchor remains the factual ceiling

The method now fairly satisfies "no hallucination-label training", but factual unsupported intervals still depend on Qwen's question generation, answerability, evidence extraction, relation labeling, and E/C/N/U comparison. This is acceptable only with the existing honesty policy:

```text
semantic_anchor_only is a required baseline;
reader errors cap factual detection;
native certificates explain measured adoption, not factual truth.
```

Do not let successful native certificates compensate for a bad semantic anchor. If the reader labels the wrong Claim or misses the relevant condition, route measurement may be technically correct for the wrong target.

Priority: **IMPORTANT**

## Completeness Against The User's Three Gaps

### Applicable constraints

**Mostly addressed as an auditable frozen-reader anchor**, not as pure internal discovery. Relation-slot QA, hidden answer, condition-preserving source answer, NULL layering, N withholding, and source-origin mediation together form a concrete solution to "what evidence should apply" under the allowed scope. The remaining risk is reader reliability, not undefined interface.

### Automatic lookback

**Addressed for position/query groups and partially for source-origin.** Native-blind search plus measured E/V/R/M effects avoids entropy top-k and manual 12/14 candidates. The lookback claim is still bounded: all-layer groups are wide, layer-band refinement is optional under budget, and source-origin requires mediation.

### Route-derived hallucination / continuous spans

**Addressed under certificate coverage constraints.** C-class grounded replacement and N-class commitment-vs-withholding are now separate. Error continuation requires history message plus input mediation, not mere textual dependence. Fact spans remain reader-Claim intervals; influence spans come from fixed-history intervention. This is the right split, provided the paper never merges semantic-only unsupported spans into route-derived successes.

## Implementation Readiness

The full method is implementable, but not implemented yet. Current `causal_contrast.py` covers only:

```text
ContinuationContrast
message_gate_delta for route/content E/V-style operations
```

It does not yet implement:

```text
native-blind candidate search
paired or broad R in model replay
donor-recipient nested mediation
MLP branch 2x2 interaction
budget scheduler
matched controls
layer-band refinement
JSON audit output
```

This is fine for method planning. It should not be described as a completed system.

## Drift Warning

No drift relative to the literal anchor. There is still a strong-claim warning:

```text
Allowed claim:
  no natural hallucination-label training; frozen-reader evidence anchoring;
  native finite-intervention certificates for adoption, source-position,
  mediated origin, cross-role allocation, and MLP interaction.

Disallowed claim:
  pure unsupervised generator-internal factuality detector.
```

## Simplification Opportunities

1. Keep broad R as diagnostic and add paired R for selective cross-role certificates; do not add Q/K attribution yet.
2. Keep mediation limited to V-message-mediated input effect. Prompt-internal full graph tracing can remain future work unless origin coverage fails badly.
3. Do not train a GNN until the 256/Claim teacher demonstrates coverage and stable witness categories on natural dev data.

## Modernization Opportunities

NONE as new trainable modules. The modern part is already the finite-intervention teacher with same-target controls and path-specific mediation. More model components would weaken the story.

## 最少必要修订

1. 正式写入 paired R，并把 broad R 降级为 `allocation_effect_only` unless paired/matched controls pass.
2. 给 nested mediation 增加 `position_effect_only`、`origin_mediated_effect`、`origin_mismatch` 三分法和 mediated coverage 分母。
3. 给 256/Claim 增加机制证书优先级；预算未到的机制必须 explicit unresolved，不能回填更容易成功的组。
4. 给 MLP 2x2 增加 collateral diagnostic；若非目标 token 变化主导，则降级为 broad MLP effect。
5. 最终输出表分开报告 semantic-only、position-only、origin-mediated、within-role、paired cross-role、content-message、MLP-interaction、unresolved 的分母。

Final verdict: **REVISE**。修订3已经从“可实现雏形”推进到“接近规格冻结的机制教师方法”。它满足不用幻觉标签训练的原始约束，也保留了纯内部归属未解的边界。下一轮只需补 paired R、mediation outcome taxonomy、预算优先级和 MLP collateral 降级；不需要增加训练组件或扩大实验菜单。
