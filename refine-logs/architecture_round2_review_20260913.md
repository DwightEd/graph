# Architecture Round 2 Method Review

日期：2026-09-13

## 审查方式

Codex MCP 不可用。本文件是协作代理的 GPT-5.5 xhigh 风格备用方法审查，按 `research-refine` 的同一 7 维加权 rubric 执行；它不是缺失 MCP 的外部 verdict，也不是实验诚实度审计。未跑 GPU，未改代码，未改冻结 RAGTruth 结果、状态文件或 manifest。审查输入为：

- `refine-logs/architecture_round1_20260913.md`
- `refine-logs/architecture_round1_review_20260913.md`
- `refine-logs/grounding_architecture_lit_20260913.md`
- `refine-logs/attribution_architecture_lit_20260913.md`
- `refine-logs/unsupervised_detector_lit_20260913.md`

额外一手文献 spot check：

- QAFactEval: https://arxiv.org/html/2112.08542
- FactGraph: https://arxiv.org/html/2204.06508

## 总体判断

第二版比初稿强很多。它接受了第一轮最重要的批评：删除双训练头、删除自由 learned semi-CRF、删除尚无可靠监督目标的 intervention graph student，把首版改成冻结 Qwen3 语义 QA 锚点、冻结 Llama observer 消息图、有限干预组搜索。它也明确承认 QA/核对是模型估计，不是独立真值；路由分支只给采用、回看和条件影响证书，不单独判断事实真假。这些修订保住了问题锚点。

但它仍未到 READY。现在的主要问题不再是模块堆叠，而是贡献边界和证书定义。语义部分很像 QAGS/QAFactEval 系列的 relation-slot QA factuality pipeline 加更细的 provenance logging；生成图部分像后验 attribution/intervention audit。二者相接可以成为一个有价值的方法，但必须证明新增贡献不是“QA fact-check 后附一张路线解释图”。此外，主目标 `F = log P(B|h,D) - log sum P(A|h,D)` 与所谓共同前缀 gate 的计算对象还不够干净：完整续写在分歧后进入不同 teacher-forced 历史，不能继续说同一个 query 上的同一组门控解释了两条完整陈述。特异性控制项暂用 `F_text`，与主 log-odds 不同单位，这是必须修正的 blocker。

Verdict: **REVISE**

Overall score: **6.9 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 7.6 | 不用自然幻觉标签训练、区分事实区间和影响区间、回看改为固定历史有限干预证书，这些保留了大部分锚点。但原交接目标强调从冻结模型内部识别归属；当前事实归属由外部 Qwen reader 决定，native 图只做有条件路由审计。若用户要求 internal-only attribution detector，这是 scope drift。 |
| Method Specificity | 6.9 | 接口、阈值、E/V gate、预算和返回状态比上一轮清楚很多。但完整续写目标、共同前缀门控、同量纲特异性、NULL 判定和 Claim 关系标签仍有符号未闭合。 |
| Contribution Quality | 6.3 | 主贡献从“训练一个双视图图检测器”收敛成“QA 证据锚点 + 原生计算干预证书”。方向更可实现，但语义 QA 部分的新意受 QAFactEval/QAGS/FactGraph 挤压，route 证书是否是核心贡献还没有被公式和评价对象牢牢固定。 |
| Frontier Leverage | 7.7 | 冻结 foundation model 作为语义 reader 和 observer、有限干预、固定历史 log-odds 都是合适的现代工具；没有强行加 GNN/SAE/adapter 是加分项。 |
| Feasibility | 6.5 | 不训练新组件、每回答 64 branch forward、最多 2 个风险 Claim，比初稿可信。仍要面对 Qwen 调用数、两分支 teacher-forcing、E gate 重归一合法性、全层 simultaneous gate、可靠 c* 缺失和多数样本无证书的覆盖风险。 |
| Validation Focus | 7.1 | 主比较“语义锚点 alone vs 路由审计”对齐核心问题，并避免旧式 trivial relation experiments。但若控制项不同量纲或大量样本 certificate_unresolved，路由解释覆盖/特异性评价会失真。 |
| Venue Readiness | 6.0 | 可以成为一篇严谨的 route-verified factuality audit 论文雏形；现在还太容易被审稿人读成 QA factuality metric with post-hoc intervention traces。 |

Weighted calculation: `7.6*.15 + 6.9*.25 + 6.3*.25 + 7.7*.15 + 6.5*.10 + 7.1*.05 + 6.0*.05 = 6.90`.

## Scope Check

当前方案只**部分完成**原交接目标。它完成的是：

```text
外部冻结 reader 给出可审计的事实/证据锚点；
冻结 native observer 测量这些锚点在原生成轨迹中的 route/content 依赖；
用有限干预证书输出 lookback、影响区间、延续/纠正标签。
```

它没有完成的是更强命题：

```text
仅从冻结生成模型内部信号自动识别正确事实归属和缺失证据。
```

这一区分必须进入方法摘要。内部有无信号不能被问答结果直接替代；问答只能定义待核查的语义目标和候选证据。若用户最终坚持“内部图本身识别事实归属”，第二版应判为 **RETHINK** 而不是 REVISE。若接受“事实归属由冻结 reader 锚定，内部图只负责测量实际采用”，则第二版是可修的 REVISE。

## 是否真正回应三个关键缺口

### 1. 适用证据

**部分回应。** 关系槽问题、隐藏 answer_quote、保留阶段/地点/条件/否定/数量单位、来源全文回答、精确引文映射、invalid/uncertain/not_stated 分流，明显比初稿的 `membership(c,e)` 可审计。它解决了“当前陈述应该用哪些约束”的操作接口。

未闭合处是 NULL 和全源缺失。方案说“没有发现集合不自动证明全文没有支持”，同时又需要 `not_stated` 作为 N。若 Qwen 没找到答案，最多是 reader-estimated NULL，不是语义真空证书。必须在输出和评价中区分：

```text
NULL_discovered: reader says not_stated under current prompts.
NULL_verified_family: searched candidate family has no valid evidence.
NULL_global: source truly lacks evidence. 首版不能宣称这个。
```

此外，Qwen 同时生成问题、回答来源、核对引文、生成反事实替代，循环风险真实存在。使用同一个冻结 reader 的多提示自洽不是独立验证；它只是 weak semantic anchor。

### 2. 自动回看

**比初稿实质改善，但仍未完全闭合。** 二叉区间树、允许非相邻组、父子都保留、实测 forward 作为 oracle、预算内返回 non_unique/unresolved，这些回应了“不要 entropy top8 硬筛”和“回看不一定唯一”的要求。

阻断点在证书对象：`routing_witness` 要求 E gate，但 E gate 只在角色内部重归一，所以它测的是同一 role 内部的路由重分配，不测 source-vs-history 角色质量，也不测跨角色采用。如果错误来自“历史角色压过来源角色”，E gate 不会给 routing_witness；V gate 有效又只能叫 content_witness。这个边界可以接受，但必须写成覆盖限制，不能把未能证 E 解释为没有路由错误。

### 3. 路由衍生 hallucination 与连续 span

**定义更诚实，但贡献仍偏弱。** `unsupported(c)` 来自语义核对，`routing_supported(c)` 要求实测 routing_witness，历史延续/纠正规则也避免了“强依赖就继承错误标签”的旧问题。

问题是“条件依赖不等于致错”。即使某个错误陈述的 log-odds 被某来源/历史组正向推动，也只能说该组提高了这个固定错误陈述相对于有限替代集合的偏好。它不能证明自由生成一定因此出错，也不能证明这就是唯一错误原因。当前文稿已经用长标签收窄了 claim，但还需要把 JSON certificate 的枚举名也收窄为 `contrast_preference_witness` 或 `route_preference_witness`，避免 `routing_error` 这种过强名词。

连续 span 方面，删除自由 semi-CRF 后，边界主要来自 Claim 抽取和关系标签。这样可实现，但不再是 learned continuous span detector。应如实说：事实区间是 QA Claim 覆盖后的后处理合并，不是由图本身学出的连续 span。图只给影响区间。

## Critical Blockers

### 1. QA + 事后解释的新意不足

QAFactEval 已经把 factuality metric 拆成 answer selection、question generation、source QA、answer overlap/question filtering/answerability。FactGraph 已经用 structured semantic graph/text adapters 做 factuality，并报告 subsentence-level inconsistency 能力。当前修订的语义阶段如果只是“用 Qwen 抽 relation question，再 source QA，再比较答案”，在相关工作面前不是新方法，只是更大的 reader 和更细日志。

可成立的新意必须放在以下窄点：

```text
对每个 source-derived/relation-slot factuality decision，
输出冻结生成器在固定历史下的 measured route/content certificate，
并据此区分 unsupported-but-routed、unsupported-without-route、
history-continuation、correction、influence interval。
```

Concrete fix:

- 标题和主贡献不要写“来源约束图检测器”或“QA 证据锚点”。
- 写成“route-certified factuality audit”或“contrastive route witness for unsupported claims”。
- 语义 QA 是必要锚点和强 baseline；图贡献只在 route certificate、lookback coverage、continuation/correction specificity 上主张。
- 主实验必须把 `semantic_anchor_only` 当第一基线。若完整方法只提升解释覆盖，不提升 unsupported detection AUROC，就不要声称事实检测性能提升。

Priority: **CRITICAL**

### 2. 完整续写 log-odds 与共同前缀 gate 的对象不一致

文稿定义：

```text
F = log P(B|h,D) - log sum_{a in A} P(a|h,D)
```

又说只在“两分支相同的前缀 query”上干预；分歧后 teacher-forcing 不同，但同一个 gate 在两分支作用于相同位置。这里不够精确。若 gate 只施加在共同前缀末 query，那么它只能解释下一步或前缀状态对分支的影响，不解释完整续写的全部 token logprob。若 gate 施加在分歧后的每个 token query，那么 B 分支和 A 分支的 query state、history keys、甚至候选 token keys 都不同，不再是“同一个计算节点”的共同前缀 gate。

Concrete fix: 把目标拆成两个，不要混写。

```text
F_first =
  log P(b_1 | h,D) - log sum_{a in A} P(a_1 | h,D)
```

`F_first` 是 routing certificate 的主目标，因为 query/prefix 完全相同。多 token answer 可以用第一个分歧 content token 或 answer-slot 的第一可区分 token。

```text
F_branch =
  log P(B | h,D, teacher-forced B-prefixes)
  - log sum_a P(a | h,D, teacher-forced a-prefixes)
```

`F_branch` 是完整陈述偏好诊断。它可以报告，但证书名称必须是 branch-conditioned contrast，不是共同前缀 route cause。对 `F_branch`，group 必须定义成 source/history evidence keys plus branch-local query positions，而不是同一个 query node。

若坚持完整续写作为主证书，必须返回：

```text
query_scope = shared_prefix_only | branch_conditioned
branch_map = {B: query_ids_B, A_k: query_ids_Ak}
```

并明确它不证明自由生成修复。

Priority: **CRITICAL**

### 3. 特异性控制项 `F_text` 与主目标不同量纲

文稿自己标出这个问题，审查不能放过。主效应是坏/好陈述的 log-odds；对照却用 supported Claim 的平均文本 logp 变化。二者单位、长度敏感性、符号意义都不同。用 `F_text` 做 specificity 会把格式/共同支持/流畅度扰动误判成或漏判成 factual contrast specificity。

Concrete fix: 所有 specificity 都用同类目标。更合理的首选不是另建后置控制陈述，而是在同一个 `B/A` 续写 log-odds `F` 上比较候选组与匹配无关证据组：

```text
Control group C:
  same role,
  same query_scope,
  similar token length,
  similar baseline attention mass,
  semantically unrelated to the Claim under the frozen reader,
  and passing the same gate validity checks.

specificity(G,c) =
  Delta_del(G,c) - max_C Delta_del(C,c)
```

这比“另选一个 supported control claim 并构造自己的 contrast”更干净，因为目标、query、role、gate 类型和单位都一致。它直接检验消息组选择性：这个候选证据组对当前坏/好对比的作用是否超过匹配无关组。

但该控制只支持：

```text
selected message group is specific to this contrast target
```

它不支持：

```text
intervention is globally harmless,
the model would freely generate the corrected answer,
or the full task behavior is unchanged.
```

若没有语义可信且匹配合格的 control group，就输出 `specificity_unresolved`。后置 `F_text` 只能保留为 collateral-dependence diagnostic，不能作证书。另建控制陈述的方案可以作为补充 collateral check，但不应是主 specificity。

次选方案：

```text
Specificity contrast:
  对同回答中一个已支持 control claim c0，
  构造一个条件不兼容或 answer-swapped 的 c0_bad，
  保留原 c0_good，
  F0 = log P(c0_bad | h0,D) - log P(c0_good | h0,D)

specificity(G,c) =
  Delta_del(G,c) - max_{c0 in controls} |Delta_del(G,c0)|
```

若不能可靠构造 control contrast，则 specificity 状态必须是 `specificity_unresolved`，对应证书不能升级到 `routing_witness`，只能叫 `route_effect_observed`。

保留 `F_text` 只能作为依赖诊断，不得参与路由致错、特异性、sufficiency 或贡献主指标。

Priority: **CRITICAL**

### 4. 反事实修订与证据归属存在循环依赖

来源语义读者先判断原 Claim 不支持，再生成一个来源支持的最小替代 c*，再用同一 reader 验证 c*。这会形成闭环：如果 reader 误读来源，它可能同时产生错误的 not_stated、错误的替代陈述和错误的支持证据，最后 Llama 干预只是在解释一个错误构造的 contrast。

Concrete fix:

- c* 只允许由 `source_answer` 的精确引文替换 `answer_span`，其它 premise_spans 不得由模型自由改写。
- 每个 c* 保存 edit script：`replace answer_span with source_answer_quote`、保留的限定条件列表、删除/新增 token 列表。
- 若需要改动 predicate、条件或前提才能得到 c*，该样本退出 route-cause 分类，标 `contrast_requires_rewrite`。
- 生成多个 c* 时，不让 reader 看原先 E/C/N 判断，只看 question、source_answer_quote、原 Claim skeleton。

这样 c* 是 source-derived slot repair，不是 teacher 自己想出的正确陈述。

Priority: **CRITICAL**

### 5. `not_stated` 仍被用得过强

方案说 N 需要来源问答和条件核对均为 not_stated，但这仍是 reader 的估计。长来源、多跳证据、表格、隐含同义关系下，reader 没找到答案不能支持“没有适用证据”。如果 N 直接进入 `unsupported(c)`，事实检测上限完全被 Qwen answerability 支配。

Concrete fix:

```text
unsupported(c) =
  contradiction(c) OR reader_not_stated_with_search_coverage(c)

reader_not_stated_with_search_coverage(c) requires:
  source chunks searched and logged,
  answerability prompts agree,
  no exact/alias candidate found in retrieved relevant units,
  and coverage_status != partial/unknown.
```

否则输出 `evidence_unresolved`，不要给高 r。自然评价可以把 unresolved 当 0.5 score 或单列 abstention，而不是当 negative。

Priority: **CRITICAL**

### 6. 多数自然样本无证书时，结构可能只剩 QA baseline

第二版把 route certificate 建立在可靠 c*、有效共同前缀、有效 E/V gate、匹配 control group、半量稳定、预算足够这些条件上。每个条件都合理，但自然数据中很可能大量样本失败：没有可靠 c*，只有 not_stated；source answer 不是 slot repair；候选太长；共同前缀太短；E gate invalid；64 branch forward 不够完成证书；没有匹配无关 control group。

如果多数风险 Claim 输出 `no_semantic_contrast`、`specificity_unresolved` 或 `budget_unresolved`，完整结构在事实检测上就退化为外部 Qwen QA，native 图只覆盖少数可解释案例。这不一定没价值，但不能宣称解决了用户要求的“自动回看和 route-derived continuous spans”。

Concrete fix: 预注册覆盖门槛和降级含义：

```text
certificate_coverage =
  #risk_claims_with_valid_route_or_content_certificate / #risk_claims

READY-to-evaluate threshold:
  report full method only if certificate_coverage >= predeclared floor
  on source-disjoint dev before looking at natural hallucination labels.

If below floor:
  paper claim downgrades to semantic QA audit with sparse route case studies.
```

这个门槛不能用自然标签调。它是方法适用性检查，不是性能 claim。

Priority: **CRITICAL**

## Important Issues

### A. E gate 的“路由证书”范围太窄

E gate 在 role 内重归一，保留接收状态原来的 role 总质量。这能测试“同一 role 内哪些 source/history key 被看见”，但不能测试“source 角色是否被 history 角色挤掉”。若错误来自历史复述主导而来源证据被整体忽略，E gate 可能无效，V gate 可能有效。当前规则会把这种情况降为 content_witness 或 unresolved。

Concrete fix: 增加一个只作诊断的 role-mass gate，不作为第一版主证书也可以：

```text
R gate:
  scale all source-role attention mass by lambda,
  redistribute removed mass to same query's non-source roles by original proportions.
```

若不加 R gate，必须在方法限制中写明：routing_witness covers within-role routing, not source-vs-history role allocation。

Priority: **IMPORTANT**

### B. 全层同时 gate 破坏定位解释

默认 `layers=全部32层` 作为 group，会让证书变成“这个 evidence group 在某些层整体有影响”，不是精确回看位置。之后再做层细分诊断可以，但主输出若叫 lookback group，必须包含 query 和 layer 的粒度或明确是 query-level evidence group。

Concrete fix:

- 首版主搜索 group 设为 `(role, evidence_keys, query_interval, layer_band)`，layer_band 默认 `{early, mid, late}` 或 `{all}` 但证书名不同。
- `all_layers` 通过只能叫 `global_route_effect`。
- 要称 `lookback_group`，至少需要在一个 layer_band 上通过 delete/half/sign/specificity。

Priority: **IMPORTANT**

### C. 64 branch forward 预算需要跟组判据配平

一个证书至少需要 base、sham、delete、half、keep、control；两分支 contrast 又翻倍。若每回答最多 64 branch forward，还要解释 2 个风险 Claim、多个 source candidates、父子细分和非相邻组合，预算很容易只够粗筛，不够证书。

Concrete fix: 预注册两级预算：

```text
screen_budget:
  delete-only, no certificate, returns candidates.

certificate_budget:
  for top M candidates only; includes half, keep, specificity controls.
```

证书覆盖率分母必须是“需要解释的风险 Claim 数”，不是“进入 certificate 阶段的候选数”。预算不足只能 `budget_unresolved`，不能把 delete-only 当 contributory。

Priority: **IMPORTANT**

### D. `r = P(C)+P(N)` 混合了不兼容和未说明

C 是明确反证，N 是未说明；二者对定位和 route contrast 不同。C 可以构造 answer replacement，N 常常没有唯一正确替代。把二者加成同一个 unsupported risk 会让后续 route 证书目标混乱。

Concrete fix:

```text
r_contra = P(C)
r_missing = P(N)
unsupported_for_detection = max or calibrated pair, reported separately
eligible_for_contrast =
  C with source_answer available OR N with source-derived answer slot available
```

对 N 类样本，如果没有 c*，只能返回 evidence_missing/unresolved，不返回 route-derived error。

Priority: **IMPORTANT**

### E. Claim 关系标签仍是弱语义判断

same_claim/elaboration/correction/new_claim 由 Qwen 输出。它决定连续错误、纠正和新陈述分组，但没有独立约束。若这一步错，历史延续和纠正误报会很高。

Concrete fix: 关系标签必须带引用和结构条件：

```text
same_claim/elaboration requires shared predicate or explicit coreference plus compatible slot.
correction requires explicit negation/replacement marker or incompatible answer to same question.
new_claim if predicate/slot/question changes beyond threshold.
unknown if evidence is only topic similarity.
```

不需要训练新模型，但需要 deterministic checks 约束 reader 输出。

Priority: **IMPORTANT**

## 对用户偏好的张力判断

用户偏好是无监督图，但字面 anchor 是“不以幻觉标签训练”。第二版满足字面 anchor：没有使用自然 RAGTruth 幻觉标签训练、阈值选择或早停；也没有新增训练组件。但它不满足强意义的无监督：Qwen3 的预训练/指令能力、问答、answerability、关系标签、反事实修订和 E/C/N 判断都是语义监督先验。应在文稿中明确：

```text
This is hallucination-label-free and training-free for the audit model,
but not semantics-free or unsupervised in the strict sense.
```

如果论文继续自称“无监督图检测”，会被审稿人直接打中。更稳的说法是“no natural hallucination-label training; frozen-reader evidence anchoring; measured route certificates”。

## 最少必要修订

1. 修正目标定义：把 `F_first` 作为共同前缀 routing certificate 主目标，把 `F_branch` 作为 branch-conditioned 完整续写诊断；两者证书名、query_scope 和可解释范围分开。
2. 删除 `F_text` 参与 specificity；主 specificity 改成同一 `B/A` 目标上的匹配无关证据组对照。不能构造合格 control group 时，证书降级为 unresolved/effect-only。
3. 把 c* 限制成 source_answer_quote 对 answer_span 的 slot repair，保存 edit script；需要自由改写 predicate/条件/前提的样本退出 route-cause 分类。
4. 改写贡献句：语义 QA 是锚点和基线，核心贡献只主张 measured route/content certificate 对 unsupported claims、automatic lookback、history continuation/correction 和 influence interval 的增量。
5. 收窄 `not_stated`：区分 reader-estimated NULL、searched-family NULL、global NULL；首版不得声称 global NULL。
6. 重新配平 64 branch forward：screen 和 certificate 两级预算分开，delete-only 不得升级成 contributory/routing_witness。
7. 明确 E gate 只覆盖 within-role routing；若要覆盖 source-vs-history role allocation，另定义 R gate 或把该失败模式标 unresolved。
8. 增加 scope 声明和 certificate coverage 门槛：若多数自然样本无 route/content 证书，方法 claim 降级，不得让 QA 替代内部机制测量。

## Drift Warning

有明确 scope 漂移风险。若原目标被解释为“冻结模型内部自己识别事实归属”，当前方案已经漂到外部 reader factuality audit + native route measurement。若原目标被解释为“不用自然幻觉标签训练，先由冻结 reader 锚定事实，再测冻结 generator 内部采用”，漂移可控。方法文稿必须把这两种解释分开，不能靠改名默默越界。

## Simplification Opportunities

1. 不再引入任何训练式组件、GNN、SAE 或 learned segmenter。第二版在结构复杂度上已经接近可实现边界。
2. 首版只支持 slot-repair contrast；复杂 paraphrase/c* 进入 unresolved。这样可以减少循环依赖和 log-odds 不可比。
3. 首版只把 E/V gate 作为 route/content witness；Q/K、MLP、role-mass gate 都放诊断或限制说明。

## Modernization Opportunities

NONE as additional modules. 现代性已经来自 frozen reader + frozen observer + finite intervention。继续加模型会削弱贡献。唯一需要现代化的是证书语义：用 contrastive finite-intervention witness，而不是把 QA score 包装成图学习。

## 进入下一轮前的 READY 条件

下一轮不是要新增实验，而是把方法对象写到不可误解：

- `F_first` / `F_branch` 的定义、用途、query_scope、证书名称完全分开。
- specificity 与主目标同量纲。
- c* 只能来自 source-derived slot repair，不能自由生成正确陈述。
- `unsupported`、`routing_witness`、`content_witness`、`route_preference_witness`、`influence_interval` 的枚举规则可以直接写成代码。
- NULL/abstain/coverage 的分母处理固定。
- 64 branch forward 预算足够支持证书，否则输出降级规则固定。

Final verdict: **REVISE**。第二版已经从“不可识别的训练式双图模型”变成“可实现的冻结审计器”，这是实质进步；但仍有五个阻断项不能放过：scope 从内部归属识别收窄为外部 QA 锚定后的机制测量、QA 管线新意不足、完整续写目标与共同前缀 gate 混写、`F_text` 不同量纲控制、语义反事实修订循环依赖。修掉这些后才值得进入完整自然链条实现。
