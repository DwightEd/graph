# Architecture Round 3 Method Review

日期：2026-09-13

## 审查方式

Codex MCP 不可用。本文件是协作代理的 GPT-5.5 xhigh 风格备用方法审查，按 `research-refine` 的同一 7 维加权 rubric 执行；它不是缺失 MCP 的外部 verdict，也不是实验审计。未跑 GPU，未改代码、冻结 RAGTruth 结果、状态文件或 manifest。审查输入为：

- `refine-logs/architecture_round2_20260913.md`
- `refine-logs/architecture_round2_review_20260913.md`
- `refine-logs/grounding_architecture_lit_20260913.md`
- `refine-logs/attribution_architecture_lit_20260913.md`
- `refine-logs/unsupervised_detector_lit_20260913.md`
- `route_graph/causal_contrast.py`

`route_graph/causal_contrast.py` 只被当作定义核对：它实现有限 continuation contrast 与 typed E/V gate 的局部内核，不构成方法实证；7 项 CPU 测试通过也不能替代自然数据有效性。

## 总体判断

第三版修掉了第二轮最关键的两个技术硬伤。共享前缀干预下的完整有限事件概率比

```text
F_G = log P_theta,G(B | D,h) - log P_theta,G(A | D,h)
```

是定义良好的：同一 gate 作用在同一公共前缀 query 节点，之后两条 teacher-forced 分支各自用真实网络重算。分支后计算路径不同并不使 `F_G` 无效。正确解释范围是：共享前缀消息组对两个预先枚举完整事件概率比的有限干预效应。它不解释后缀每个生成节点，也不证明自由生成会修复。第三版已写明 `query_scope=shared_prefix_only`，这点可接受；不必强制退回首分歧 token。`F_first` 作为诊断即可。

同一 `F` 上的 matched-source control 也比上一轮的后置 `F_text` 控制合理得多。它保持目标、role、query、gate 和单位一致，能检验当前消息组相对匹配无关组的选择性。文稿也正确保留了限制：这不是全任务无损、自由生成修复或全局 harmlessness 证据。

仍不能 READY。最大的未解问题是 scope 和内部独立归属：当前事实归属由外部 Qwen reader 锚定，native 图只在给定语义目标后搜索采用证书。文稿现在透明承认这一点，所以不是隐性漂移；但透明不等于完成了“从冻结模型内部识别归属”的强目标。其次，source key 的 V 已经吸收更早 prompt/context 信息，key 落在某段引文并不等于消息内容就是那段原始证据；当前 E/V 证书只能默认叫 position-anchored route/content effect，不能升级成 constraint-information origin。再次，E gate 只覆盖 role 内路由，不能覆盖 source-vs-history 角色分配；V gate 只能说明内容消息效应，不能说明 attention 路由选择。最后，正确证据确实进入但输出仍错时，现有 E/V 框架会落入 `adoption_unresolved`，还缺一个最小的聚合覆盖证书，否则用户关心的“证据来了但 MLP/残差聚合写错”会被过早弃权。

Verdict: **REVISE**

Overall score: **7.1 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 7.3 | 字面 anchor 保持：不以自然幻觉标签训练，事实区间和影响区间分开，自动回看不靠熵硬筛。但强 scope 仍未完成：事实归属不是从 frozen generator 内部独立识别，且 source-key route 不等于原始证据信息起源。 |
| Method Specificity | 7.3 | `F_G`、matched-source control、预算拆分、N/withholding、E/V gate 和输出降级规则都比第二版清楚。仍缺 native-blind 候选流、constraint-origin 追踪、跨角色 gate、聚合覆盖证书和 claim 降级。 |
| Contribution Quality | 6.5 | 作为 route-certified factuality audit 有清楚形状；作为无监督内部图 detector 还不成立。QA pipeline 本身不是新贡献，position route witness 若不加 origin mediation，不能支撑“约束信息流”强 claim。 |
| Frontier Leverage | 8.0 | 冻结 reader、冻结 observer、有限干预和同量纲 contrast control 是合适的 foundation-model-era 工具。没有新增 GNN/SAE/训练头是正确克制。 |
| Feasibility | 6.9 | 每 Claim 64 branch、每回答 256 branch 的预算现在自洽，且无新训练组件。风险在 Qwen 调用量、自然样本 c*/withholding 覆盖、matched control 可用率、origin mediation 和全层 gate 的运行成本。 |
| Validation Focus | 7.4 | 覆盖门槛、semantic-only baseline、route witness 覆盖/特异性、延续/纠正误报都对准核心问题。还需要把 position-only witness 与 origin-mediated witness 分开评价。 |
| Venue Readiness | 6.2 | 方法已能进入实现前规格冻结，但离 top-venue READY 仍有距离：贡献边界窄，内部归属不足，source-key position 与信息起源混淆，route/aggregation 覆盖未知。 |

Weighted calculation: `7.3*.15 + 7.3*.25 + 6.5*.25 + 8.0*.15 + 6.9*.10 + 7.4*.05 + 6.2*.05 = 7.12`.

## 已解决或基本接受的上一轮问题

### 1. 共享前缀完整续写目标

接受修订。`F_G` 是同一 `theta,G` 下的有限事件 log-odds，公共前缀 gate 后真实网络随 teacher-forced 分支重算。这是一个清楚的 causal estimand。它不需要被缩小成首 token，只要文稿始终保留三个限制：

```text
query_scope = shared_prefix_only
event_space = {observed event B, one enumerated alternative A}
claim = prefix intervention changes finite event preference, not free-generation outcome
```

需要继续防止的写法是“解释了后缀所有生成节点为什么错”。后缀 token 的 logp 可以分解报告，但主因果对象仍是共享前缀干预对完整事件概率比的影响。

### 2. N 类 withholding

接受方向。把缺失证据类单独改成 `commitment-vs-withholding` 是必要的，因为用户原始问题包含“来源没有当前阶段时长却生成了具体时长”这类情形。只允许 grounded slot repair 会排除核心案例。第三版没有把 withholding 冒充来源正确答案，这点正确。

仍需固定 withholding 模板带来的先验偏差。`an unspecified duration` 这类表述往往比具体数值更长、更不自然，`F_G` 会包含风格、任务偏好和非承诺表达的语言先验。这个问题不阻断定义，但阻断过强 claim。N 类只能报告：

```text
specific commitment preference over a fixed withholding alternative
```

不能报告：

```text
preference for hallucination over the correct answer
```

### 3. 同目标 matched-source control

接受修订。相比构造另一个 supported control claim，同一 `B/A` 目标下的 matched unrelated evidence group 更合理，因为 specificity 保持同量纲。当前匹配条件已经包含 role、query_scope、key 数、baseline attention mass、可见 query 比例、语义无关和哈希排序冻结，足以作为首版操作定义。

限制也必须保留：它只证明 selected message group 对该 contrast target 更特异，不证明 intervention 全局无害，也不证明任务其他陈述不变。

### 4. 预算

预算现在自洽：每风险 Claim 32 次两分支测量，即 64 branch forward；2 给 base/sham，20 给筛选，10 给最多两个最终组验证。这个安排可以实现，不再是开放式 beam search。真正风险是覆盖率，不是公式预算。

## 三个核心缺口的第三轮判定

### 1. 条件适用证据

**部分解决。** Qwen reader 的 relation-slot QA、条件引文核对、E/C/N/U、NULL 分层和 coverage 分母已经给出了可执行证据锚点。它足以作为外部语义 anchor。

**未解决。** 证据归属仍不是 native model 内部独立发现。当前 source candidates 主要由语义 reader 给出，native 图验证这些候选的采用。内部图有无自发指向正确/错误证据的信号，还没有被方法结构单独捕获。

最小补法不是训练新模型，而是增加一个 native-blind 候选流：

```text
NativeCandidateSearch(c):
  ignore semantic relevance during candidate proposal;
  partition all visible source/history units by source_id, paragraph/sentence, and role;
  run cheap root-level E/V/R screening on these units under the same F_G;
  take top native groups plus matched controls;
  only after selection ask the frozen reader whether each selected group is applicable,
  contradictory, nonapplicable, unsupported_history, or unrelated.
```

这样 Qwen 不再同时决定“哪里该看”和“哪里实际被采用”。Qwen 仍决定语义关系，但 native 图至少独立提出 adoption candidates。若 native-blind search 找不到语义相关组，必须报告内部归属失败，而不是只说语义候选无证书。

Priority: **CRITICAL**

### 2. 自动回看

**较好解决，但粒度仍偏宽。** 二叉 query 区间、父子并存、非相邻并组、matched controls、half/full 稳定和预算降级，都使自动回看从启发式变成可执行搜索。

**未解决。** 默认 32 层共同 gate 得到的是 query interval all-layers witness，不是精确 layer/path/node 回看。文稿已诚实命名 `query_group_all_layers`，可接受；但用户若要“准确自动回看”，需要最少一个层带证书，而不是只给 all-layer 宽区间。

最小补法：

```text
LayerBandRefinement:
  after an all-layer group passes,
  test fixed bands {early, middle, late} with the same full certificate
  only for the accepted group, not during screening;
  if no band passes but all-layer passes, report distributed_all_layer_witness.
```

这不增加训练组件，且避免把单层失败误读成无机制。

Priority: **IMPORTANT**

### 3. 路由衍生 hallucination 与连续 span

**部分解决。** 对 C 类是 grounded-replacement contrast；对 N 类是 commitment-vs-withholding contrast；历史延续和纠正规则比上一版更可执行。事实区间来自语义 Claim，影响区间来自固定历史干预，二者分开。

**未解决。** 连续 span 不是 native 图自动发现，而是 Qwen Claim 覆盖和关系标签的后处理。route-derived 也只对有 contrast、有证书、有 control 的样本成立。若证书覆盖率低，方法只剩 QA factuality audit 加少数 route examples。

最小补法：

```text
ClaimPolicy:
  route-derived span = semantic unsupported interval
    with at least one accepted native witness;
  semantic-only span = unsupported interval without native witness;
  unresolved interval = invalid/U/coverage/budget/control failure.
```

主表必须同时报告三类分母。不要把 semantic-only spans 混入 route-derived 成功率。

Priority: **CRITICAL**

## 仍然阻断 READY 的问题

### 1. Scope 透明但没有完全解决

文稿现在诚实写明：本方法不是证明仅靠生成器内部即可区分事实归属。这是必要修复，但意味着它只完成了较弱目标：

```text
frozen-reader anchored factuality + frozen-generator adoption measurement
```

而不是较强目标：

```text
frozen-generator-internal evidence attribution detector
```

如果项目最终论文要卖“内部无监督图检测”，当前方案不够；必须改成 route-certified audit，或加入上面的 native-blind candidate stream 来至少证明内部采用候选不是由 reader 全程指定。

Priority: **CRITICAL**

### 2. Source-key endpoint 不等于约束信息起源

当前 E/V 证书把 key 的文本位置与语义 reader 的引文区间对齐，然后测该 key/message 对 `F_G` 的影响。这个操作能证明：

```text
the selected source-position message has a finite effect on the contrast.
```

它不能证明：

```text
the raw constraint information originated from that cited source text.
```

原因是 source key 的 V 表示已经经过多层 attention/MLP/residual，可能吸收了 prompt 中更早的 task、其它 source、格式、实体共指、前文数字或全局上下文。key 落在不适用引文，不等于消息内容就是不适用证据；key 落在适用引文，也不等于适用约束真正通过这段文本进入。

最小修订有两条路线：

```text
Conservative v1:
  rename all E/V witnesses as position_anchored_route_effect
  or position_anchored_content_effect;
  do not use "constraint information flow" or "evidence origin"
  unless an upstream mediation check passes.
```

可选 origin-mediated 证书：

```text
NestedMediation(selected source span S, selected key message K, target c):
  outer gate: K -> target F_G, as current E/V witness;
  inner gate: raw source/input span S -> K's value/residual representation
              or K's local contribution to F_G;
  origin_witness iff both inner and outer effects pass,
              and removing S specifically reduces K's contrast-relevant effect.
```

如果不做 nested mediation，就必须保留 prompt-internal graph tracing 为未来工作，并把当前证书写成“position route/content effect”。这会收窄但不杀死方法：它仍可回答“模型在这些位置消息上形成了可测偏好”，但不能回答“原始约束信息从这里流入”。

Priority: **CRITICAL**

### 3. QA + route audit 的贡献边界仍薄

QAFactEval/QAGS 系列已经覆盖 answer selection、QG、source QA、answerability/filtering、answer overlap；FactGraph 已覆盖 structured semantic graph factuality。第三版把 QA 明确降为锚点和 baseline，这是对的，但论文贡献就必须由 native证书提供。

READY 前必须能写出单句贡献：

```text
Given a frozen-reader factuality anchor, measured finite interventions over
the frozen generator's native message graph identify whether an unsupported
claim was preferentially supported by source/history message groups, and
separate unsupported commitment, history continuation, correction, and
influence interval under fixed-history replay.
```

如果最终自然链条显示 witness 覆盖率低于预注册门槛，这个贡献降级为 case-study audit，不能主张完整方法。

Priority: **CRITICAL**

### 4. 跨角色 routing 仍没有证书

E gate 保留 role 总质量，只测 role 内重新分配。V gate 只测内容消息贡献。许多真实错误可能是 source role 被 history role 压过，或历史承诺整体吸收了 query，而不是在同一 role 内选错 key。当前文稿把这个标为 `cross_role_unresolved`，诚实但不足。

最小充分结构建议：增加无训练的 R gate，作为第三类证书，不需要 Q/K 分解。

```text
R gate(lambda):
  for each selected query/head/layer,
  take the source+history domain as the redistribution universe;
  scale the selected role/key mass by lambda;
  redistribute removed mass inside source+history by original proportions;
  keep source+history combined mass and all other roles fixed;
  invalidate if the redistribution domain has no remaining mass.

Delta_R = F_base - F_R(lambda=0)
```

证书命名：

```text
cross_role_allocation_witness
```

解释范围：

```text
source-vs-history allocation effect under fixed prefix, not Q/K cause.
```

这样可覆盖“历史压过来源”而不引入训练、SAE 或完整 QK attribution。

Priority: **CRITICAL**

### 5. 正确证据进来但 MLP/残差聚合写错仍缺覆盖

当前规则说：E 没有选择性效应而 V 有是内容依赖；正确证据可见但输出错且没有明确对比路由证书时，记 `adoption_unresolved`。这会把一个关键机制缺口完全交给 unresolved：source evidence 进入 attention 后，后续 MLP/residual/历史竞争把输出推向错误承诺。

最小充分结构建议：增加一个实测 2x2 evidence-MLP interaction，不做 MLP 细因果归因。

```text
States:
  F_base       = F(V_applicable on,  MLP on)
  F_noV        = F(V_applicable off, MLP on)
  F_noM        = F(V_applicable on,  MLP off)
  F_noV_noM    = F(V_applicable off, MLP off)

Delta_Vapp = F_base - F_noV
Delta_MLP  = F_base - F_noM

Omega =
  (F_base - F_noM)
  - (F_noV - F_noV_noM)
```

使用规则：

```text
correct_evidence_present(c) iff Delta_Vapp < 0
  meaning the applicable evidence message opposes the bad event B.

mlp_bad_push(c) iff Delta_MLP > 0
  meaning the MLP branch pushes the bad event B over A.

evidence_mlp_antagonism_witness(c) iff
  Delta_Vapp <= -tau_evidence
  and Delta_MLP >= tau_mlp
  and Omega >= tau_interaction
  and matched controls / repeats pass under the same F.
```

`Omega > 0` 表示 MLP 的坏偏好效应依赖适用证据消息存在；可以叫 evidence-MLP antagonism/interaction witness。它不能排除同时存在 routing_error、history override 或其它聚合路径，也不能叫唯一成因。这个结构比直接弃权更能回应用户关心的“正确证据进来了但输出仍错”，且无需新训练组件。

Priority: **CRITICAL**

### 6. 三类 message type 可以形成同一 oracle 下的最小完整结构

如果下一版把拟议补齐写进正式方法，我认为同一图 oracle 的三类 message type 足以形成首版完整可检验结构：

```text
within-role E/V:
  selected source/history key within a fixed role changes F_G.

cross-role R:
  same renormalization math as E,
  but the redistribution domain expands from one role to source+history;
  preserve source+history combined mass and leave other roles unchanged.

evidence-MLP interaction:
  2x2 V_applicable on/off x MLP on/off under the same F_G.
```

这个组合覆盖三种最重要失败：

```text
wrong within-role key was selected;
history/source role allocation favored the wrong source of information;
applicable evidence entered but downstream MLP/aggregation converted or overrode it.
```

这仍不是 pure internal factuality detector，因为语义适用性仍来自 reader anchor；但它让 native graph 不再只是解释 reader 预选的单一路由，而是真正提供三类可证伪机制测量。

关键条件是预算必须重写。64 branch/Claim 可以支持当前 E/V 证书，但不能同时承诺 within-role、cross-role R 和 MLP 2x2 interaction 都完整验证。下一版必须二选一：

```text
Option A:
  keep 64 branch/Claim,
  screen all three types cheaply,
  certify only the top one or two mechanism types by predeclared priority.

Option B:
  increase per-Claim branch cap,
  separately budget E/V, R, and MLP interaction certificates.
```

不能继续写“64 branch 足够所有证书”。delete-only 或 missing-control 的结果不能升级成完整 witness。

### 7. N/withholding 的语言先验仍可能主导

`commitment-vs-withholding` 是必要类别，但固定非承诺短语可能让 `F_G` 同时测量事实缺失、任务风格、回答流畅性和长度偏好。matched-source controls能证明某证据组特异影响该 contrast，却不能保证 contrast 本身纯粹是 factual commitment。

最小修订：

```text
For N class, report:
  commitment_preference_witness, not hallucination_repair_witness.

Use two fixed withholding templates per semantic type if both pass syntax:
  one terse, one natural sentence-preserving.
Certificate requires same sign across templates, or template_sensitive.
```

若第二个模板太贵，至少在 dev 前冻结一个 template-sensitivity sample rate，而不是只在成功样本上看。

Priority: **IMPORTANT**

### 8. 覆盖门槛需要绑定 claim 降级

第三版已有覆盖门槛：有效语义问题覆盖 ≥80%，可构造 A/B ≥70%，完整 E/V witness ≥50%。这很好。但还需要写明每个门槛失败时降级到什么 claim：

```text
semantic coverage fail -> no full factuality audit claim
contrast coverage fail -> no route-derived claim for N/C population
witness coverage fail -> no automatic lookback / continuation population claim
matched control fail -> no selective route certificate claim
```

否则低覆盖会被事后转成“我们只关注有证书样本”，这会改变问题。

Priority: **IMPORTANT**

## 对 `causal_contrast.py` 的方法层核对

当前内核与文稿方向一致：

- `ContinuationContrast` 禁止重复/互为前缀的重叠事件，计算完整 observed-vs-enumerated alternatives log-odds。
- `shared_queries` 包含预测首个分歧 token 的公共前缀 query。
- `message_gate_delta` 区分 `content` 和 `route`，route gate 保留 role attention mass，invalid normalization 直接报错。

限制也很清楚：

- 它只返回 O 前 delta，不执行完整模型重放、证书预算、matched controls、branch logp 收集或 JSON 输出。
- 它没有 branch_map、R gate、evidence-MLP interaction、layer-band refinement 或 native-blind candidate search。
- 它验证的是计算对象可以编码，不是方法在自然样本上有效。

这不扣方法定义分，但不能在论文或报告里当作实证支撑。

## 最少必要修订

1. 加入 native-blind candidate stream：native 图先独立提出 adoption candidates，再由 reader 判定这些候选与 Claim 的语义关系。不要让 Qwen 同时决定事实和全部待测位置。
2. 将当前 E/V 证书收窄命名为 `position_anchored_route_effect` / `position_anchored_content_effect`；若要宣称 constraint-information origin，增加 nested mediation 或 prompt-internal tracing。
3. 增加跨角色 R gate：复用 E gate 的重归一数学，但把允许重分配域从单 role 扩为 source+history，保持二者合质量和其它 role 不变。
4. 增加 evidence-MLP 2x2 interaction witness：`V_applicable on/off × MLP on/off`，用 `Delta_Vapp`、`Delta_MLP` 和 `Omega` 判断 evidence-MLP antagonism；只称 interaction，不称唯一成因。
5. 保留 `F_G` 为主目标，不退成首 token；但所有输出名必须包含 `shared_prefix_event_preference` 或等价字段，避免解释范围膨胀。
6. 对 N 类改名为 `commitment_vs_withholding`，并冻结 withholding template sensitivity 处理；不得与 grounded replacement 合并报 route-derived repair。
7. 重写预算：64 branch/Claim 只能 certify 预声明优先级下的有限机制类型；若要同时验证 E/V、R、nested mediation 和 2x2 MLP，需要增加预算或分阶段选择。
8. 把覆盖门槛和 claim 降级规则绑定：低 coverage 时方法 claim 自动降级，不能靠 unresolved 改名成成功。
9. 分开报告 `semantic_only_unsupported`、`position_route_witnessed`、`origin_mediated_witnessed`、`cross_role_witnessed`、`content_witnessed`、`evidence_mlp_interaction_witnessed` 和 `unresolved` 的分母。

## Drift Warning

有残余 scope drift 风险，但已经不是隐性漂移。文稿透明承认当前方案不是 pure internal unsupervised detector。若最终目标是“内部图独立识别事实归属”，当前方案仍应判 **RETHINK**；若目标是“不用自然幻觉标签训练，以冻结 reader 锚定事实并测 frozen generator 的消息采用”，当前方案是可继续修的 **REVISE**。

## Simplification Opportunities

1. 不新增训练组件。native-blind candidate search、R gate、evidence-MLP 2x2 interaction 都是有限干预算子，不需要 GNN、SAE 或 learned segmenter。
2. 对首版只支持三类证书：within-role route/content、cross-role allocation、evidence-MLP interaction。Q/K 和具体 MLP neuron 归因继续作为后续诊断。
3. 不扩充 QA reader pipeline。当前语义锚点已经足够复杂；新增工作应在 native measurement coverage，而不是更多语义 prompt。

## Modernization Opportunities

NONE as trainable modules. 第三版的现代性来自 frozen-reader semantic anchoring、frozen-observer finite interventions、matched same-target controls 和 explicit certificate taxonomy。继续加模型会削弱主线。真正需要补的是内部测量覆盖，而不是更强 reader。

## 下一轮 READY 条件

READY 不要求自然性能结果已经出来，但要求方法对象完整到可实现且不会改名漂移：

- scope 句固定：不是 pure internal detector，而是 frozen-reader anchored route-certified audit。
- native-blind candidate stream 写进算法。
- E/V 证书命名收窄为 position anchored；origin-mediated claim 需要 nested mediation 或明确放弃。
- R gate 与 evidence-MLP 2x2 interaction 至少有操作定义、预算和证书降级规则。
- `F_G`、matched controls、N/withholding、预算和 coverage 门槛保持冻结。
- 输出 taxonomy 能直接落成 JSON，且每类分母不会互相偷换。
- 若 coverage 低，claim 自动降级为 semantic QA audit with sparse route certificates。

Final verdict: **REVISE**。第三版已经是可实现的研究方法雏形，不再是初稿那种不可识别的训练式图模型；但它仍不是 READY。最小下一步不是加实验菜单，而是补上 native-blind 候选、source-key position 与 information-origin 的区分、跨角色 R gate、evidence-MLP 2x2 interaction 和预算/claim 降级规则，让内部机制测量真正承担一部分归属工作，而不是被外部 QA 完全指定。
