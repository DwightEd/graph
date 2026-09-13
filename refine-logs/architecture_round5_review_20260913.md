# Architecture Round 5 Final Method Review

日期：2026-09-13

## 审查方式

Codex MCP 不可用。本文件是协作代理的 GPT-5.5 xhigh 风格备用方法审查，按 `research-refine` 默认第5轮上限执行；它不是缺失 MCP 的外部 verdict，也不是实验审计。未跑 GPU，未改代码、冻结 RAGTruth 结果、state 或 manifest。审查输入为：

- `refine-logs/architecture_round4_20260913.md`
- `refine-logs/architecture_round4_review_20260913.md`
- `refine-logs/grounding_architecture_lit_20260913.md`
- `refine-logs/attribution_architecture_lit_20260913.md`
- `refine-logs/unsupervised_detector_lit_20260913.md`
- `route_graph/causal_contrast.py`
- `route_graph/native_audit.py`
- `tests/test_causal_contrast.py`
- `tests/test_native_audit.py`

CPU tests and HF tiny-model checks只说明局部算子有工程可实现性；它们不是自然样本有效性结果。

## 总体判断

第5轮后，方法规格已经可以进入实现。修订4把第4轮剩余的真正定义缺口基本补齐：paired R 有同源 H、同移出质量、不同 destination 的 matched control；origin mediation 有 position/origin/mismatch 三分法和全分母；MLP 2x2 增加 slot/context 四态分解和 `broad_MLP_event` 降级；预算按 256 forward/Claim 重新记账，并固定机制顺序、pending N 和失败分母；donor 改成每个 B/A 完整同 shape 重放，避免 bf16 长度差 sham 问题。

这不是论文或实证 READY。没有自然完整链条结果，就不能给 9 分或 READY verdict。当前能通过的是“specification ready to implement as a mechanism teacher”：目标、干预、证书、预算、降级和输出分母已经具体到工程可以开始实现。不能通过的是“scientific/paper ready”：语义 reader 是否可靠、native-blind 证书覆盖率是否足够、origin mediation 是否常见、MLP interaction 是否不是少数 case、256/Claim 成本是否能承受，都还没有自然数据证据。另有一个实现前必须修正的执行顺序小洞：当前阶段叙述仍像 Qwen 先完成全部语义，再由 Llama 验证；native-blind 候选要求 Llama 先冻结候选/control pool，再让 Qwen 只标这些冻结对象。

Verdict under `research-refine` rubric: **REVISE**

Specification verdict: **READY TO IMPLEMENT**

Paper/empirical verdict: **NOT READY**

Overall score: **8.3 / 10**

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8.8 | 满足原始锚点：不以自然幻觉标签训练，测来源/历史采用，事实区间和影响区间分开，N 类 withholding 保留缺失时长案例。也明确不宣称纯内部事实归属已解。 |
| Method Specificity | 8.7 | `F_G`、E/V、paired R、nested mediation、MLP 2x2、slot/context 分解、预算、pending N、coverage 和输出分母都已定义。剩余只是符号/JSON层的小钉子，不是概念空洞。 |
| Contribution Quality | 7.8 | 贡献已经收敛为 frozen-reader anchored native mechanism certificates，而不是 QA pipeline 或 GNN。新意能成立，但依赖自然 witness 覆盖率和与 semantic-only baseline 的差异。 |
| Frontier Leverage | 8.5 | 冻结 reader、冻结 observer、有限事件干预、同目标 controls、path-specific mediation 和机制教师优先于 GNN 蒸馏，技术路线合适。 |
| Feasibility | 7.6 | 无新训练组件，HF 后端已覆盖部分真实 replay。256/Claim 机制教师成本很高，但作为小规模 source-disjoint dev 诊断是可执行的；不适合直接全量部署。 |
| Validation Focus | 8.5 | 下一步验证已经对准自然完整链条、coverage、语义锚点、native witness、延续/纠正误报和成本，不再是 trivial relation experiments。 |
| Venue Readiness | 7.1 | 方法规格接近可投稿前的实验设计，但没有自然结果、覆盖率和消融证据，不能叫 paper-ready。 |

Weighted calculation: `8.8*.15 + 8.7*.25 + 7.8*.25 + 8.5*.15 + 7.6*.10 + 8.5*.05 + 7.1*.05 = 8.26`.

## 已解决的定义问题

### 1. Scope 不再是方法漏洞

当前写法符合用户字面要求：不以自然幻觉标签训练。用户没有禁止使用来源文本或冻结语义模型。方法也没有偷换成“reader 已经证明 native 内部归属”。保留的正确边界是：

```text
This method is hallucination-label-free and frozen-reader anchored.
It measures native generator adoption under explicit semantic targets.
It does not solve pure generator-internal factual attribution.
```

这已经是合格 scope，不应继续作为硬 blocker。

### 2. Paired R 解决了 broad R 的选择性问题

Broad R 仍只能是 allocation diagnostic。正式 paired R 则有可比控制：

```text
H fixed
removed mass fixed
query/layer/head fixed
destination S vs matched unrelated C
same F_G
```

这能支持 `selective_cross_role_effect`。它仍不定位 Q/K 原因，但不需要为首版补 Q/K attribution。

### 3. Source-origin mediation 的身份自洽

donor input `S` 缩放、捕获指定 key `K` 的 V、recipient 只替换 `A_ij V_j` 中的 V，并用同一 `F_G` 测量。这是清楚的 V-message-mediated input path。每个 B/A 完整同 shape donor replay 也修掉了 bf16 sham 风险。

它允许的 claim 是：

```text
input_origin_mediated_effect through selected V-message channel
```

它不允许的 claim 是：

```text
complete information origin or natural total causal effect
```

修订4已经保留 `origin_nonexclusive=true` 和三分法；足够进入实现。

### 4. MLP 交互判据现在足够窄

`Delta_E <= -.5`、`Delta_M >= .5`、`Omega >= .5` 的方向与 `F = bad-over-alternative` 一致。slot/context 四态分解也补上了关键保护：如果上下文项主导，就降级为 `broad_MLP_event_interaction`，不冒充槽级聚合覆盖。

这个机制可以回应“正确证据进来了但聚合写错”的用户缺口，同时不声称唯一成因。

### 5. 预算现在是机制教师预算，而非部署承诺

256 forward/Claim 的表格闭合，且明确 branch、donor、recipient 都按真实模型调用计费。固定顺序和 pending N 防止先报成功后跳过模板敏感性。它可以作为机制 teacher 的 source-disjoint dev 预算，不应被包装成低成本全量审计。

## 余下 Required 是什么性质

### A. 真正的定义小修

这些应在实现前补进最终方法文档，但不是概念 blocker：

1. 明确 `Delta_pairedR(H -> S) = F_base - F_pairedR(H -> S)`，并写清正号对应“把质量从 H 移到 S 降低坏事件偏好”还是相反。当前文字可读，但符号还应机器化。
2. 给最终 JSON 固定枚举名：`semantic_only_unsupported`、`position_effect`、`origin_mediated`、`origin_mismatch`、`within_role`、`paired_cross_role`、`MLP_interaction`、`broad_MLP_event`、`pending_N`、`unresolved`。
3. 冻结 Qwen prompts、withholding templates、source unit segmentation、matched-control hash ordering 和 budget priority as config，避免实现时滑动。
4. 把执行顺序改成四阶段，防止 native-blind 泄漏并适配单卡调度：

```text
A. Qwen: output Claims, no-answer-leak source QA, and freeze B/A targets.
B. Llama: native-blind candidates plus quality-matched control pool,
   with no semantic candidate labels yet; persist frozen IDs.
C. Qwen: label only frozen candidates/pool semantically;
   do not receive Delta, F, native ranks, or mechanism verdicts.
D. Llama: certify fixed candidates and eligible controls:
   position/origin/pairedR/MLP/layer-band.
CPU: aggregate intervals and denominators.
```

`D` 失败不能回到 `B` 重新挑更容易成功的组。

这些是规格钉子，不需要第6轮方法扩张。

### B. 不是定义漏洞，而是必须实证验证

这些不能靠继续改方法文本解决：

1. Qwen evidence anchor 在自然 RAGTruth train/dev source-disjoint split 上的 coverage 和错误率。
2. native-blind candidates 是否能在足够比例风险 Claim 上找到位置/route/content witness。
3. origin mediation 能否把 position witness 升级为真实 input-mediated witness，而不是大多 `origin_mismatch`。
4. paired R 是否有足够 matched controls，且不是只产生 effect-only。
5. MLP interaction 是否槽项主导，还是多数降级为 `broad_MLP_event`。
6. 256/Claim 在一张 4090 上的真实延迟、显存、失败恢复和样本吞吐。
7. semantic-only baseline 与 full native certificate audit 在 route-derived continuation/correction 上是否有实际增量。

这些是下一步 GPU/自然链条任务，不应作为继续重写方法的理由。

## 三个用户缺口的最终方法审查

### 1. 当前证据的条件归属

在允许 frozen reader 的 scope 下，**规格已解决**。relation-slot QA、条件引文、E/C/N/U、NULL 分层、withholding 和 source-origin mediation 形成完整接口。纯内部独立归属仍未解决，但文稿已经把它列为更强未证问题。

### 2. 自动回看位置

**规格已解决到可实现层级**。native-blind candidate stream、query tree、E/V、paired R、layer-band optional refinement、position/origin 分母，已经避免 entropy/top-k 和手工候选。准确率和覆盖率待自然验证。

### 3. 路由衍生错误、延续、纠正和连续 spans

**规格已解决，实证未解决**。C 类 grounded replacement、N 类 commitment-vs-withholding、history input mediation、paired cross-role、MLP interaction、fact interval vs influence interval 的分离都已定义。成功与否取决于自然 coverage 和 semantic anchor 可靠性。

## 对当前实现状态的判断

`causal_contrast.py` 与 `native_audit.py` 已经支持进入实现的关键底座：

- finite continuation contrast
- E/V message delta
- paired transfer destination
- native Llama eager replay hooks
- MLP gate
- donor value capture/replacement
- sham/recompute/hook cleanup 的 CPU 不变量

但完整系统还没有实现：

- Qwen evidence anchor
- native-blind search scheduler
- matched-control selector
- paired R full certificate orchestration
- nested mediation budget runner
- MLP 2x2 slot/context decomposition
- coverage accounting
- final audit JSON and batch pipeline

所以当前状态是：**kernel/backend ready, pipeline not yet implemented, no natural validity evidence**。

## Simplification Opportunities

NONE for method structure. 不要再加 GNN、SAE、Q/K full tracing、AMR 或新训练头。下一步应实现机制教师并跑自然链条；若教师信号有效但太贵，再蒸馏。

## Modernization Opportunities

NONE as additional modules. 现代性已经足够：frozen semantic target、native finite interventions、same-target controls、path-specific mediation、mechanism teacher before student。

## Final Status At Skill Round Limit

This is the best current method specification. It should stop refining and move to implementation/experiment planning.

Required before implementation:

- apply the definition small fixes above;
- apply the four-stage execution-order fix above;
- freeze prompts/templates/config/schema;
- implement missing pipeline modules.

Required before paper/scientific READY:

- source-disjoint natural complete-chain results;
- coverage and cost report;
- semantic-only vs full native certificate comparison;
- failure breakdown for unresolved/origin_mismatch/broad_MLP_event/pending_N.

Final verdict: **REVISE** under the paper-review rubric, because no natural evidence exists and overall score is below 9. For engineering handoff, verdict is **READY TO IMPLEMENT AS A MECHANISM TEACHER**. Further method-only rounds are unlikely to improve the proposal more than actual implementation and natural-chain measurement will.
