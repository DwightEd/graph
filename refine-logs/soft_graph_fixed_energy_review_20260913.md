# Soft graph fixed energy 收窄复核

日期：2026-09-13  
模式：`research-refine` 有界规格复核；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。范围只复核 B 主线的 4 个收窄改动，并给出固定可编码能量规格。未训练、未启 GPU、未改代码。

## 结论

这 4 个改动应采纳。它们修正了上一版软图检测器里最危险的三处漂移：unknown 被当成更正确、correction 被错误地反向传播、native 负效应被误用成“当前错误 claim 更可信”。收窄后的 B 可以作为主检测器规格：全 claim 输出风险分数和定位排序，A/B 证书只作为 overlay。

它仍不是无监督真值证明。风险分数是固定因子图预测，关系估计错就会预测错；必须用 RAGTruth/人工标签独立评价，不能用证书子集自证。

## 接受的 4 个收窄

1. **删除 `-λ_unknown * pU`。** Unknown 不提供任何方向性证据，只能降低 confidence，并把最终 risk 向 0.5 收缩。不能因为 relation unknown 就默认更 supported。

2. **history correction/quote/new_topic 阻断 reuse。** Correction 不自动翻转 previous risk；两个互相否定的句子都可能不受来源支持。只有明确 `reuse` 的前向边才传播 previous risk。Quote、new topic、elaboration without factual reuse 都不传播。

3. **full-source N_soft 可进入预测，不进入证书。** 全来源 finite verifier 的 S/C/N/U 都是预测；`qN_global` 可以进入 detection logit。certificate 层仍要求完整 absence 核查。局部 candidate 的 not-stated 只能映射为 U，不能由片段缺失推出全文 N。

4. **native 负效应不改判 correctness。** 负向 target effect 保留为 localization/diagnostic，首版风险调制只用正向、对照调整后的 target-dependence 权重。Relation unknown 时 native 只影响 localization/confidence，不影响 risk 方向。

## 固定变量与输入

对每个 claim/event `i`：

- `q_global_i = {qS, qC, qN, qU}`：冻结 Qwen full-source finite verifier 的预测分布，输入完整 source + response claim/event，不读 native effect、不读标签、不生成 free-text source answer。四项归一；invalid 时设 `qU=1`。
- `Z_i`：matcher/owner 给出的 frozen source candidate 集合，含 NULL/UNKNOWN、同值多事件竞争、未搜索/剪枝统计。
- `π_iz`：candidate 权重，来自固定 matcher cost 或 source-owner pointer score。若首版不训练 owner，则由 matcher softmax 产生。不得使用 hallucination label、RAGTruth、A/B F 或 native Δ。
- `q_local_iz = {qS, qC, qU}`：冻结 Qwen 对候选 ID 的 finite relation 预测。局部 not-stated、unrelated、condition_failed、candidate_absent、invalid 全部并入 `qU`。
- `w_src_iz`：source candidate 对原始 claim B 的正向、对照调整 native dependence 权重。
- `E_i`：历史候选边集合，只含 `j < i`。
- `p_reuse_e`：冻结 finite history verifier 对边 `e=(j,i)` 为 reuse 的预测；correction/quote/new_topic 不进入 reuse。
- `w_hist_e`：history group 对当前 B 的正向、对照调整 native dependence 权重。

所有变量按响应顺序计算；history 不看未来。

## Native 正向权重

对 claim B 的原始完整 logP，不做长度均值：

```text
Δ_B(g) = logP_base(B) - logP_gated(B)
```

正值表示 group g 推动当前已生成 claim B。对 matched controls：

```text
control_pos(g) = max_c max(0, Δ_B(c))
δ(g) = Δ_B(g) - control_pos(g)
w_native(g) = max(0, tanh(δ(g) / 0.5))
```

0.5 nats 固定。没有合格 control、sham 失败、common-prefix 边界失败、feature/cache hash 不一致时：

```text
w_native(g) = 0 for risk
native_edge_status = measured_uncontrolled_or_invalid
```

该边仍可出现在 localization diagnostics，但不能调制 risk。

负效应：若 `Δ_B(g) < 0` 或 `δ(g) < 0`，记录 `suppresses_B`，不进入 `w_native`，不把当前 claim 判得更正确。

## Candidate 权重与重复计数

为避免 top-k assignment 重复计数，风险项必须在唯一 role-candidate key 上聚合，而不是遍历所有 beam assignment。

唯一 key：

```text
k = (response_role_id, source_node_id, source_event_or_record_id)
```

如果同一 key 出现在多个 beam assignment 中，`π_ik` 取 assignment softmax 的边际和，最多 1。若同一 source node 映射到同一 response role 的多个等价 candidate id，先按 key 合并。不同 response roles 可以各自计入，因为多角色错误是合法 event-level 信号；但同一 role 不能因多条 beam 路径放大。

同值不同事件不合并。它们必须保留不同 `source_event_or_record_id`，并进入 ambiguity/confidence。

## 剪枝、unknown 与 candidate 质量

剪枝和未搜索不能改变 risk 方向，只能降低 graph confidence 或把 source/history 项置零。

每个 role pool 计算：

```text
pool_quality_r = 1.0
if unsearched_raw_unit_ids not empty or feature_missing: pool_quality_r = 0.0
elif same_value_competition outside retained pool: pool_quality_r = 0.5
elif near_tie_pruned_candidate exists: pool_quality_r = 0.5
```

`near_tie_pruned_candidate` 定义为：同 role/value type，未保留，且 unary cost 不高于最佳保留非 NULL candidate 加 0.05。0.05 是固定 matcher-cost margin，不用标签调。

source risk 权重使用：

```text
α_ik = π_ik * pool_quality_role(k) * w_src_ik * (qS_local_ik + qC_local_ik)
```

若 `pool_quality=0`，该 role 的 source graph 项不影响 risk；仍报告 candidate coverage failure。若 `pool_quality=0.5`，可以有弱调制，但 confidence 也反映 ambiguity。NULL/UNKNOWN candidate 不进入 `Lsource`。

## 固定能量

设 `eps = 1e-6`，`clip2(x)=min(2,max(-2,x))`。

### Global full-source term

```text
Lglobal_i = log((qC_global_i + qN_global_i + eps) / (qS_global_i + eps))
Kglobal_i = qS_global_i + qC_global_i + qN_global_i
```

`qN_global` 是 soft detection 预测，可以提高 unsupported risk。它不是 certificate N。

### Source adoption term

```text
r_ik = clip2(log((qC_local_ik + eps) / (qS_local_ik + eps)))
α_ik = π_ik * pool_quality_role(k) * w_src_ik * (qS_local_ik + qC_local_ik)
Lsource_i = Σ_k α_ik r_ik / max(1, Σ_k α_ik)
Ksource_i = min(1, Σ_k α_ik)
```

解释：只有 source candidate 与 claim 有已知 support/conflict relation，且对应消息正向推动 B，并通过 matched controls，才调制 risk。Relation unknown、local not-stated、unrelated、condition failed 均不给方向。

如果 relation says support 且 native 正向推动 B，`r_ik < 0`，risk 降低；这是“模型依赖了支持 B 的来源”。如果 relation says conflict 且 native 正向推动 B，risk 增加。Native 本身不判真。

### History reuse term

历史边只允许 reuse：

```text
eligible(e=j->i) = e.direction == "forward" and e.label == "reuse"
```

Correction、quote、new_topic、unknown、future edge 全部 `eligible=false`。

对 eligible edge：

```text
β_e = w_hist_e * p_reuse_e * relation_known_e * origin_scope_quality_e
lprev_j = clip2(L_final_j_before_shrink)
Lhistory_i = Σ_e β_e lprev_j / max(1, Σ_e β_e)
Khistory_i = min(1, Σ_e β_e)
```

`relation_known_e=1` 仅当 previous/current slot or event relation 有 frozen finite verifier 支持；否则 0。`origin_scope_quality_e=1` for mapped previous slot, `0.5` for mapped previous event, `0` for unmapped history dependence. 这不是新模块，只是把已有 slot/event scope 写成权重门。

如果 previous claim 本身低风险，reuse 可降低当前风险；如果 previous claim 高风险，reuse 可提高当前风险。Correction 不反向使用 previous risk。

### Final logit and shrink

```text
L_raw_i = Lglobal_i + Lsource_i + Lhistory_i
K_i = 1 - (1 - Kglobal_i) * (1 - Ksource_i) * (1 - Khistory_i)
risk_raw_i = sigmoid(L_raw_i)
risk_i = 0.5 + K_i * (risk_raw_i - 0.5)
confidence_i = K_i
unknown_mass_i = 1 - K_i
```

Unknown 只通过 `K_i` 收缩到 0.5，不作为负 logit。`risk_i` 不称 calibrated posterior；只称 fixed-energy risk score。校准必须外部评价。

## History 方向规则

History 必须满足：

- `j < i`，不能用未来 claim；
- current relation finite verifier 明确 `current_slot/event_derives_from_previous_slot/event`；
- edge type 为 reuse；
- native history group 对当前 B 有正向、specific、controlled target dependence；
- quote/correction/new_topic 阻断传播，即使文本相似或 attention 高。

Supported recovery 另报 certificate/diagnostic，不进入 reuse propagation。不能因为当前句纠正前句就把 previous risk 自动取负。

## 输出合同

每个 claim 输出：

```json
{
  "claim_id": "...",
  "soft_graph_energy_version": "fixed_energy@20260913",
  "q_global": {"S": 0.0, "C": 0.0, "N": 0.0, "U": 1.0},
  "Lglobal": 0.0,
  "source_terms": [
    {
      "key": ["response_role_id", "source_node_id", "source_event_or_record_id"],
      "pi": 0.0,
      "pool_quality": 1.0,
      "w_native_positive_specific": 0.0,
      "q_local": {"S": 0.0, "C": 0.0, "U": 1.0},
      "alpha": 0.0,
      "signed_relation_logit": 0.0,
      "status": "used|relation_unknown|native_uncontrolled|pool_ambiguous"
    }
  ],
  "history_terms": [
    {
      "previous_claim_id": "...",
      "edge_label": "reuse|correction|quote|new_topic|unknown",
      "eligible": false,
      "w_native_positive_specific": 0.0,
      "previous_logit_clipped": 0.0,
      "beta": 0.0,
      "origin_scope": "mapped_previous_slot|mapped_previous_event|unmapped"
    }
  ],
  "Lsource": 0.0,
  "Lhistory": 0.0,
  "Lraw": 0.0,
  "confidence": 0.0,
  "unknown_mass": 1.0,
  "risk_score": 0.5,
  "scope": "soft_detection|target_dependence_only|certificate_overlay",
  "certificate_ids": [],
  "failure_reasons": []
}
```

Batch outputs must also include the four synchronized baselines:

1. `qwen_fullsource_no_graph`: uses only `Lglobal` and the same shrink from `Kglobal`.
2. `qwen_matcher_no_native`: uses `Lglobal + source relation terms` with `w_native=1` only for relation-known candidate weighting, or equivalently reports matcher relation aggregation without native; predefine one of these and keep fixed.
3. `graph_no_history`: uses `Lglobal + Lsource`, no history propagation.
4. `full_graph`: uses all three terms.

RAGTruth or human labels only evaluate these outputs; they do not tune weights, thresholds, native scale, candidate margins, or checkpoint.

## Remaining risks

- `Lglobal` can dominate if Qwen full-source finite verifier is overconfident. This is acceptable only because the no-graph baseline is synchronized; if full graph does not improve over it, graph claim fails.
- Candidate pruning can still hide the correct source. The required pool-quality and same-value ambiguity fields prevent false high confidence but do not solve recall.
- History propagation can amplify an earlier wrong prediction. That is a model risk, not a definition bug, as long as it is forward-only, reuse-only, normalized, and independently evaluated.
- Soft `N_global` may be wrong because full absence is hard. That is allowed for detection; certificate N remains stricter.

## One smaller fallback if this is still too much

If implementation time forces one further cut, keep only:

```text
L = Lglobal + Lsource
```

and report history edges as localization diagnostics. Do not drop the distinction between prediction and certificate. This fallback is smaller than adding another learned head and still lets graph affect detection through relation-known positive source dependence.

## Final Required status

The proposed fixed energy is acceptable after these coding constraints are written into the spec:

1. Unknown shrink via `K_i`; no unknown logit penalty.
2. Full-source `qN_global` allowed for soft detection; local candidate not-stated folded into U.
3. Positive, control-adjusted native dependence only; negative effects localization-only.
4. Source terms deduplicated by `(response_role_id, source_node_id, source_event_or_record_id)`.
5. Candidate pruning and same-value ambiguity lower `pool_quality/confidence`, never risk direction.
6. History uses only forward reuse edges; correction/quote/new_topic block propagation.
7. Weights fixed: `λ_source=λ_history=1`, native scale `0.5` nats, clip bound `2`, eps `1e-6`; no hallucination-label tuning.
8. Four baselines emitted with identical Qwen verifier artifacts.

With these constraints, B is a codable full-coverage detector while A remains the high-confidence certificate layer. It does not solve calibration or verifier error by definition; those must be measured on independent labels.
