# soft_graph_energy.py 有界工程复核

日期：2026-09-13  
模式：`code-review-and-quality` 有界 CPU 规格复核；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。范围仅审查 `route_graph/soft_graph_energy.py` 与已冻结的 fixed-energy 规格。未改实现、未启 GPU。运行了只读导入和两个手写 CPU 例子来验证边界行为；未创建测试文件。

## Verdict

**Request changes before using this as the driver for soft graph predictions.**

实现已经正确落住了几条核心收窄：unknown 没有负 logit；`qN_global` 只在 full-source soft prediction 中进入；local candidate 只允许 S/C/U；native 负效应不会影响 risk；history 已乘 previous confidence；四个 baseline 都同步输出。文件规模小、接口清楚，没有安全或性能上的明显问题。

但仍有 3 个 Required 会造成假高置信或错误传播，另外 1 个 confidence 语义必须在输出合同中固定，否则下游很容易把 coverage confidence 当成预测置信度。

## 已验证正确的点

- `distribution()` 要求有限、非负、归一且标签集合精确匹配，能阻止 bool/NaN/漏标签输入。
- `positive_dependence()` 只在 `sham_exact`、`prefix_exact`、`controls_valid`、至少两个 controls、`measurement_scope == raw_origin_mediated_target_dependence`、`repeat_valid` 全满足时给正权重；否则返回 0。
- source term 去重使用完整 key，local relation 没有 N 入口，`signed_relation_logit = log(C/S)` 被 clip 到 ±2。
- full graph 的 history beta 已乘 `prior["confidence"]`，避免上一条几乎全 U 但 raw logit 大时无衰减传播。
- `infer_response()` 按 span 顺序构建 previous registry，拒绝未来或重叠 history edge。

## Required 1：history 不能用 argmax reuse 代表“明确 reuse”

位置：`soft_graph_energy.py:84-89`。

当前逻辑：

```python
label = max(probs, key=probs.get)
beta = native * probs["reuse"] * scope if label == "reuse" else 0.
```

这会在关系分布完全打平时传播 reuse，因为 `distribution()` 返回的标签顺序让 `max()` 选中 `reuse`。手写 CPU 例子中，`q_relation={reuse:0.2, correction:0.2, quote:0.2, new_topic:0.2, unknown:0.2}` 仍产生 `eligible=True`、`beta≈0.193`、当前 risk 从 0.5 推到 0.518。幅度不大，但它违反“明确 reuse 而非 correction/quote/new_topic”的规格。

Required 修复：history edge 必须有独立的 known-reuse gate，不能只靠 argmax。最小可编码规则二选一：

```python
reuse_known = edge.get("relation_known") is True and edge.get("edge_label") == "reuse"
```

或若只能用 logits/probs：

```python
reuse_known = probs["reuse"] >= 0.5 and probs["reuse"] - max(probs[k] for k != "reuse") >= 0.25
```

阈值若采用必须冻结，不能用 RAGTruth 调。Correction、quote、new_topic、unknown、平局和低 margin 都应 `beta=0`，只保留 localization/diagnostic。

## Required 2：source candidate `pi` 只检查单项，不检查每个 response role 的边际和

位置：`soft_graph_energy.py:56-75`。

当前只拒绝完整 key 重复：

```python
key = tuple(item["key"])
if key in seen: raise ValueError(...)
```

但同一个 `response_role_id` 可以提交多个不同 source/event key，每个 `pi=1`。手写 CPU 例子中，一个 role 的 3 个 conflict source terms 全部 `pi=1`，full graph 直接 `confidence=1.0`、`risk_score≈0.881`。如果这些 `pi` 本应是 matcher assignment marginals，这就是重复路径/候选质量错误导致的假高置信。

Required 修复：解析 key 的结构并按 role 检查边际质量。

```python
role_id, source_node_id, source_event_or_record_id = item["key"]
role_mass[role_id] += pi
if role_mass[role_id] > 1 + 1e-6: raise ValueError(...)
```

如果 source_terms 只包含非 NULL 候选，允许 role_mass < 1；缺失质量代表 NULL/UNKNOWN，不应补成 confidence。不同 response roles 可以各自有质量；同值不同事件仍保持不同 key，但不能让同一 role 的多个 beam 路径重复放大。

## Required 3：native 风险资格还需要显式 raw-origin 和双对照身份

位置：`soft_graph_energy.py:20-33`。

当前用 `measurement_scope == "raw_origin_mediated_target_dependence"` 代表 raw-origin-mediated，并用 `controls_valid=True` + `len(control_deltas)>=2` 代表双合格对照。这接近规格，但对工程驱动层还不够硬：上游若错误复用一个 control 两次、或只设置 scope 字符串但没有 raw origin mediation artifact，本函数仍会给风险权重。

Required 修复：输入合同至少增加并检查：

```python
native["raw_origin_mediated"] is True
native["control_ids"] has at least two distinct IDs
native["control_statuses"] all == "passed" for the used controls
native["origin_artifact_ref"] or equivalent nonempty provenance ref
```

如果父代理决定 `measurement_scope` 是唯一 canonical raw-origin 证明字段，也要在 spec 和 artifact schema 中明示它由 D 阶段 raw-origin mediation validator 生成，普通调用方不能手写该字符串。否则这会成为假 native-specific 权重入口。

## Required 4：`confidence` 当前是 evidence/coverage confidence，不是预测确定性

位置：`soft_graph_energy.py:103-106`。

当前：

```python
conf = 1 - (1 - global_confidence) * (1 - ks) * (1 - khist)
risk_score = 0.5 + conf * (sigmoid(Lraw) - 0.5)
```

这符合“unknown shrink”的数学用途，但名字容易误导。若 `qS=0.5,qC=0.5,qU=0`，risk 仍为 0.5，而 confidence 可以接近 1；这不是错误的 energy 计算，却不是“模型很确定标签”。它表示 relation/source/history 非 U 覆盖充足但方向中性或冲突。

Required 合同修复：下游字段名或文档必须写成 `evidence_confidence` / `coverage_confidence`，并另报 `directional_confidence = confidence * abs(2*risk_raw - 1)` 或至少不要把 `confidence` 当 calibrated certainty。若保持字段名 `confidence`，merge/evaluation 必须把它解释为 unknown shrink weight，不作为 precision/abstention certainty。

## Optional：输入 shape 可更早显式验证

`claim["span"]`、`item["key"]`、`edge["origin_scope"]` 目前靠运行时索引/KeyError 暴露。作为驱动层可以接受，但更好的错误信息是显式验证：span 为两个递增 int；source key 长度为 3；origin_scope 属于固定集合。这个不是当前 blocker。

## CPU 边界验证

我运行了两个小例子：

1. 均匀 history relation 分布仍被选为 `reuse` 并传播，确认 Required 1。
2. 同一 role 三个 `pi=1` source terms 被接受并把 confidence 推到 1，确认 Required 2。

未发现 GPU 调用、外部 IO 或训练路径。没有发现现成 `soft_graph_energy` 测试；后续应补最小 CPU tests 覆盖上述两个 Required、native qualification 失败、negative native effect risk 权重为 0、unknown shrink。

## Final status

- Critical: 0
- Required: 4
- Optional: 1

实现方向可以保留，但在修复 Required 1-3、并固定 Required 4 的报告语义前，不应作为 full-graph detector 的冻结评分引擎。修完后再做一次小范围 CPU 复核即可，不需要 GPU。
