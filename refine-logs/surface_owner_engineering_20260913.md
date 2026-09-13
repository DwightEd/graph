# Surface graph / masked owner engineering review — 2026-09-13

范围：`next_iteration/surface_graph.py`、`next_iteration/surface_owner.py` 与现有 surface-graph CPU tests。未运行 GPU、未修改冻结模块；owner 尚未接入 runner 或产生检测结果。

## 结论

**Critical：0**

**Required：2**

1. **`masked_dense` 特征没有在 matcher 入口绑定到实际 capture artifact、数组字节或冻结模型身份。** `capture_masked_contexts()` 返回 records，其中包含 document/input IDs/model identity，但 `match_owners()` 只接收可伪造的 `{document_id: array}` 映射，检查 key 存在和形状/有限性后就写出 `dense_feature_stage_executed=true`（[surface_owner.py](../next_iteration/surface_owner.py#L66)）。它没有读取或验证 records、document/input IDs、layers、representation、array hash 或 model identity。因此其它模型、非 masked document 的向量或手工数组可被报告为本轮 masked-owner capture。将 capture 结果封存为一个 manifest：每个 document ID、原始 input IDs/hash、layers/representation、model identity、array byte hash；让 `match_owners()` 接受该 manifest+arrays 并逐项验证后才允许 `masked_dense`。`lexical_only_preflight` 应继续明确标识，不得升格。

2. **当前独立 Ruff 检查仍失败。** [surface_owner.py](../next_iteration/surface_owner.py#L148) 在 nested hook 的默认参数中调用 `len(batch)`，触发 Ruff B008；本次独立检查显示 `Found 1 error`。移至函数体或预先赋给局部常量，并将 owner tests 和 Ruff 一起作为入口验收。

## 已确认

- surface graph 保留完整原文 base/slot spans、sealed graph 和非 target 两侧 context；address window 与任意 SRL anchor 不决定边界。`on` 等 stop/function token 被排除为 response slot，content runs 有 1–6 token 上限，不把 lexical proposal 描述为语义 role。
- time range、duration、date、number range 为完整不可重叠 lexical slot；`None`、bool、long/multiline literal 与 response explicit quote 均保留显式 closed edit policy。Data2txt scalar occurrence 带 literal-record parent span/text/path，重复值保留不同 parent context。
- owner ranking成本只使用 masked response/source contexts、path lexical 与 masked vectors；response target quote 或其原 replay hidden state 不进入 cost。same-source-value 只作为竞争计数输出，不参与 rank cost。
- capture 使用右 padding attention mask/position IDs，拒绝超长而不截断，hook 在 `finally` 清理，且检查每层恰好写入一次和向量有限。它是 auxiliary masked encoder，未声称 native trace 或 factual support。
- 实际 model/code/input 冻结仍应由调用方负责；Required 1 要求把该调用方的冻结记录变成 `match_owners()` 可验证的接口，而不是在本模块中虚构模型效果。

## 验证

独立 CPU：`tests/test_surface_graph.py` **13 passed**（8.58s）。未运行 GPU。当前没有 `test_surface_owner.py`，故 capture/provenance 与 hook 合同尚缺 CPU 覆盖。

## Recheck closure — 2026-09-13

此前 **2 项 Required 均已关闭**；当前范围为 **Critical 0 / Required 0**。

- `masked_dense` 现在强制 `capture` manifest 和 `expected_model_identity`。它核验 sealed receipt、精确 masked document、input-ID digest、model identity、layers、representation、当前 capture code hash、float32 三层 shape、有限性和 array bytes hash；裸 vectors、不同模型、改数组或改 document 都不能标为 dense capture。
- hook batch count 已预先绑定，Ruff B008 消失。实际随机 tiny Llama CPU 回归比较 batch padding 与逐条 capture 的 vectors，检查所有 forward hooks 清空，并确认超长 document fail-closed；这些只是工程合同，不是研究机制或效果结果。

独立 CPU：`tests/test_surface_graph.py tests/test_surface_owner.py` **23 passed**（13.36s）；Ruff 通过。新 `surface_verifier.py` 按委托未纳入本次审查。
