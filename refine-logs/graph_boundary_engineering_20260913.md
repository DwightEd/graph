# Graph-boundaries engineering review — 2026-09-13

范围：`next_iteration/graph_boundaries.py` 与其新测试。该模块尚未连接 GPU 或检测运行；以下是纯坐标/范围接口审查，不构成新检测结果。

## 结论

**Critical：0**

**Required：3**

1. **`localized_words()` 没有将传入 text 绑定到 base graph 的原始文本。** 它只用调用方给出的字符坐标在 [graph_boundaries.py](../next_iteration/graph_boundaries.py#L143) 扫词；同长度但已改的 response 仍可用原 role span 写出局部风险。最小复现将 `Alpha shipped 9 crates.` 改为 `Alpha shipped 7 crates.`，原 quantity slot 仍得到 localization。应在 `base_graph()` 保留 `text_sha256`，并在 `localized_words()` 先核验输入 text 的 hash（或每个 base exact substring）后再读任何 bag/local decision。新增回归 `test_localization_rejects_changed_text_at_frozen_role_coordinates` 当前失败。

2. **非断言 quote/punctuation base 在词输出中丢失其状态。** `base_graph()` 正确标记 `kind="nonassertive"`，但 [localized_words()](../next_iteration/graph_boundaries.py#L154) 一律给无 local decision 的词 `unknown_within_base`。`""` 于是被当作未定事实 scope，和“无证据的事实词”无法区分，也不符合设计中 quote/punctuation 应显式为 nonassertive 的合同。保留 `.5` 与完整分母，但输出 `localization_status="nonassertive"`（或等价明确 scope）并禁止其携带事实 bag 判定。新增 `test_quote_only_base_remains_explicitly_nonassertive_at_word_output` 当前失败。

3. **connected region 的 `noncontiguous` 标志只检查 component 的索引集合，未检查跳跃边。** [graph_regions()](../next_iteration/graph_boundaries.py#L207) 中 `0→2` reuse 和 `1→2` reuse 会得到 IDs `[0,1,2]`，因索引无空洞而错误标为 `noncontiguous=false`，尽管 `0→2` 跨过未直接连接的文本。`contiguous_regions` 恰好仍是 `[0]` 与 `[1,2]`，两种输出相互矛盾，消费方可能把 connected region 渲染为连续范围。应由 allowed edge 集决定 connected region 是否包含任一 `right != left+1` 跳跃，或明确输出跳跃 edges；不能只依据排序后的 node IDs。新增 `test_jump_edge_keeps_a_connected_region_noncontiguous_even_when_ids_fill_the_gap` 当前失败。

## 已确认

- 原始 sentence envelope 不受 128-address unit 或 pointer anchor 端点影响；跨 unit 的长句保留完整。`in a` 类右开 anchor 被标为 `truncated_anchor_candidate`，未缩短 base。
- 有资格的词级风险需精确 role/base/span、`scope_verified=true`、限定 basis 和 SHA-256 evidence reference；未达门槛的决策不传播 bag 风险，显式相反的 local risks 回 `.5`。
- correction/quote/new_topic 高置信边会从 direct merge allowed set 移除；非邻接 reuse 边仍保留，不会直接写词级风险。
- 调用方仍必须验证 finite/bridge evidence artifact 本体；该纯映射层只检查证据引用格式，未将其解释为事实或 native 结论。

## 验证

原有 12 项通过；我新增的 3 项精确回归均失败，形成以上 Required 的可执行验收条件。Ruff 对模块和测试通过。未运行 GPU，未读取或修改正在执行的 `source_pointer_prepare` 输出。

## Recheck closure — 2026-09-13

此前 **3 项 Required 均已关闭**；当前范围为 **Critical 0 / Required 0**。

- base graph 现封存 `text_sha256`、长度与全对象 digest；`localized_words()` 在读取 words/decisions 前核验这些值及每个 base substring。改写同长度 response 不能再沿旧角色坐标局部化。
- nonassertive base 保留在完整词分母，但固定 `localized_risk=.5`、`base_bag_risk=null`、`localization_status="nonassertive"`，且不接受局部 decision 传播。它不再与事实 scope 的 unknown 混淆。
- connected region 同时报告 `spatially_contiguous` 和所有 `jump_edges`；`noncontiguous` 现在检测缺少相邻 allowed link，即使 component IDs 恰好填满索引也不能伪装成连续范围。连续 word-risk 仍未由该模块赋值。

独立 CPU：`tests/test_graph_boundaries.py` **15 passed**（6.36s），含本审查新增的三项回归；Ruff 通过。
