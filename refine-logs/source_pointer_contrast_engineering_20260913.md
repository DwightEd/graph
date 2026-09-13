# Source-pointer contrast engineering review — 2026-09-13

范围：`next_iteration/source_pointer_contrast.py`、`next_iteration/source_pointer_prepare.py` 和 `tests/test_source_pointer_contrast.py`。未运行 GPU、未修改冻结路由或当前运行的 A 批次。

## 结论

**Critical：0**

**Required：2**

1. **四项 finite C 结果只绑定请求内容，未绑定产生结果的冻结 reader。** `verify_contrast()` 保存 instruction/payload/labels 的 request digest 和裸 prediction（[source_pointer_contrast.py](../next_iteration/source_pointer_contrast.py#L192)），`verification_status()` 也只复算该 digest（L206）。记录没有 reader 模型、tokenizer、generation/code identity，也没有 FrozenReader immutable cache/request-record digest。因此来自不同 reader、不同生成配置或缓存记录的四个 prediction 可以混合后通过最终验证。应把冻结 reader identity 和每个 cache/request-response record 的 hash 纳入 sealed verification，并在最终化/未来 runner 时同已冻结运行 settings 核验；添加混合 reader identity 或 cache-record hash 的拒绝回归。

2. **新 preparation 的代码身份在计算完成后才描述，未冻结实际执行代码。** `prepare_run()` 直到完成所有 draft 后才在 manifest 中 hash `source_pointer_prepare.py` 与 `source_pointer_contrast.py`（[source_pointer_prepare.py](../next_iteration/source_pointer_prepare.py#L50)）。这两个文件在执行期间改变时，早期 drafts 可由旧代码构造、manifest 却标为新代码；也没有保存可重放快照。先在执行前冻结/复制或至少记录完整依赖代码清单，在写 drafts 前后比对其哈希，并在 manifest 绑定验证过的快照。新 bridge 输出将供后续 C/native 使用，不能只继承父 run 的旧 snapshot（该 snapshot 不含 `next_iteration/`）。

## 已确认

- `prepare_contrast()` 重新编译 source/response catalog；catalog 会反过来重编译并验证 inventory 和 pointer/literal graph。它比较 B catalog、所选 target/source/membership、response claim/event/role 坐标及完整单槽字符编辑。最终化再次重建 draft、核对完整 B/A 事件 token 序列、共同前缀和 source/target keys。
- `None`、bool、subject、condition、quoted slot、空/不可打印 surface 和同 surface 的零对比均 fail-closed。数值使用原始 literal text；没有单位转换或从字段名推导单位。重复值将固定 source membership 交给 node check，未验证的多 occurrence 会停在 `ambiguous_same_value_origin`。
- 新 `source_pointer_prepare.py` 在处理前要求**所有** frozen input 行都有 A/B artifact，调用 `verify_executed_code()` 与 `read_artifact()`；后者递归校验 A→B、row/settings/data hashes 和 B feature 文件。它只遍历已冻结 B 的每个 `source_term`，不做新搜索、不读标签、不调用 reader/native，并把 candidate 与每种 reject 均写入 attempts 分母。当前 soft-graph run 仍为 A 阶段 20/36，故未对半批调用它。
- `certificate_count=0`、`not_ground_truth=true` 与 `mechanism_scope=no_binding_pointer_identification; separate_native_tests_required` 正确保留：即使将来 raw-origin 通过，也只是输入来源/介导路径，不能称作抽象 pointer 证明。

## 验证

- 独立 CPU：`tests/test_source_pointer_contrast.py` **16 passed**（6.11s）；Ruff 通过。
- 仅 CPU 加载本地 `Meta-Llama-3.1-8B-Instruct` tokenizer 成功（23 tokens / 23 offsets）；未加载模型权重或启动 GPU。

## Final recheck — 2026-09-13

此前 **2 项 Required 均已关闭**；当前范围为 **Critical 0 / Required 0**。

- `ReceiptReader` 只在 `FrozenReader` 已经发布 immutable cache record 后记录其绝对路径和文件 SHA。verification 在创建及最终化时都重新读取该文件，验证 filename 为完整 request digest，并将 frozen reader identity、instruction、payload、labels、`max_new_tokens=1` 和 retained prediction 对齐。因此混用 reader、篡改 cache bytes/hash 或替换 request/prediction 会在 native 前失败。
- preparation 在生成任何 draft 前创建四个 `next_iteration` 文件的字节快照；在写 drafts 前后再核验 live 和 snapshot 字节，并最后发布 manifest。父 A/B snapshot 也在开始及 drafts 前复验；A/B artifact 与 B feature-file 仍由 `read_artifact()` 递归验证。中断或代码漂移没有完成 manifest，不能作为后续输入。
- 我新增真实本地 Llama tokenizer 的完整 finalize 回归：它由实际 offset/token IDs 构造 source mask，通过 draft、receipt verification 和 `build_contrast`，并验证完整 original branch token sequence 精确等于 frozen replay IDs、两分支都有 slot token。只加载 tokenizer，未加载模型权重。

独立 CPU：`tests/test_source_pointer_contrast.py` **22 passed**（13.08s）；Ruff 通过。
