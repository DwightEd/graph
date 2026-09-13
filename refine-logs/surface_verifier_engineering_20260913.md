# Surface verifier engineering review — 2026-09-13

范围：`next_iteration/surface_verifier.py` 与其 CPU tests。未运行 GPU；finite outputs 是冻结模型预测，不是事实标签或 native 结果。

## 结论

**Critical：0**

**Required：1**

1. **独立 `assess_target()` 输出缺少消费期的 receipt revalidation。** 生成时它确实检查 cache receipt（[surface_verifier.py](../next_iteration/surface_verifier.py#L30)），但返回的 sealed assessment 没有等价于 `verification_status()` 的 validator。后续 caller 可以读取 `scores` 来决定“correct/uncertain target 不进入其余 4 checks”，却不会重读 reader cache、复算 exact target request 或比较 frozen reader identity。cache 在 assessment 写入后变化、或 assessment 被移植到另一 response/source/slot 时，self-seal 仍可完整而基础 reader evidence 已失效。新增 `validate_target_assessment(response_graph, sources, slot_id, assessment, reader_identity)`，验证 graph hashes、slot ID、target payload/request digest、reader identity、receipt 和 prediction/scores；只允许通过该函数的结果影响是否跳过后续 checks。为缓存字节篡改、错误 slot/source graph、错误 reader identity 加 CPU 回归。

## 已确认

- strict native handoff 的 5 个固定 requests 正确区分：original target proposition SCNU、edited **full base** SCNU、selected occurrence SCIU、preservation PFU、edit-kind VKU。所有门槛均达到 .8 前不会产生 contrast；它们的 receipts 在 `verification_status()` 中逐项重验。
- `finalize()` 重建 exact surface edit，核对 response/source 与 observer row，再从 base 的实际末端构造 B/A；未调用旧 `text_units` 向后扩展。原 B 分支必须是 frozen observer replay 的精确 prefix，`from_sequences` 与 `align_row`/`slot_masks` 绑定完整 continuation 和 target/source keys。
- source parent、draft、prediction 或 reader identity 在 strict chain 中被篡改均被拒绝。输出坚持 `certificate_count=0`、`not_ground_truth=true`，并明确 native position/origin/routing 尚未测量。
- 新增 standalone target helper 只用于节省对已支持/uncertain target 的其余 finite requests；它不能降低 5-check strict native gate 的门槛。

## 验证

独立 CPU：`tests/test_surface_verifier.py` **11 passed**（13.49s），含本审查新增的本地 Llama tokenizer 完整 finalize 回归。该回归验证原始完整 base B branch token IDs 与 frozen replay 一致，且两分支都有 slot mask；只加载 tokenizer，未加载模型权重。Ruff 通过。

## Recheck closure — 2026-09-13

此前 **1 项 Required 已关闭**；当前范围为 **Critical 0 / Required 0**。

- `validate_target_assessment()` 现在在任何 target-only 结果被消费前重新核验 sealed assessment、response/source graph hashes、slot、frozen reader identity、canonical target request、published cache receipt 与由 immutable prediction 重新推导的 scores。`surface_runner` 调用点先执行该验证；改 reader/cache/scores 或图/slot 请求均不能用于跳过其余 checks。
- `_target_slot` 在 target request 和 strict five-check 的 target payload 中统一为 ID/base/span/quote/type。selected-occurrence request 去除其 masked-context 辅助检索文本，但保留 full source、raw/display surface、parent text、record parent 和 field path；因此既避免不必要的重复/检索 payload，也保留 selected-owner 与条件核验所需的原文出处。
- five-check native gate 未放宽：target-only helper 仅允许对正确或不确定 target 早停，不能替代 strict candidate 的原始-target、edited-base、selected-occurrence、preservation、edit-kind 全部 receipt-bound checks。

独立 CPU：`tests/test_surface_verifier.py` **12 passed**（13.38s），含真实本地 Llama tokenizer 的完整 B/A token/slot-mask 回归；Ruff 通过。未启动 GPU。
