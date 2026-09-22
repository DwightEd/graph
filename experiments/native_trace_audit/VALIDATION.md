# 实施与验证记录

2026-09-22。此次完成软件实现、真实输入核验和 CPU 数值验证；没有新的 8B 自然样本残差结果。

## 已核验的真实输入

从用户上传的 `reanchor_review.tar(1).gz` 原样提取四个 context：

| 案例 | 侧 | prompt tokens | 决策前缀 tokens |
|---|---|---:|---:|
| 14315_headwear_scope | supported | 434 | 469 |
| 14315_headwear_scope | unsupported | 434 | 469 |
| 14375_onion_stage | supported | 535 | 661 |
| 14375_onion_stage | unsupported | 535 | 661 |

逐一核验 token 数与解码片段对齐、角色引文、角色 token 无重叠、角色只位于 prompt 内、
同案例两侧 prompt IDs 完全一致。输入检查结果在
`results/native_trace_input_review_20260922/`，明确 `model_run=false`。

旧包只有 top-8 routes、范数、旧候选读出和干预结果，没有可重建本次完整账本的 FFN/残差向量。
本轮不从缺失数据推断“FFN 覆盖证据”。新的采集命令只补这四个前缀及四个受控读出。

## 软件验证

在 CPU torch 2.14.0+cpu、Transformers 4.57.6 上，完整相关回归 **212 项通过**：

```bash
PYTHONPATH=teaching/state_audit/src python -m pytest -q \
  teaching/state_audit/tests tests/test_native_trace_audit.py tests/test_short_span_capture.py
```

随后报告、配对汇总与受控前缀编译的针对性检查 **28 项通过**，与上述测试有重叠，不能相加。

- 小型随机权重 Llama、Mistral、Qwen2，包含 GQA 与 Mistral sliding-window mask。
- 新采集的 logits 与原生完整前缀输出一致；来源分组覆盖所有 key，逐头写入仍保持身份。
- FP32、BF16 均检查带显式舍入项的账本闭合；不是把偏差归到任意一个模块。
- 更换候选顺序只反转分数/梯度，不改变 attention、残差或 FFN 原始输出。
- 后续读取与 FFN 两类局部导数，都用微小正负方向差分核对。这些仅用于数值测试；
  真实样本采集不运行删除/替换世界。
- 连续查询共享 KV、每个前缀 token 只推进一次；改变未来 token 不影响较早采集。
- 异常退出恢复参数标志、attention backend 与 hooks。
- reanchor 比较固定来源集合，滚动边界本身不构成切换；观察历史不足时不报阴性。
- 实际小模型端到端采集、离线报告、图片、审阅包和续跑已运行；续跑不重做已完成前向。
- 审阅包解压后离线重新生成报告成功；受控前缀保留原始 token IDs，且不虚构实际生成词。

报告图已实际生成并检查可读性。小模型输出属于软件验证，不代表 8B 或自然幻觉机制。

## 仍需服务器执行

当前工作环境没有用户的 Llama-3.1-8B 权重或 GPU，因此尚未采得这四段前缀的新 FFN、
残差、候选写入与局部依赖。用户服务器执行 README 中的一键命令后，`report.json`
会记录真实完成数、计算时间和账本误差，逐查询 NPZ 记录实际显存峰值。
默认共 40 个查询，不触发完整数据集检测、旧贡献目标重采集或新的生成。
