# 本轮实际验证

2026-09-22，CPU 软件验证，未运行用户服务器的 8B 或全量自然数据。

```bash
python -m pytest -q tests/test_native_support.py teaching/state_audit/tests/test_native_trace.py
```

结果：32 passed。新检测检查及共享采集钩子的既有回归均通过。

- 实际小型 Llama 原生 logits 与逐 token 缓存回放对齐，覆盖 float32/bfloat16。
- 禁止 `autograd.grad` 后仍能采集；没有参数梯度，异常后钩子清理。
- 当前目标 token 未进入自身预测输入；修改未来 token 不改变此前采集。
- 保存 attention、逐头逐源写入及 value energy；账本可还原最终 logit gap。
- 局部项加历史项严格组成总分；正向历史支持与反对历史分别处理。
- 新 prompt 支持能抵消旧风险；来源模式不匹配时不传播。
- 缩放写入不改变归一分数；FFN 正负不能直接成为错误标签。
- 远处 prompt 区域之间的读取增强可观察，首行的变化明确缺测。
- 断点补采与一次采集结果一致；标签仅在独立评价阶段读取。

另使用 teaching 的 `build_demo` 创建本地随机权重 Llama，通过真实命令
`python main.py support` 完成 6-token 采集、评分、NPZ、CSV、HTML 和 summary 写入。
最大绝对账本闭合误差为 `2.773035817638103e-08`。
这是入口与软件集成检查，不是自然样本成绩或幻觉机制证据。

默认真实档案只有 4 个自然前缀、322 个已观察回答 token。
本轮没有计算它们的新检测分数，也没有把候选文本拼成“真实错误续写”。
