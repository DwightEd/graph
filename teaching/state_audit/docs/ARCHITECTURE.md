# State Audit 0.2：先确定职责与接口

本次先写本文件与接口骨架，再实现函数。最终交付不保留空实现。

## 边界

- dataset/schema.py：Example；统一 JSONL 与 RAGTruth 适配器只负责数据。
- model/adapter.py：ModelAdapter；模型结构、表征位置、原生 forward 与读出。
- model/sites.py：公开表征名、张量轴；模型适配不得散落到研究代码。
- state.py：LayerState、ModelState；状态容器，逐层懒加载，不预设研究特征。
- capture.py：CaptureSpec、capture_sample/run；观察器、落盘、释放钩子。
- operations/：Target 与 Delete/Replace/Inject/Steer，以及原生收缩内的 ReplaceSource；纯张量变换。
- intervention.py：把操作临时装到模型；作用范围由上下文管理器管理。
- generation.py：GenerationOptions；同题 × seed 重采样，原回答回放。
- pairing.py：按明确审阅结果配对；未知标签不充当正确样本。
- analysis/：纯数组分析和已有 reanchor/角色审计，无模型写操作。
- experiments/：固定前缀读出、任意操作组合、两组条件交互示例。
- storage.py：JSON/NPZ，配置与完成标记，禁止覆盖不同实验。
- cli.py/pipeline.py：参数与流程；不嵌入张量数学。

## 核心接口

0.3 在这些边界内扩展：capture_targets 提供定点观察；experiments/contrasts 提供固定候选读出；
experiments/messages 用已有操作组合出源消息恢复；analysis/interactions 只计算数值分解。
仓库中的 head_interactions 实验复用这些 API，教学包仍不依赖外部研究目录。
详见 [INTERACTIONS.md](INTERACTIONS.md)。

```python
examples = load_examples(path)
model, tokenizer = load_model(checkpoint, revision, device, dtype)
plan = CaptureSpec(layers=(0, 1), representations=("attention", "residual_after"))
capture_sample(model, answer, output, plan)
state = ModelState.open(output)
layer = state.layer(0)

selection = Target("residual_after", layers=(0,), positions=(12, 13))
operation = Inject(selection, vector, scale=0.5)
with intervene(model, [operation]):
    hidden = model.forward(token_ids)
```

位置一律是输入绝对 token 位置；预测回答 token t 的位置为 P+t−1。
head 在 Q/attention/head_readout 中是 query head，在 K/V 中是原生 KV head。
操作列表无两头限制；同一位置按列表顺序应用，跨位置遵守模型执行顺序。
Delete 对 attention 是置零边质量，不自动重新 softmax；需要重新分配剩余质量时，应另写明确的归一化操作，不能偷偷改变消融含义。
attention 替换/注入必须非负、不能打开因果/滑窗 mask；此处保留边界检查。
Steer 是沿单位方向加指定长度，不暗含分类器或训练。

## 实施次序

1. 建立此设计和公开接口骨架。
2. 迁移数据适配与离线审计，填充表征选择和纯操作。
3. 实现模型适配、采集、模型内干预与复用状态容器。
4. 实现多 seed 生成、审阅配对、CLI 与可运行示例。
5. 验证原生对齐、三种模型、任意多头、无操作等价、因果 mask、清理、续跑。

这是 API/目录重构；旧缓存留在原位，v1 审计通过独立读取路径保留。新的 capture
配置和状态格式使用版本 2。没有自动迁移后重跑旧实验，也不宣称产生自然数据结果。

## 本轮完成情况

以上模块已实现，交付中没有空函数/占位类。支持 14 类表征、四类操作、多 seed 采样、
已审阅回答配对、全输入/回答位置两种采集范围。模型支持 Llama/Mistral/Qwen2。

验证：73 项 CPU 小模型测试通过；覆盖三种原生家族、GQA/滑窗/无未来边、
三头来源边删除与 A·V 消息相减等价、Q/K/V/MLP/全局状态操作、
同状态替换、钩子恢复、重采样身份/续跑/标签隔离、全位置到回答位置对齐。
可编辑安装及 demo/check/pair/Python 干预示例/JSON 干预入口已实际运行。
未运行用户服务器上的大模型与自然数据，不提供新的机制或检测成绩。

0.3 增量验证：包含新交互接口和仓库实验工作流在内共 102 项测试通过。

0.3.1 修复 BF16 来源恢复：改为完整原生 A @ V 内的来源替换，新增压力与隔离测试，共 108 项通过。
原始来源观测与替换后的头总状态分别保留；详见 INTERACTIONS.md。
定点采集、GQA 来源消息、BF16 恢复和独立 compare_messages 示例已运行；原 0.2 API 回归通过。
