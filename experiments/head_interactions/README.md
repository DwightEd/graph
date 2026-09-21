# 多头条件依赖、抑制与补偿审计

这是候选答案和来源角色辅助的机制实验，不是新的无监督检测器。
模型、采集和干预全部复用 `teaching/state_audit`，不再复制旧实验里的 Llama forward 或 hook。
先读 [实验设计](DESIGN.md)。

## 运行已有穿衣／烹饪配对

从仓库根目录运行，依赖研究环境里的 torch、transformers 4.57.x、numpy、tqdm：

```bash
git pull --ff-only origin main
bash experiments/head_interactions/run_all.sh
```

默认读取 `outputs/paired_head_transport_v1` 的 `paired_config.json` 和
`pairs/*/reviewed_case.json`、`*/onset/context.json`，沿用记录中的模型路径。
不需要旧 attention NPZ，不重采样回答；它会重新运行模型以测量干预。
模型路径迁移时显式指定：

```bash
bash experiments/head_interactions/run_all.sh \
  --paired-input /path/to/paired_head_transport_v1 --model /path/to/original/checkpoint
```

`--model` 对已存 token IDs 只用于迁移原模型路径。更换 tokenizer/model 时使用下面的 Example 文本输入重新编码，
不能把旧模型的 token IDs 直接交给新 tokenizer。教学库当前支持 Llama/Mistral/Qwen2，保持原生 GQA/滑窗。

默认 `heads_llama32.json` 有 8 个探索性候选：7 个中间层头，外加用于下游历史路线对照的 L31H14。
这些编号来自旧案例，不是本轮已确认的功能头，也不宣称在不同模型中功能相同。
其他模型用 `--plan` 指定适合它的层/头；本配置不按此次效果挑头。
每组可包含任意多个 head、位置和层，但不同干预单元不能覆盖同一个来源消息。

默认只在 claim 前的最后一个 query 干预。加 `--query-offsets -2 -1 0` 比较更早位置：
它改变干预位置，候选评分的前缀仍然固定；不能把这个实验说成提前检测或 span 持续性统计。
默认加入一个同层、同头数、同来源范围的随机头 panel；不做能量匹配。
两条自然回答各自在自身固定前缀上比较。它们的历史与措辞不同，尤其旧 onion 候选分别是温度和时长，
不是严格的同属性事实对照；这会保存在 context/inventory 中。旧导出没写 task 时显示 unknown。

## 增加真正的语义对照

自动构造两类明确给出事实的控制题，每类有 base、适用条件交换、值交换、同义改写四个条件：

```bash
export PYTHONPATH="$PWD/teaching/state_audit/src${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.head_interactions.controlled \
  --model /path/to/checkpoint --output outputs/head_control_cases.json

OUTPUT=outputs/head_interactions_controlled \
bash experiments/head_interactions/run_all.sh --cases outputs/head_control_cases.json
```

这里的“12 / 18 分钟”和“gold / cloth cap”是教学构造的证据，不是原 RAGTruth 事实或自然幻觉。
候选 0/1 按值固定；`preferred` 指明哪一个被证据支持。保存三个读出：

- `sum_margin`：supported − rival 的完整候选 logp 和之差，主审计量。
- `mean_margin`：每 token 均值之差，检查候选长度影响。
- `candidate_sum_margin`：候选 0 − 候选 1，跨语义交换时始终保留同一方向。

`semantic_differences.csv` 用第三个读出比较交互变化，不能因 preferred 翻转就得到伪“作用反转”。
base 与 applicability_swap 同时进行双向 donor 替换；value_swap 和 paraphrase 提供对照。
只有两个模板，不能把八个条件算作八个独立自然来源，更不能报告自然检测 AUROC。

## 复用 teaching 的已有回答或适配其他数据集

`--cases` 接收 JSON。已有 teaching run 的回答直接引用样本目录，`target` 为待评分的回答 token 索引：

```json
{
  "model": "/path/to/checkpoint",
  "cases": [{
    "id": "claim1", "answer": "../runs/resampled/samples/000000", "target": 12,
    "dataset": "RAGTruth", "task": "QA",
    "candidates": [" 12 minutes.", " 18 minutes."], "preferred": 0,
    "source_quotes": {"evidence": ["An exact reviewed quote."], "value_source": ["Another quote."]},
    "metadata": {"history_status": "reviewed", "side": "unsupported"}
  }]
}
```

`answer` 相对 cases.json；回答原 token IDs 不变，前缀严格截在目标之前。
也可以提供 `example: {id, source_id, prompt, evidence, metadata}` 和 `response_prefix`，
复用 teaching 的 Example、chat template、offset 对齐。或者直接提供 `prefix_ids`、`prompt_length`、
`candidates`、`sources`，适配已经规范化的其他数据集。
`sources` 是绝对 token 位置，`source_quotes` 是必须唯一出现的已审核引文；特殊 token 会被排除。
无法严格对齐的引文会报错，不猜测来源。不清楚来源角色时可将 plan 的 source 设为 all，
研究完整 head_readout；它保留原生整个头的贡献，不能冒称只删了证据。

修改计划例子：

```json
{
  "groups": [
    {"name": "A", "sites": [{"layer": 12, "heads": [0, 2], "source": "evidence", "offsets": [-1, 0]}]},
    {"name": "B", "sites": [{"layer": 20, "heads": [5], "source": "history", "offsets": [0]}]}
  ]
}
```

`source: all` 为整头；其他名字选择该来源的 A·V 消息。删除不重归一化，增强默认 gain=1.5。
恢复指定来源时使用 teaching 的 ReplaceSource，在完整原生 A @ V 中替换来源权重和值；其他来源保留。
不再把分别舍入的 BF16 消息相加。多层恢复逐层重算，候选使用等长、因果的尾部填充。
注意 `write` 是本次原始路由的 A·V 分量，`total_write` 才包含人为替换后的整个头。

## 结果与续跑

默认目录 `outputs/head_interactions_v2`，每个 trial 成功后原子保存 JSON/NPZ；中断重跑同一命令。
配置改变使用新 OUTPUT，不混用旧结果。进度分 panel 与 trial 两级。
完成后自动生成仓库根目录下的 `outputs/head_interactions_v2_review.tar.gz`。
v1 八组的删除后恢复失败诊断见 [FAILURE_REVIEW_20260921.md](FAILURE_REVIEW_20260921.md)。
协议版本升为 2，不读取旧 trial 作为新恢复结果；已有 v1 目录和压缩包保持原样。
控制阈值仍为 0.01 nats。新增失败类型计数及头状态恢复误差，不放宽门槛。

|文件|回答的问题|
|---|---|
|`summary.json` / `REPORT.md`|实际完成多少 panel、多少来源、控制是否失败|
|`inventory.csv` / `controls.csv`|历史是否审核、候选是否可比、空操作／自替换／删除后恢复是否通过|
|`interactions.csv`|F11/F01/F10/F00、条件化作用及交互；负交互不能直接命名为抑制|
|`singles.csv`|删除与增强的完整结果；是否保护了正确候选但仍未取得优势|
|`adaptation.csv`|删除 A 后，下游 B 的自然变化是否缓冲了删除效应|
|`patterns.csv`|预定 0.05 nats 实际效应阈值下的候选模式，不是机制真值|
|`selected_minus_random.csv`|相同案例/位置下，候选头与随机头的效应差|
|`semantic_differences.csv`|固定候选身份时，条件交换/值交换/改写是否改变交互|
|`aggregate.csv`|分 dataset/task/generator/side/variant 的 source 均值与 bootstrap 区间|
|`messages.csv`|每层/头/query 的来源质量、所选消息与实际总写入范数|
|`cases/*/query_*/panel/context.json`|完整前缀、候选、来源位置、干预节点具体是哪些 token|
|`cases/*/query_*/panel/trials/*.json`、`*.npz`|逐候选 token logp、实际操作与完整的所选头消息向量|

每个物理对会有三个读出行；不把这些行或多个 head 当独立样本。
统计按 source 聚合再 bootstrap；未做多重比较校正，两个自然来源不足以作普遍性结论。
空来源/零路由质量保留为零并在 messages/context 中可见；其零效应不是阴性机制证据。

只重建已完成 panel 的报告和压缩包，不加载模型权重：

```bash
bash experiments/head_interactions/run_all.sh --stage report
```

准备输入并检查引用对齐、不运行大模型：

```bash
bash experiments/head_interactions/run_all.sh --stage prepare
```

## 代码阅读顺序

1. `inputs.py`：统一输入及干预计划。
2. teaching 的 `experiments/contrasts.py`、`experiments/messages.py`：读出、读取消息、恢复消息。
3. `protocol.py`：明确列出四条件、剂量、下游恢复及 donor 对照。
4. `trials.py`：逐条件保存与续跑；`report.py`：纯离线报告。
5. `run.py`：按输入、模型、条件、报告的顺序调用。

软件验证与自然实验分开记录。CPU 随机小模型验证数值和工程行为，不能解释为发现了幻觉机制。

2026-09-21 修复后实际验证：teaching 全部测试与本实验共 108 项通过，包含 Llama/Mistral/Qwen2、
GQA、增大 value 幅值后的 BF16 逐层精确恢复，以及不同长度 donor、query 隔离、同头多来源组合。
此前 102 项中仅检查小幅值 BF16 的误差低于 0.01，未覆盖此次真实数据暴露的问题。
已经分析用户上传的 v1 自然数据；修复版尚未在其 8B CUDA checkpoint 上运行。
没有新增自然机制结论或 AUROC。
