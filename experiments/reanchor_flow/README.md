# Evidence-to-Target Causal Corridor

本目录的主方法是 ETCC：先找 source-unit contribution 和多路由 carrier，再以固定
target contrast 做真实 message 干预。旧 schema-v8 detector 保留为可复现实验基线，不再作为
机制主方法扩展。

方法、结论门槛与旧假设偏差见 [`METHOD.md`](METHOD.md)；输入/输出字段见
[`SCHEMA.md`](SCHEMA.md)；实验推进和停止规则见 [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)。

## 运行

真实 message backend：

```bash
bash experiments/reanchor_flow/run_corridor.sh \
  --pair data/etcc/example_pair.npz \
  --flow-signal message \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --query-chunk 8
```

纯 attention routing 对照：

```bash
bash experiments/reanchor_flow/run_corridor.sh \
  --pair data/etcc/example_pair.npz \
  --flow-signal attention \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --query-chunk 8
```

`--flow-signal` 决定选边和 throughput 使用的数据：

- `attention`：`edge_score == clean softmax attention`；不计算 target gradient；
- `message`：`edge_score` 是 clean-corrupt 真实 `AV` message 对固定 margin 的有符号作用。

两种模式的因果确认都使用真实 pre-`W_O` message code，不用 attention mask 的变化冒充
message patch。加 `--materialize-messages` 才会额外保存 clean/corrupt/delta 的完整 hidden-size
post-`W_O` vector；默认只保存数值重建和原位干预所需的 float32 head code，以控制文件体积。

默认 `--carrier-scope all`，会包含 prompt 与 response carrier。`response` 是明确的便宜消融，
会漏掉 prompt 内中继。`--root-screen-limit 0` 对所有 candidate units 做精确双向 patch。

## 构造 pair

输入不是 hallucination label 文件，而是运行前冻结的受控 pair。最小 Python API：

```python
from experiments.reanchor_flow.worlds import (
    PairedWorld,
    SourceUnits,
    TargetContrast,
    save_world,
)

world = PairedWorld(
    sample_id="sample-1",
    tokenizer_id="Meta-Llama-3.1-8B-Instruct",
    corruption="same-length supporting-fact replacement",
    clean_token_ids=clean_ids,
    corrupt_token_ids=corrupt_ids,
    response_start=response_start,
    units=SourceUnits(token_unit_id, unit_names, unit_kinds),
    candidate_unit_id=(3, 4),
    targets=(
        TargetContrast(
            query_position=q,
            positive_token_id=correct_id,
            negative_token_id=competing_id,
            origin="controlled clean-vs-corrupt fact",
        ),
    ),
).check()
save_world("data/etcc/sample-1.npz", world)
```

`build_source_units` 可从 RAGTruth historical prompt 对齐 passage、sentence 或 Data2txt field，
但准确 supporting span 和 matched corruption 仍需受控数据或独立标注。

当 pair 同时列出多个 candidate units 时，它只用于 root screening；程序选中 root 后会把
其他候选恢复为 clean token，并在 isolated world 中重新捕获最终 corridor。

## 模块边界

```text
worlds / units
      ↓
attribution → flow → throughput
                         ↓
                 corridor / audit
                         ↓
                       run
```

没有第二套 message intervention：精确 edge delete、pre-`W_O` patch 和 residual patch 都复用
`experiments/common/llama_message_intervention.py`。

## 测试

```bash
pytest -q \
  experiments/common/tests/test_llama_message_intervention.py \
  experiments/reanchor_flow/tests
```

测试覆盖 native/manual forward 一致性、GQA、绝对坐标、双后端分离、selector/content 恒等式、
真实 message 重建、throughput 守恒、pair label firewall，以及 delete-and-restore 正控制。

## 旧基线

原 `analyze/evaluate/detect/all` 命令仍可复现 schema-v8 frozen detector。它的最新 held-out
token AUROC 为约 `0.58–0.62`，onset AUPRC 约 `0.007–0.011`；这些结果是停止继续堆检测
特征、转向机制 corridor 的依据，不是 ETCC 结果。
