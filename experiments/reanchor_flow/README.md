# Evidence-to-Target Causal Corridor

本目录的主方法是 ETCC：先找 source-unit contribution 和多路由 carrier，再以固定
target contrast 做真实 message 干预。旧 schema-v8 detector 保留为可复现实验基线，不再作为
机制主方法扩展。

方法、结论门槛与旧假设偏差见 [`METHOD.md`](METHOD.md)；输入/输出字段见
[`SCHEMA.md`](SCHEMA.md)；实验推进和停止规则见 [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)。

## 真实 RAGTruth 子集：无需手工 pair

先在三个任务各取一个 source-diverse 样本、每个样本审计一个无标签 target：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
conda activate research

bash experiments/reanchor_flow/run_subset.sh \
  --split test \
  --task all \
  --samples-per-task 1 \
  --targets-per-sample 1 \
  --target-policy uncertain \
  --flow-signal message \
  --carrier-scope response \
  --max-response-tokens 128 \
  --edge-coverage 0.90 \
  --query-chunk 8 \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --source-info /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl \
  --output experiments/reanchor_flow/outputs/native_subset_message
```

命令会自动执行 source unit 对齐、target/runner 冻结、native graph、source Value cut、
root/carrier/corridor 因果验证和紧凑保存。重跑完全相同的命令会验证并跳过已有 target。
无需 `paired_world.npz`。

native target gradient 采用逐层 reverse VJP，每次只保留一层 autograd graph；这减少的是
整网深度图的驻留，不构成“24GB 一定可跑”的承诺。长上下文的单层 eager attention 仍有
\(O(S^2)\) 显存开销，正式 pilot 前应按实际长度做峰值显存检查。

capture 完成后才能打开 hallucination labels：

```bash
python -m experiments.reanchor_flow.run subset-evaluate \
  --split test \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --output experiments/reanchor_flow/outputs/native_subset_message
```

要做 matched cohort 的 attention 对照，保持选择参数不变，只改变 signal 和输出目录：

```bash
bash experiments/reanchor_flow/run_subset.sh \
  --split test --task all --samples-per-task 1 --targets-per-sample 1 \
  --target-policy uncertain --flow-signal attention \
  --carrier-scope response --max-response-tokens 128 \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --source-info /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl \
  --output experiments/reanchor_flow/outputs/native_subset_attention
```

子集模式固定 `a=observed token`、`b=native run 中排除 a 后的 top runner`。
`message` graph 的 transport 是真实 `||W_O(A V)||`；`attention` graph 的 transport 是
raw softmax attention。两种 graph 都另外计算同一个
`phi=<grad F, A V>`，再用真实 Value-message cut/patch/block 做因果确认。因此切换
`--flow-signal` 只改变候选路由，不改变 target 功能量和干预算子。

这个模式识别的是 observed target 对指定 Value-source cut 算子的依赖。cut 不直接
mask Q/K，但删除 source self-message 后，后续 state 与 Q/K 会在 cut world 中自然演化，
因此不声称冻结了 native selector。RAGTruth 又没有 corrected target 或精确
supporting span，所以结果不能表述成“正确事实被采纳”。严格事实结论仍使用下面的受控 pair 模式。

## 受控 clean/corrupt pair

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

受控 pair 中 `--flow-signal` 决定选边和 throughput 使用的数据：

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
units/worlds/native world → attribution → flow → throughput
                                              ↓
                                      corridor/audit
                                              ↓
                              subset manifest/report
```

没有第二套 message intervention：精确 edge delete、pre-`W_O` patch 和 residual patch 都复用
`experiments/common/llama_message_intervention.py`。

## 测试

```bash
pytest -q \
  experiments/common/tests/test_llama_message_intervention.py \
  experiments/reanchor_flow/tests
```

测试覆盖 native/manual forward 一致性、GQA、绝对坐标、双 transport 分离、独立 functional
score、throughput 守恒、固定 runner、label firewall、断点恢复，以及 native/root-cut 两个世界的
delete-and-restore 正控制。

## 旧基线

原 `analyze/evaluate/detect/all` 命令仍可复现 schema-v8 frozen detector。它的最新 held-out
token AUROC 为约 `0.58–0.62`，onset AUPRC 约 `0.007–0.011`；这些结果是停止继续堆检测
特征、转向机制 corridor 的依据，不是 ETCC 结果。
