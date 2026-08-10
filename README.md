# Attention Graph Lab

这个仓库负责一条清晰的数据链：**模型特征抽取 → 最小化存储 → 不同构图 → 下游分析/训练**。

## 1. 目录与数据层次

一个 canonical split：

```text
train/
├── manifest.json
├── index.jsonl
├── labels.jsonl              # 可选；构图不读取
├── attention/
│   └── <sample_id>.npz
├── hidden/                   # 可选
│   └── <sample_id>.npz
└── token_stats/              # 可选
    └── <sample_id>.npz
```

`index.jsonl` 只负责 `sample_id / source_id / attention path`。不同模态通过相同 `sample_id` 和 `token_ids` 对齐。

## 2. Attention：唯一 6 字段接口

每个 attention NPZ **只保存 6 个数组**：

| 字段 | shape | dtype | 含义 |
| --- | --- | --- | --- |
| `token_ids` | `[N]` | `int32` | prompt+response token id |
| `response_idx` | `[]` | `int32` | 第一个 response token 的位置 |
| `attention_diagonal` | `[L,H,N]` | `float16` | 每层每头的 attention 对角线 |
| `response_row_ptr` | `[L*H*R+1]` | `int32` | response-query attention 的 CSR 行指针 |
| `response_column_indices` | `[M]` | `int32` | source token 位置；source 可来自 prompt 或 response history |
| `response_values` | `[M]` | `float16` | 对应 attention 权重 |

其中 `R=N-response_idx`，CSR 行号：

```text
row = (layer * H + head) * R + (target - response_idx)
```

因此 `layer / head / target / source / weight` 都可从 CSR 解码，不重复保存。

`attention_floor` 不放在每个样本，而放在 `manifest.json`。当前旧正式 RAGTruth cache 使用 `floor=0.01`：只有 response-query 的非对角 attention `> floor` 才进入 CSR；没出现的值只能解释为 `<= floor`，不能当成精确 0。`attention_diagonal` 完整保留。

## 3. Hidden states 与 logits 统计

Hidden state 单独保存，不塞进 attention 文件：

```text
hidden/<sample_id>.npz
  token_ids         [N]      int32
  hidden_layer_ids  [K]      int16
  hidden_states     [K,N,D]  float16
```

Logits 不长期保存完整 `[N,V]` Tensor。抽取后立即压成两个 token-level 统计：

```text
token_stats/<sample_id>.npz
  token_ids       [N]  int32
  token_log_prob  [N]  float32
  entropy         [N]  float32
```

其中：

```text
token_log_prob[t] = log p(x_t | x_<t)
entropy[t]        = H(p(. | x_<t))
```

位置 0 没有前一个预测位置，两个值均置 0；所有 response token 都有效，因为 `response_idx > 0`。

## 4. 已有数据迁移

正式 RAGTruth attention cache：

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

转换成 canonical 结构：

```bash
PYTHONPATH=. python main.py archive-attention \
  --formal-root /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --output-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b
```

旧 HaluEval/BoolQ `token_trace_v2` 或其他包含 `hidden_states / token_log_prob / entropy / logits` 的 PT，可一次性提取需要字段：

```bash
PYTHONPATH=. python main.py archive-features \
  --trace-dir /path/to/extraction/traces \
  --output-dir /path/to/canonical_split
```

如果旧 trace 只有完整 logits，转换器会计算 `token_log_prob` 与 `entropy` 后只保存这两个紧凑统计。

旧属性图不再作为长期原始数据保存；它从 canonical attention 重新构建即可。

## 5. 重新抽取模型特征

一次 forward 同时抽 attention、token stats，并可选抽指定 hidden layers：

```bash
PYTHONPATH=. python main.py extract \
  --model-path /share/home/.../Meta-Llama-3.1-8B-Instruct \
  --dataset-path /share/home/.../RAGTruth/dataset \
  --output-dir /share/home/.../model_traces/new_run/train \
  --split train \
  --floor .01 \
  --hidden-layers 7,15,23,31 \
  --device cuda
```

不指定 `--hidden-layers` 时只保存 attention + token stats，避免无意间产生大体积 hidden-state 数据。

## 6. 构图

```bash
PYTHONPATH=. python main.py build \
  --cache-dir /share/home/.../model_traces/new_run/train \
  --output-dir /share/home/.../graphs/original/train \
  --kind original \
  --tau .05 \
  --node-features attention \
  --device cuda
```

当前图视图：

- `original`：复现旧 threshold-union topology；
- `relation_topk`：每个 response target 分别保留 prompt/history top-k；
- `relation_topk_channels`：top-k 边同时保留 layer/head channel 值；
- `hypergraph`：按 `(layer, head, response target, PR/RR)` 构造超边。

节点特征可选：

```text
none
attention
hidden
stats
attention+hidden
attention+stats
hidden+stats
all
```

不同节点特征只改变 `x`，不改变同一种构图方法的 topology。

## 7. 原属性图的存储优化

旧属性图保存 `edge_attr [E,L*H]` dense Tensor，是主要磁盘开销。新 `original` 图保持完全相同的 threshold-union 边语义，但每条边的 layer/head 属性改为稀疏表示：

```text
edge_index    [2,E]
edge_type     [E]       0=Prompt→Response, 1=Response→Response
edge_ptr      [E+1]
edge_channel  [T]
edge_value    [T]
```

需要运行旧 CHARM 风格代码时，`graphs.dense_edge_attr()` 可临时恢复 `[E,L*H]`；不再把这个巨大 dense Tensor长期写盘。

## 8. 两个阈值

- `floor`：**特征存储阈值**。模型原始 dense attention 中 response-query 非对角项只有 `> floor` 才保存；这是有损压缩。
- `tau`：**构图阈值**。在已经保存的 attention 上决定哪些 token pair 成为边/超边。

必须有 `tau >= floor` 才能精确重建 threshold graph。
