# Attention Graph Lab

本仓库把 LLM attention 保存成可校验的 canonical archive，并在同一份数据上进行图构建、结构特征分析和幻觉 onset 验证。

## 模块职责

当前代码按职责分层，不再使用 `rich_*` / `transition_*` 叠加式补丁模块：

```text
extract.py / archive.py
        │  模型或旧缓存 -> canonical archive
        ▼
research_dataset.py
        │  只负责 canonical 数据访问、CSR 解码、token relation
        ▼
graph_features.py
        │  只负责 graph -> token feature vector
        ▼
sample_analysis.py
           单条样本、正确对照、run-centric 统计与可视化
```

其他实验各自独立：

- `graph_tsne.py`：dataset-level graph t-SNE，一个点是一条 response。
- `node_tsne.py`：dataset-level node t-SNE，一个点是一个 response token。
- `onset_experiment.py`：跨样本配对 onset 统计验证。
- `onset_validation.py`：onset experiment 使用的图随机化原语。

单样本实验唯一 notebook：

```text
notebooks/sample_analysis.ipynb
```

## Canonical attention

每个 split：

```text
<split>/
├── manifest.json
├── index.jsonl
├── labels.jsonl
├── attention/<sample_id>.npz
├── hidden/<sample_id>.npz       # optional
└── token_stats/<sample_id>.npz  # optional
```

每个 attention NPZ 固定六个字段：

| field | shape | meaning |
| --- | --- | --- |
| `token_ids` | `[N]` | prompt + response token IDs |
| `response_idx` | `[]` | first response-token index |
| `attention_diagonal` | `[L,H,N]` | attention diagonal |
| `response_row_ptr` | `[L*H*R+1]` | sparse response-query CSR pointer |
| `response_column_indices` | `[M]` | earlier source positions |
| `response_values` | `[M]` | retained attention weights |

CSR 只保存 `attention > attention_floor` 的 response-query 非对角值，并且 source 必须严格早于 target。未保存的边只能解释为 `<= attention_floor`，不能解释为精确 0。

## 数据抽取

RAGTruth 入口：

```bash
python main.py extract \
  --model-path /models/Meta-Llama-3.1-8B-Instruct \
  --dataset-path /data/RAGTruth/dataset \
  --output-dir /data/model_traces/run/test \
  --split test \
  --generator-model llama-2-7b-chat \
  --floor 0.01 \
  --device cuda
```

`observer_model` 是执行 teacher-forcing、提供内部 attention 的模型；`generator_model` 是原始 response 的生成模型。二者会分别写入 metadata，不应混为一谈。

已有正式 attention cache 可转换为同一 canonical 格式：

```bash
python main.py archive-attention \
  --formal-root /path/to/formal_cache \
  --output-root /data/RAGTruth/model_traces/llama31_8b

python main.py verify-attention \
  --archive-root /data/RAGTruth/model_traces/llama31_8b
```

补 research metadata：

```bash
python main.py enrich-index \
  --canonical-root /data/RAGTruth/model_traces/llama31_8b \
  --dataset-path /data/RAGTruth/dataset
```

## 图构建

图缓存始终是 topology-only；标签、hidden state 和任何学习 embedding 不写入图 PT。

```bash
python main.py build \
  --cache-dir /data/RAGTruth/model_traces/llama31_8b/train \
  --output-dir /data/RAGTruth/graphs/llama31_8b/relation_topk_channels/train \
  --kind relation_topk_channels \
  --k-prompt 8 \
  --k-history 8 \
  --device cuda
```

支持 `original`、`relation_topk`、`relation_topk_channels` 和 `hypergraph`。

## Graph feature pipeline

`research_dataset.py` 将 canonical CSR 解码为：

```text
(layer, head, source, target, weight)
        ↓ aggregate same source->target
(source, target, relation_weight, channel_count, edge_type)
```

所有手工 feature 的正式实现统一在 `graph_features.py`。

`response_graph_features(sample)` 为每个 response token 提取 32D causal incoming-graph descriptor，包括：

- prompt grounding；
- response-history dependence；
- sparsity/density；
- edge-weight concentration；
- history locality；
- early/middle/late layer routing。

`static_feature_blocks()` 将累计 locality 改为互斥的 `1 / 2-4 / 5-8 / 9-16 / >16` 距离区间，最终形成六个语义 block，共 33 维。

`dynamic_state()` 再生成 19 个 token-to-token transition features，包括 block-wise delta、rolling deviation、source JS divergence、prompt/history source JS、neighbor turnover 和 layer-routing shift。

## 单条样本可视化

拉取代码后直接打开：

```bash
jupyter lab notebooks/sample_analysis.ipynb
```

修改 notebook 顶部：

```python
ERROR_SAMPLE_ID = "10071"
GENERATOR_MODEL = None
```

然后 **Restart Kernel -> Run All**。

核心调用只有一个：

```python
from sample_analysis import SampleAnalysis

analysis = SampleAnalysis(DATA_ROOT, output_root=OUTPUT_ROOT)
result = analysis.visualize(sample_id)
```

输出到：

```text
outputs/sample_analysis/<sample_id>/
```

包含每个 hallucination run 的局部 PCA/t-SNE、transition curve、block deviation、same-generator correct-control null，以及 selected response 与正确 controls 的 joint projection。

单样本代码职责：

- `graph_features.py`：只算向量，不读 label，不选 control，不画图。
- `sample_analysis.py`：只组织单样本实验、正确 control、统计和图。
- notebook：只配置参数并调用 API，不实现算法。

## Dataset-level visualization

Graph-level：

```bash
jupyter lab notebooks/graph_tsne.ipynb
```

Node-level：

```bash
jupyter lab notebooks/node_tsne.ipynb
```

它们是不同实验，不与单样本 notebook 混用。

## Paired onset validation

跨样本 confirmatory analysis：

```bash
python scripts/validate_onsets.py \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --output-dir outputs/onset_validation/test \
  --device cuda \
  --effect-width 3 \
  --bootstraps 10000 \
  --permutations 10000 \
  --rewires 100 \
  --seed 0
```

设计与限制见 [`docs/onset_validation.md`](docs/onset_validation.md)。

## Tests

核心分析测试：

```bash
python -m unittest \
  tests.test_graph_features \
  tests.test_sample_analysis \
  tests.test_node_tsne \
  tests.test_graph_tsne
```

完整测试：

```bash
python -m unittest discover -s tests -v
```
