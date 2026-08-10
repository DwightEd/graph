# Attention Graph Research

这个仓库只做四件事：**特征抽取、构图、统计、可视化**。数据对象保持最小，不在每个 `.pt` 里保存 schema、commit、绝对路径、manifest 或描述性统计。

## 现有数据在哪里

当前正式 attention cache：

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

目录下有 `train/` 和 `test/`。这是 **response-query sparse attention CSR**，不是完整 `[L,H,N,N]` dense attention。

旧属性图：

```text
/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/ragtruth_original_attribute_graphs/fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05
```

旧图包含大量复现 metadata 和 `[E,L*H]` dense `edge_attr`。它可以用于兼容实验，但不再作为新研究的数据接口；新图应从 raw attention cache 重新构造。

## 最小 feature 接口

新抽取只保存：

```python
{
    "sample_id": str,
    "source_id": str,
    "response_idx": int,
    "token_ids": Tensor[N],
    "attention_diagonal": Tensor[L,H,N],
    "row_ptr": Tensor[L*H*R + 1],
    "source_index": Tensor[M],
    "attention_weight": Tensor[M],
    "attention_floor": float,

    # 只有请求 hidden layer 时才存在
    "hidden_layers": Tensor[K],
    "hidden_states": Tensor[K,N,D],
}
```

其中 `R=N-response_idx`。CSR 行号：

```text
row = (layer * H + head) * R + (target - response_idx)
```

因此 `data.attention_entries(sample)` 可以直接展开为：

```text
layer, head, source, target, weight
```

这些字段只表示 attention 边。节点由 `token_ids` 的位置得到，Prompt/Response 由 `response_idx` 得到，节点属性由 `attention_diagonal` 或可选 `hidden_states` 得到。

### 标签单独保存

抽取时标签写到 `labels/<split>/<sample_id>.pt`：

```python
{
    "sample_id": str,
    "task": str,
    "y_token": Tensor[N],
    "response_label": int,
}
```

图构建完全不读取标签。

## 最小 token graph 接口

所有普通token图只保存共同必需字段：

```python
{
    "sample_id": str,
    "source_id": str,
    "response_idx": int,
    "token_ids": Tensor[N],
    "x": Tensor[N,F],
    "edge_index": Tensor[2,E],
    "edge_type": Tensor[E],       # 0=Prompt->Response, 1=Response->Response
    ...                            # 构图方法真正需要的边属性
}
```

没有 `node_role`，因为它可由 `response_idx` 直接计算；没有 `split`，因为目录已经区分；没有 `tau`/路径/commit/schema，因为它们是实验配置而不是图本身。

## 已实现的构图

### `original`

复现原属性图：任一 layer/head 上 `attention > tau` 就建立 token-pair 边，`edge_attr[E,L*H]` 保留各通道超过阈值的值。它主要用于与旧CHARM实现对齐。

### `multiplex`

每个 `(layer, head, source, target)` attention 都是一条独立有向边。保存：

```text
edge_index, edge_weight, edge_channel, edge_type
```

这是研究 layer/head-specific topology 最直接的表示。默认不再二次阈值，直接使用 cache 已经保留的所有 sparse attention。

### `support`

对每个 `(layer, head, response target)` 按 attention 从大到小选择，直到累计质量达到 `mass`（默认0.8）。用于研究 support collapse 和弱边是否有用。如果 cache floor 截断了尾部，小于 floor 的部分无法恢复。

### `relation_topk`

对每个 `(layer, head, response target)` 分别保留前 `k_prompt` 个 Prompt source 和前 `k_history` 个 Response source。用于显式研究 Prompt→Response 与 Response→Response 模式。

### `hypergraph`

每个 `(layer/head, response target, PR/RR relation)` 构成一条超边，source token 为成员，并加入target自身。用于研究共同attention support，而不是pairwise边。

## 节点属性可以独立控制

所有构图函数都有：

```text
node_feature = diagonal | hidden | none
```

这样可以直接做：

```text
固定拓扑 + 不同节点属性
固定节点属性 + 不同拓扑
```

从而回答究竟是连接关系、attention权重还是hidden state更重要。

## 使用现有 cache 构图

`data.load_feature()` 兼容当前 `fresh_attention...` 的旧字段，所以**不用重新提取已有attention**。

```bash
CACHE=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
OUT=/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/graph_research

python main.py build --features "$CACHE/train" --output "$OUT/original/train" --kind original --tau .05
python main.py build --features "$CACHE/train" --output "$OUT/multiplex/train" --kind multiplex
python main.py build --features "$CACHE/train" --output "$OUT/support80/train" --kind support --mass .8
python main.py build --features "$CACHE/train" --output "$OUT/relation_topk/train" --kind relation_topk --k-prompt 8 --k-history 8
python main.py build --features "$CACHE/train" --output "$OUT/hypergraph/train" --kind hypergraph --tau .05
```

## 重新抽取 attention / hidden states

```bash
python main.py extract \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --dataset /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --output /share/home/tm902089733300000/a903202310/lys/data/feature_extraction/graph_features \
  --split train \
  --generator-model llama-2-7b-chat \
  --floor .01 \
  --hidden-layers 7,15,23,31
```

不需要hidden state时省略 `--hidden-layers`，文件会更小。

## 统计和可视化

```python
from data import load_feature, load_graph
from stats import channel_stats, graph_stats
from visualize import plot_layer_head, plot_token_graph

sample = load_feature(".../attention_10005.pt")
metrics = channel_stats(sample)
plot_layer_head(sample, "concentration")

graph = load_graph(".../10005.pt")
print(graph_stats(graph))
plot_token_graph(graph)
```

`channel_stats` 当前提供 `prompt_mass / response_mass / concentration / support_size / response_mean_lag`，用于先研究正确与错误的图模式，再决定后续无监督模型。
