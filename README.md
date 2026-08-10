# Attention Graph Lab

这个仓库只做四件事：**抽取 attention、把旧 cache 转成统一格式、构图、分析图**。

## 1. 唯一 attention 数据接口

每个样本 `.npz` **只保存 6 个数组**：

| 字段 | shape | dtype | 含义 |
| --- | --- | --- | --- |
| `token_ids` | `[N]` | `int32` | prompt+response token id |
| `response_idx` | `[]` | `int32` | 第一个 response token 的位置 |
| `attention_diagonal` | `[L,H,N]` | `float16` | 每层每头的 self-attention 对角线 |
| `response_row_ptr` | `[L*H*R+1]` | `int32` | response attention 的 CSR 行指针 |
| `response_column_indices` | `[M]` | `int32` | 被关注的 source token 位置 |
| `response_values` | `[M]` | `float16` | 对应 attention 权重 |

其中 `R=N-response_idx`，CSR 行号为：

```text
row = (layer * H + head) * R + (target - response_idx)
```

没有另外保存 `node_role`、`edge_type`、layer/head id：它们都可以从上述字段直接得到。

### split 目录

```text
train/
├── manifest.json
├── index.jsonl
├── labels.jsonl          # 可选，构图不读取
└── attention/
    ├── 10001.npz
    └── ...
```

`manifest.json` 只保存整个 split 共享的 `attention_floor / num_layers / num_heads / count`。
`index.jsonl` 只保存 `sample_id / source_id / path`。

`attention_floor` 必须保留在 manifest：CSR 中没出现的 attention 只能解释为 `<= floor`，不能当成精确 0。

## 2. 现有 RAGTruth 数据

正式旧 cache：

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

一次性转换：

```bash
PYTHONPATH=. python main.py archive-attention \
  --formal-root /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --output-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_attention
```

以后构图只读转换后的 `train/` 或 `test/`，不再兼容旧 PT 字段。

## 3. 构图

```bash
PYTHONPATH=. python main.py build \
  --cache-dir /share/home/.../model_traces/llama31_attention/train \
  --output-dir /share/home/.../graphs/original/train \
  --kind original --tau .05 --device cuda
```

当前实现：

- `original`：复现旧 threshold-union attributed graph；
- `relation_topk`：每个 response target 分别保留 prompt/history top-k；
- `relation_topk_channels`：top-k 图同时保留稀疏 layer/head channel 值；
- `hypergraph`：按 `(layer, head, target, PR/RR)` 构造超边。

图文件不保存标签。`node role` 由 `response_idx` 得到；PR/RR 由 edge source 与 `response_idx` 得到。

## 4. 新数据抽取

未来重新抽 attention 时直接写相同的 6 字段格式：

```bash
PYTHONPATH=. python main.py extract \
  --model-path /share/home/.../Meta-Llama-3.1-8B-Instruct \
  --dataset-path /share/home/.../RAGTruth/dataset \
  --output-dir /share/home/.../model_traces/new_run/train \
  --split train --floor .01 --device cuda
```

这样旧数据迁移和新数据抽取最终使用完全相同的构图接口。
