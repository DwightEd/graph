# 稀疏注意力图构建

构图特征与图文件不使用标签。`extract`/`build` 不写标签，且图索引和 manifest
不保存标签字段；`archive-attention` 仅将正式 RAGTruth cache 中的 `y_token` 转成
隔离的 `labels/{train,test}.jsonl` sidecar，构图不读取该 sidecar。

唯一入口是 `main.py`：它只解析 CLI 参数，构造相应配置并调用 `run()`，最后打印一行摘要。

```text
CLI 参数 -> ExtractionConfig -> AttentionExtractor.run() -> attention cache
CLI 参数 -> BuildConfig      -> GraphDatasetBuilder.run() -> graph files/index/manifest
CLI 参数 -> ArchiveConfig    -> AttentionArchiveConverter.run() -> canonical archive
```

| 文件 | 职责 |
| --- | --- |
| `main.py` | `extract`、`build`、`inspect`、`archive-attention` 与 `verify-attention` 的唯一 CLI 入口。 |
| `extract.py` | 按样本提取并保存稀疏注意力缓存。 |
| `cache.py` | 固定 cache schema 的校验、保存与加载。 |
| `archive.py` | 正式 attention cache 的完整性校验、最小化归档、读取和验证。 |
| `build.py` | 枚举 cache，逐样本构图并写图数据集。 |
| `graphs.py` | 三种 token graph 的构图函数。 |
| `hypergraph.py` | attention hypergraph 的构图函数。 |
| `ragtruth.py` | 读取 RAGTruth 样本并完成 prompt/response 分词。 |
| `tests/` | 不依赖二进制 fixture 的单元测试。 |
| `README.md` | 数据契约、构建命令与运行约束。 |
| `requirements.txt` | 运行时依赖及 Transformers 版本锁定。 |

## `attention-response-csr-v1` cache 契约

记 `N` 为 token 数，`R=N-response_idx` 为 response token 数，`L` 为层数，`H` 为头数，`C=L*H` 为通道数，`M` 为 CSR 中的保留条目数。通道顺序为 layer-major、head-minor。

| 字段 | shape | dtype | 含义 |
| --- | --- | --- | --- |
| `schema` | 标量 | `str` | 固定为 `attention-response-csr-v1`。 |
| `sample_id` | 标量 | `str` | response 样本标识。 |
| `source_id` | 标量 | `str` | 原始 source 标识。 |
| `response_idx` | 标量 | `int` | response 在 `token_ids` 中的起始位置。 |
| `token_ids` | `[N]` | `int64` | 全上下文 token id。 |
| `attention_diagonal` | `[L,H,N]` | `float16`、`bfloat16` 或 `float32` | 每通道的 self-attention 对角线。 |
| `response_row_ptr` | `[C*R+1]` | `int64` | CSR 行指针。 |
| `response_column_indices` | `[M]` | `int32` | CSR 源 token 索引。 |
| `response_values` | `[M]` | 与抽取 dtype 相同的浮点型 | 保留的 attention 值。 |
| `attention_floor` | 标量 | `float` | 抽取保留阈值。 |

对于 layer、head 和 response target，CSR 行号是

```text
row = (layer * H + head) * R + (target - response_idx)
```

每行 `response_column_indices` 严格递增，且严格因果：`column < target`。抽取时的选择始终是 `attention.to(float32) > floor`；保存后的半精度值可能等于量化后的 floor，不能由存储 dtype 的直接比较反推选择结果。

## 图契约

构图在 `--device`（通常是 GPU）逐样本进行；每张图立即转为 CPU 再 `torch.save`，不会在 CPU 或 GPU 累积整个 split。合法 cache 即使 CSR 为空，或阈值后没有边/超边，也会输出空图。

### TokenGraph

令 `E` 为边数，`T` 为所选边对应的稀疏 trace 条目数。

| 字段 | shape | dtype | 含义 |
| --- | --- | --- | --- |
| `sample_id`, `source_id`, `response_idx` | 标量 | `str`, `str`, `int` | 来自 cache 的标识与切分点。 |
| `token_ids` | `[N]` | `int64` | 节点 token id。 |
| `node_attr` | `[N,C]` | cache `attention_diagonal` 的浮点 dtype | 通道化 diagonal 特征。 |
| `edge_index` | `[2,E]` | `int64` | 行 0 为 source，行 1 为 response target。 |
| `edge_type` | `[E]` | `int8` | `0`=prompt→response，`1`=response-history→response。 |
| `edge_attr`（仅 `original`） | `[E,C]` | cache `response_values` 的浮点 dtype | 每边的 dense 通道值。 |
| `edge_weight`（top-k） | `[E]` | `float32` | 已保留通道值（缺失计零）的通道均值。 |
| `trace_ptr`（channels） | `[E+1]` | `int64` | 每条 top-k 边在 trace 中的 CSR 指针。 |
| `trace_channel`（channels） | `[T]` | `int32` | 稀疏值的 channel 编号。 |
| `trace_value`（channels） | `[T]` | cache `response_values` 的浮点 dtype | 稀疏 channel 值。 |

`original` 和 `hypergraph` 的阈值选择都严格使用 `selection = response_values.to(torch.float32) > float(tau)`；权重仍保留原始 dtype。`tau` 必须有限、在 `[0,1]` 且不小于 cache 的 `attention_floor`。`relation_topk`/`relation_topk_channels` 在已保留条目中分别选择每个 target 的 prompt 和 history 前 `k_prompt`/`k_history` 条，并非完整注意力上的 top-k。

### AttentionHypergraph

令 `Q` 为超边数，`I` 为 incidence 数。

| 字段 | shape | dtype | 含义 |
| --- | --- | --- | --- |
| `sample_id`, `source_id`, `response_idx`, `token_ids`, `node_attr` | 同 TokenGraph | 同 TokenGraph | 节点及来源信息。 |
| `incidence_index` | `[2,I]` | `int64` | 行 0 为 token 节点，行 1 为 hyperedge id。 |
| `incidence_weight` | `[I]` | cache `response_values` 的浮点 dtype | source attention 或 target diagonal 权重。 |
| `hyperedge_target` | `[Q]` | `int64` | 每个超边的 response target。 |
| `hyperedge_channel` | `[Q]` | `int32` | 每个超边所属通道。 |
| `hyperedge_type` | `[Q]` | `int8` | `0`=prompt source，`1`=response-history source。 |

一个 `(channel, response target, edge_type)` 组合形成一条超边：它包含该类型下超过 `tau` 的 source token，以及 target token 自身。

## 直接构建现有 cache

下列命令在前台显示 `tqdm` 进度。输出根目录固定为远端数据目录；目标目录必须不存在或为空，输入 cache 目录必须存在且至少有一个顶层 `.pt` 文件。

```bash
GRAPH_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b_instruct__llama2_good_all__fullctx__c8847872bedf
CACHE_ROOT=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876

PYTHONPATH=. python main.py build \
  --cache-dir "$CACHE_ROOT/train" \
  --output-dir "$GRAPH_ROOT/relation_topk_channels/train" \
  --kind relation_topk_channels --k-prompt 8 --k-history 8 --device cuda

PYTHONPATH=. python main.py build \
  --cache-dir "$CACHE_ROOT/test" \
  --output-dir "$GRAPH_ROOT/relation_topk_channels/test" \
  --kind relation_topk_channels --k-prompt 8 --k-history 8 --device cuda
```

现有正式 raw cache 根目录是

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

其中 `train` 有 2,497 个样本、`test` 有 449 个样本，raw cache 合计 31,791,710,510 B（约 29.61 GiB）。旧 dense 图目录是

```text
/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/ragtruth_original_attribute_graphs/fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05  # legacy
```

旧 dense 图合计 158,239,660,436 B（约 147.37 GiB），其中 dense `edge_attr` 约占 93.10%。本地旧 `ragtruth_graph.tar.gz` 为 7,294,690,891 B（约 7.29 GB / 6.79 GiB），它不是 raw cache。

## 抽取与依赖

`requirements.txt` 固定 `transformers==4.46.3`。抽取契约依赖该版本中 Llama decoder layer 的 eager attention 输出：加载模型时使用 `attn_implementation="eager"`，前向调用 `output_attentions=True`、`use_cache=False`，并从每层返回的 `[batch, heads, tokens, tokens]` attention 收集数据。升级 transformers 前须重新验证该返回契约。

```bash
pip install -r requirements.txt
NEW_CACHE_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b_instruct__llama2_good_all__fullctx__c8847872bedf

PYTHONPATH=. python main.py extract \
  --model-path /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --dataset-path /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --output-dir "$NEW_CACHE_ROOT/train" --split train \
  --generator-model llama-2-7b-chat --floor .01 --dtype float16 --device cuda

PYTHONPATH=. python main.py extract \
  --model-path /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --dataset-path /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --output-dir "$NEW_CACHE_ROOT/test" --split test \
  --generator-model llama-2-7b-chat --floor .01 --dtype float16 --device cuda

PYTHONPATH=. python -m unittest discover -s tests
```

每个 graph 文件是 `{"schema": ..., "graph": graph.to_dict()}`；同级 `index.jsonl` 每行包含 `sample_id`、`source_id`、相对 `path`、`num_nodes` 和 `num_edges` 或 `num_hyperedges`，`manifest.json` 记录构建参数。


## Canonical RAGTruth attention archive

本转换的 feature/graph 是 label-free；labels 仅保存在隔离的
`labels/{train,test}.jsonl` sidecar，builder 不读取它们。不会删除 formal 源 cache
或已有旧图。

正式源 cache：

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

转换会校验 complete split manifest、文件 SHA256、源码一致的 replay
fingerprint 和逐样本 float16 契约；全部先写入同父 staging，内部
`verify-attention` 成功后才原子 rename。示例：

```bash
TRACE_ID=llama31_8b_instruct__llama2_good_all__fullctx__c8847872bedf
FORMAL_ROOT=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
TRACE_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/$TRACE_ID
GRAPH_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/$TRACE_ID

PYTHONPATH=. python main.py inspect --artifact-dir "$FORMAL_ROOT"
PYTHONPATH=. python main.py archive-attention --formal-root "$FORMAL_ROOT" --output-root "$TRACE_ROOT"
PYTHONPATH=. python main.py verify-attention --archive-root "$TRACE_ROOT"
PYTHONPATH=. python main.py build --cache-dir "$TRACE_ROOT" --split train --output-dir "$GRAPH_ROOT/relation_topk_channels/train" --kind relation_topk_channels --device cuda
PYTHONPATH=. python main.py build --cache-dir "$TRACE_ROOT" --split test --output-dir "$GRAPH_ROOT/relation_topk_channels/test" --kind relation_topk_channels --device cuda
```

canonical NPZ 只含六个张量：`token_ids:int32`、`response_idx:int32`、
`attention_diagonal:float16`、`response_row_ptr:int32`、
`response_column_indices:int32`、`response_values:float16`。
