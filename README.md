# Attention Graph Lab

本仓库把 RAGTruth 的注意力缓存转换为可校验的 canonical attention，再从同一份缓存构建稀疏图。入口是 `main.py`：参数解析后分别调用归档、抽取、校验或构图类。

```text
正式 attention cache ──archive-attention──> canonical archive/{train,test}
RAGTruth + 观察模型 ──extract──────────────> canonical split
canonical split ──────build───────────────> graph dataset
```

`extract` 以观察模型的一次前向同时写 attention、token statistics，以及按需的 hidden states；`archive-attention` 则将已存在的正式 cache 转成相同的 canonical attention。当前正式 cache 的观察模型为 `Meta-Llama-3.1-8B-Instruct`，而 RAGTruth 响应筛选的生成器标识是 `llama-2-7b-chat`；两者不是同一概念。

## Canonical attention split

每个 split 的目录为：

```text
<canonical_root>/<train|test>/
├── manifest.json
├── index.jsonl
├── labels.jsonl                    # 归档正式 cache 时存在；构图不读取
├── attention/<sample_id>.npz       # 必需
├── hidden/<sample_id>.npz          # 可选，独立 sidecar
└── token_stats/<sample_id>.npz     # 可选，独立 sidecar
```

`index.jsonl` 至少包含 `sample_id`、`source_id`、相对 `path`、`sha256` 和 `bytes`。已完成的 canonical archive 可用 `enrich-index` 原地补充 `split / task_type / data_source / generator_model / temperature / quality`，不会重写 NPZ；若图已经构建，可同时用 `--graph-root` 刷新 graph manifest 中绑定 canonical JSON 的输入哈希。研究访问接口见 [`docs/research_data_access.md`](docs/research_data_access.md)。

`manifest.json` 记录 split 的 `index_sha256`、attention 几何、`attention_floor` 和对齐方式；读取时会校验索引，构图时还会校验每个 NPZ 的哈希。

每个 `attention/<sample_id>.npz` **恰好**有以下六个字段：

| 字段 | 形状 | dtype | 含义 |
| --- | --- | --- | --- |
| `token_ids` | `[N]` | `int32` | prompt 与 response 的 token ID |
| `response_idx` | `[]` | `int32` | 第一个 response token 的位置 |
| `attention_diagonal` | `[L,H,N]` | `float16` | 每层每头完整 attention 对角线 |
| `response_row_ptr` | `[L*H*R+1]` | `int32` | response-query 稀疏 attention 的 CSR 行指针 |
| `response_column_indices` | `[M]` | `int32` | 每项的早先 source token 位置 |
| `response_values` | `[M]` | `float16` | 对应 attention 权重 |

这里 `R=N-response_idx`，CSR 行号为 `((layer * H + head) * R + target - response_idx)`。attention 的对齐约定为 `post_token_query_at_same_position`：位于 `t` 的 response query 对应同一 token 位置 `t`，并且 CSR 只允许指向严格更早的 token。下游若采用“预测下一个 token”的时间语义，必须自行处理这一个位置的偏移，不能把两种对齐混为一谈。

`attention_floor` 属于 split manifest，而不重复存入 NPZ。写入 CSR 的是 response-query、非对角且 `attention > floor` 的值；未出现的值只可解释为 `<= floor`，并不等于精确的零。`tau` 是之后构图时的阈值：对 `original` 与 `hypergraph`，必须满足 `tau >= attention_floor`。`floor` 决定有损存储，`tau` 决定从已存数据中选择哪些关系，二者不可互换。

可选 sidecar 不改变 attention 的六字段契约：

```text
hidden/<sample_id>.npz
  token_ids [N] int32; hidden_layer_ids [K] int16; hidden_states [K,N,D] float16

token_stats/<sample_id>.npz
  token_ids [N] int32; token_log_prob [N] float32; entropy [N] float32
```

完整 logits 不落盘；位置 0 的两个 token statistics 都为 0。加载 sidecar 时会用 `token_ids` 与 attention 样本对齐。

## 命令

从仓库根目录运行（必要时以 `PYTHONPATH=.` 让模块可见）。新抽取的 split 输出到传入的 `--output-dir`：

```bash
python main.py extract \
  --model-path /models/Meta-Llama-3.1-8B-Instruct \
  --dataset-path /data/RAGTruth/dataset \
  --output-dir /data/model_traces/run/train \
  --split train \
  --generator-model llama-2-7b-chat \
  --floor 0.01 \
  --hidden-layers 7,15,23,31 \
  --device cuda
```

省略 `--hidden-layers` 时仍写 attention 和 `token_stats`，但不写 `hidden/`。归档已有正式 cache 时，输出根目录将含 `train/` 与 `test/`：

```bash
python main.py archive-attention \
  --formal-root /path/to/formal_cache \
  --output-root /data/RAGTruth/model_traces/llama31_8b
python main.py verify-attention \
  --archive-root /data/RAGTruth/model_traces/llama31_8b
```

已重构完成后，仅补 research metadata：

```bash
python main.py enrich-index \
  --canonical-root /data/RAGTruth/model_traces/llama31_8b \
  --dataset-path /data/RAGTruth/dataset \
  --graph-root /data/RAGTruth/graphs/llama31_8b/original_tau0p01
```

旧 PT feature trace 可仅转换为独立 sidecar：

```bash
python main.py archive-features --trace-dir /path/to/traces --output-dir /path/to/split
```

## 图数据集

图构建读取一个 canonical split，并写入：

```text
<graph_root>/<train|test>/
├── manifest.json
├── index.jsonl
└── graphs/<sample_id>.pt
```

`manifest.json` 绑定输入 attention 的 manifest/index 哈希，并记录自己 `index_sha256`、构图参数及对齐方式。所有图均为 topology-only：绝不存 `x`、`token_ids`、`labels`、`y` 或 `y_token`；`num_nodes=N` 始终保留，包括没有边的孤立 token。隐状态、统计特征和任何学习得到的 embedding 都是下游按需加载或训练产生的输出，不是图缓存的一部分。图从不构造或保存稠密邻接矩阵。

| `--kind` | 稀疏布局 | 语义 |
| --- | --- | --- |
| `original` | `num_nodes`, `response_idx`, `edge_index [2,E]`, `edge_type [E]`, `edge_ptr [E+1]`, `edge_channel [T]`, `edge_value [T]` | `attention > tau` 的旧 threshold-union 拓扑；每条边的 layer/head 值以 CSR 形式保存 |
| `relation_topk` | `num_nodes`, `response_idx`, `edge_index [2,E]`, `edge_type [E]`, `edge_weight [E]` | 每个 response target 分别按平均通道分数选择 prompt 与 response-history 的 top-k |
| `relation_topk_channels` | `relation_topk` 的字段加 `edge_ptr [E+1]`, `edge_channel [T]`, `edge_value [T]` | top-k 拓扑加逐通道的稀疏注意力值 |
| `hypergraph` | `num_nodes`, `response_idx`, `incidence_index [2,I]`, `incidence_weight [I]`, `hyperedge_target [Q]`, `hyperedge_channel [Q]`, `hyperedge_type [Q]` | 按 `(layer, head, response target, prompt/history)` 建立超边；以 node–hyperedge incidence 表示 |

`edge_type` / `hyperedge_type` 为 0（prompt→response）或 1（response-history→response）。`original`、`relation_topk_channels` 的通道属性都使用 `edge_ptr` 分段的 COO/CSR 式稀疏表示；`hypergraph` 使用 incidence COO。只有兼容旧代码时，才在内存中调用 `graphs.dense_edge_attr()` 临时物化稠密 `[E,L*H]`。

relation top-k 仅在 `floor` 后保留的 CSR 项上评分，并以缺失通道为 0 计入通道平均；因此它不是对完整稠密 attention 的精确 top-k。

例如构建带通道的 relation top-k 图：

```bash
python main.py build \
  --cache-dir /data/RAGTruth/model_traces/llama31_8b/train \
  --output-dir /data/RAGTruth/graphs/llama31_8b/relation_topk_channels/train \
  --kind relation_topk_channels --k-prompt 8 --k-history 8 --device cuda
```

`--tau` 仅用于 `original` 与 `hypergraph`；relation top-k 使用 `--k-prompt` 和 `--k-history`。

## 可复现重建与清理

[`scripts/rebuild_ragtruth.sh`](scripts/rebuild_ragtruth.sh) 依次归档、校验，并为 `train` 和 `test` 构建 `relation_topk_channels`。默认输出布局是：

```text
$CANONICAL_ROOT/{train,test}/...       # 默认 .../data/RAGTruth/model_traces/llama31_8b
$GRAPH_ROOT/{train,test}/...           # 默认 .../data/RAGTruth/graphs/llama31_8b/relation_topk_channels
```

`rebuild_ragtruth.sh` 只用于全新的输出：`CANONICAL_ROOT` 和 `GRAPH_ROOT` 两个最终 root 都必须尚未存在，且二者互不包含。它先在与最终 root 相邻、确定的 PID staging path 中完成归档、校验和两个 split 的构图，全部成功后再按先 canonical、后 graph 的顺序改名发布。对于普通可捕获失败（包括两次 rename），失败时会将 canonical 回退并只清理自己的 staging path，不发布部分构建结果。SIGKILL 或极端 rename 失败仍可能留下完整 canonical 或 `.staging.<pid>`；恢复时应先核验，再移除 staging 或只为 `GRAPH_ROOT` 单独重构，因此该过程不声称绝对原子。

可覆盖路径和参数后执行：

```bash
FORMAL_ROOT=/path/to/formal_cache \
CANONICAL_ROOT=/data/RAGTruth/model_traces/llama31_8b \
GRAPH_ROOT=/data/RAGTruth/graphs/llama31_8b/relation_topk_channels \
bash scripts/rebuild_ragtruth.sh
```

[`scripts/cleanup_legacy_ragtruth.sh`](scripts/cleanup_legacy_ragtruth.sh) 会先校验 canonical archive，默认仅打印待删目录（dry run）：

```bash
bash scripts/cleanup_legacy_ragtruth.sh
```

确认删除默认的 legacy graph 目录必须显式设置：

```bash
DRY_RUN=0 CONFIRM_DELETE=DELETE_RAGTRUTH_LEGACY \
bash scripts/cleanup_legacy_ragtruth.sh
```

若还要删除 formal cache，另加 `DELETE_FORMAL=1`；脚本拒绝 canonical archive、本体重叠路径以及 `$SAFE_ROOT` 外的目标。

清理脚本的删除对象始终是固定路径：

```text
$SAFE_ROOT=/share/home/tm902089733300000/a903202310/lys
legacy graph=/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/ragtruth_original_attribute_graphs/fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05
formal cache=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

只有 `CANONICAL_ROOT` 和 `GRAPH_ROOT` 用来选择替代数据并进行验证；固定的删除对象不会跟随 `FORMAL_ROOT` override 改变。如果固定 formal cache 存在，脚本在任何 legacy 删除前会校验其两个 split manifest 与 canonical 记录的来源 SHA256。该脚本只接受 `relation_topk_channels` replacement graph；若替代数据是 `original_tau0p01`，不要直接运行该清理脚本。

## t-SNE 分析

跨样本投影通过 `ResearchDataset` / `ResearchSample` 统一加载 canonical attention、可选缓存图和标签；notebook 本身不读取 manifest、index、NPZ 或 PT。安装分析依赖并启动 notebook：

```bash
python -m pip install -r requirements-analysis.txt
jupyter lab notebooks/graph_tsne.ipynb
```

`GraphTSNEAnalysis(split_root, output_dir, graph_root=None, tau=.01, node_feature_mode="attention", seed=0).run()` 每个回答生成一个点：`ResearchSample.structural_features(graph)` 的 12 个原始阈值图状态（含 channel-edge density）按 mean/std/slope 汇总为 36 维 `topology` 描述符；node 特征也按相同方式汇总；`combined` 分别标准化两个 block 后按各自维度的 `sqrt` 缩放，使总尺度等权。高维输入先 PCA 到最多 50 维，再拟合 t-SNE。`node_feature_mode` 不能为 `none`。

有两种明确的图来源：

- `graph_root=None`：对每个 `ResearchSample` 调用 `original_graph(tau)`，直接由 `<split_root>/attention/*.npz` 的 canonical CSR 现场构图；不需要预构建图缓存。
- `graph_root=/.../original_tau0p01/<split>`：对每个样本调用 `sample.graph("graph")`，使用已验证 provenance 的 `original` 缓存图，且缓存 manifest 的 `parameters.tau` 必须等于传入的 `tau`。`relation_topk`、`relation_topk_channels` 和 hypergraph 都不适用于 topology t-SNE。

所有样本先完成特征、标准化、PCA 和 t-SNE，再调用 `dataset.labels()` 仅为颜色读取标签。输出目录固定保存 `graph_tsne.png`、`graph_tsne_response_length.png`（同一坐标按回答长度着色，用于排查长度混杂）和 `graph_tsne_coordinates.npz`（`sample_id`、`response_tokens`、`topology`、`node`、`combined`）。

单个样本的 token 级视图使用 `notebooks/sample_behavior.ipynb`：`SampleGraphVisualizer.visualize()` 生成 token graph、热图、轨迹、对照及基于 12 维结构状态的 node t-SNE；`BehaviorAnalysis.single()` 另生成 `token_tsne.png`，这是 onset 工作流使用的 11 维图行为特征投影。两者都以 response token 为点，但不是重复的嵌入；11 维投影先完成标准化和 t-SNE，之后才读取 hallucination span 上色。
