# Attention Graph Hallucination Detection

从观察模型的因果 attention 构建 RP（prompt-to-response）和 RR（response-history-to-response）token 图。当前主实验先验证**怎样构图保留可用的无标签结构信息**；遮蔽重构 GNN 只是在构图证据充分后的后续基线。

```text
canonical sparse attention -> RP/RR causal graph -> one Lookback-ratio trajectory per response token
-> train-only kNN novelty score -> labels only for evaluation
```

## 首先运行：构图验证

`validate-graphs` 固定下文定义的 Lookback ratio 逐层表示，只改变图中可见的结构。
它在 train split 拟合 robust scaling + 有上限 reference 的 kNN，在 test split 产出 token 和连续 span
（默认 8 个 response token；短回答不纳入 span）分数；整个阶段不读取或使用标签。`evaluate-graphs` 才读取
`labels.jsonl`，输出 token/span AUROC、AUPRC、data source/task 分组，以及相对 full 图的差异。

```bash
python main.py validate-graphs \
  --train-split /data/RAGTruth/model_traces/llama31_8b/train \
  --test-split /data/RAGTruth/model_traces/llama31_8b/test \
  --output-dir outputs/graph_validation --device cuda \
  --variants full no_edges marginals source_rewire binary shuffle_layers

python main.py evaluate-graphs \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --artifact-dir outputs/graph_validation \
  --output outputs/graph_validation/evaluation.json
```

每个 `<variant>.npz` 是冻结的无标签 artifact，包含标准化后的 token/span 结构签名、分数、
token/span 元数据和拟合尺度；`label_free_manifest.json` 固化构图和运行配置。候选含义：

- `full`：原始稀疏 RP/RR 图；`no_edges`：移除 RP/RR 边，仅保留 self diagonal，Lookback 为零；
- `marginals`：保留 target×RP/RR×channel 质量，抹去确切 source；`source_rewire`：保持 target/relation/channel 的质量多重集但改写 source；
- `binary`：将边权和 diagonal 都二值化，只保留支撑集；`shuffle_layers`：同步打乱边 trace 和 diagonal 的 layer 顺序。

默认候选不含可选的 `collapse_relations`/`mean_heads`。`collapse_relations` 只抹去关系 metadata；
Lookback 按 source 的 prompt/response 边界计算，因此它应与 full 相同，不能检验 learned relation
embedding。`mean_heads` 则在计算非线性比值之前合并 head，是有效的 head-aggregation 消融，
不再被错误写成预期不变性。

Lookback 只汇总 RP 和 RR 两侧质量，因此 `marginals`、`source_rewire` 和
`collapse_relations` 对该表示都是预期不变性控制：它们能证明当前结果来自两侧质量比，而不能
证明“哪个 prompt token 是证据”。具体 source 连接仍保存在单样本图中，但不伪装成 Lookback
检测分数的一部分。

## 核心模块

```text
research_dataset.py        canonical split 的惰性访问；LabelStore 只供评估/着色
attention_graph/graph.py   CSR attention -> typed RP/RR 图
attention_graph/model.py   节点初始化 h0、RP/RR message passing hK、重构头
attention_graph/train.py   不读取标签的 GNN 训练
attention_graph/score.py   冻结模型的 token 异常分数
attention_graph/visualize.py  h0/hK 的论文式联合 t-SNE 投影
attention_graph/patterns.py  无训练的结构机制、投影与逐样本图
main.py                    唯一命令行入口
```

## 数据格式

每个 attention 样本仅保存：

```text
token_ids
response_idx
attention_diagonal        [layers, heads, tokens]
response_row_ptr          [layers * heads * response_tokens + 1]
response_column_indices
response_values
```

`labels.jsonl` 与 attention 分离；`positive_runs` 为 response-relative `[start, end)`，只可在评估或可视化着色阶段读取。

## 无监督 Lookback 构图与可视化

`discover-patterns` 不训练 GNN，也不把 degree、entropy、强边、lag 等特征再堆起来。
每个 response token 只有一个机制：Lookback ratio。对 layer (l)、head (h)、第 (t)
个生成 token，令 (P) 为保留的 prompt attention 总量、(R) 为先前生成 token 的
attention 总量、(D) 为缓存中保存的当前 token diagonal，则

\[
r_{lht}=\frac{P_{lht}/N_{prompt}}
{P_{lht}/N_{prompt}+(R_{lht}+D_{lht})/(t+1)}.
\]

这不是 prompt mass fraction：prompt 和 generated 两侧先分别除以各自 token 数，避免回答
越长时 RR 端仅因 token 更多而自动占优。比值先逐 head 计算，再平均 head，并将 layer 分成
连续 bins，得到节点向量 `[layer_bins]`。未保留的 cache 质量按零处理，其 unresolved mass 单独
保存为控制量，不进入节点表示。

主要无监督分数直接复现手工 Lookback baseline：`1 - mean(r)`，不拟合标签、不挑 layer/head。
train-only 相对位置 median/MAD 校准只用于 K-Means、t-SNE 和一个单独报告的控制分数，不会
覆盖这个原始基线。K-Means 数目只按 train Davies--Bouldin 选择；t-SNE 坐标冻结后才读取
test labels 着色。

```bash
bash run_lookback_graph.sh
```

该脚本默认直接读取正式缓存中的稀疏 `attention_*.pt`；读取层原生接受其中的
CSR 字段，不要求数据迁就另一套格式，不重新提取 attention，也不复制为 `.npz`。
每个阶段均输出编号，逐图处理有进度条，t-SNE 输出迭代日志。

输出包括全部 test token 的 landmark t-SNE、Lookback 逐层对比、原始/位置校准分数分布、
响应级模式占比，以及一条真实样本的完整 response-token 图和逐节点 Lookback 热图。未指定
样本时会在所有坐标和总体指标冻结后，事后选择一个同时含正确/幻觉 token 且样本内 AUROC
最高的可读示例；这个选择仅用于说明图，并在 report 中明确记录。指定真实样本可运行：

```bash
SAMPLE_IDS=123,456 bash run_lookback_graph.sh
```

完整图显示每个 response token；为了让边可读，只画每类覆盖 80% attention mass
后的最强若干 RP/RR 边。节点纵坐标是 `1 - mean Lookback`，下方面板是每个节点的
layer-bin Lookback 热图；红/绿标签只负责着色。显示剪枝不参与节点表征或模式发现。

公式与 [Lookback Lens 论文](https://aclanthology.org/2024.emnlp-main.84/) 及其
[官方 attention 提取实现](https://github.com/voidism/Lookback-Lens/blob/main/step01_extract_attns.py)
一致；不同点是原方法还报告监督 Logistic Regression，而本项目保留其无监督的直接均值基线。

## 训练、评分、评估

```bash
python main.py train \
  --train-split /data/RAGTruth/model_traces/llama31_8b/train \
  --output-dir outputs/gnn --device cuda

python main.py score \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --checkpoint outputs/gnn/model.pt --output outputs/gnn/test_scores.npz --device cuda

python main.py evaluate \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --scores outputs/gnn/test_scores.npz --output outputs/gnn/evaluation.json
```

`visualize` 需要当前版本训练出的 GNN checkpoint；它会严格核对 layer/head、observer、alignment 与 attention floor，不接受缺少这些来源信息的旧 checkpoint。

## 论文式 before/after 投影

该图不是单个样本的边图，也不读取 MART score artifact。对同一冻结 GNN、同一 response token：

- `(a)` 为 encoder node initialization `h0`，消息传递前；
- `(b)` 为同一节点经过完整 RP/RR message passing 后的 `hK`；
- source / target 指的是 `data_source` 或 `task_type` 的两个数据域，不是 RP/RR 边；
- 两个面板用同一批节点、联合标准化、PCA 和一次 t-SNE，因此共享坐标尺度；
- 标签在 t-SNE 固定后才读取，仅决定绿色/红色/蓝色/橙色和圆/x 的绘制样式。

真实远端示例：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

python main.py visualize \
  --canonical-split /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test \
  --checkpoint outputs/gnn/model.pt \
  --domain-field data_source \
  --source-domain 'CNN/DM' \
  --target-domain 'Recent News' \
  --output-dir outputs/embedding_shift/cnndm_vs_recent_news \
  --device cuda --max-nodes-per-domain 5000 --perplexity 30 --seed 0
```

输出：

```text
embedding_shift_tsne.png  一行两面板的论文式图
embedding_shift.npz       配对 h0/hK、坐标、token 元数据和事后标签
summary.json              样本量、阳性率、checkpoint、投影配置
```

`summary.json` 的结论范围固定为：`message-passing representation shift, not domain alignment`。该图用于检查消息传递怎样重排表示空间；不能单独证明跨域对齐或检测性能。

示例刻意选择同属 `Summary` 任务的 `CNN/DM` 与 `Recent News`，避免把 QA / Data2txt 的任务差异误画成域差异。
