# Attention Graph Hallucination Detection

从观察模型的因果 attention 构建 RP（prompt-to-response）和 RR（response-history-to-response）token 图，用无标签掩码重构训练图编码器，并在冻结模型上计算异常分数。

```text
canonical sparse attention -> RP/RR attributed token graph -> masked attention GNN
-> frozen response-token representation / anomaly score -> labels only for evaluation
```

## 核心模块

```text
research_dataset.py        canonical split 的惰性访问；LabelStore 只供评估/着色
attention_graph/graph.py   CSR attention -> typed RP/RR 图
attention_graph/model.py   节点初始化 h0、RP/RR message passing hK、重构头
attention_graph/train.py   不读取标签的 GNN 训练
attention_graph/score.py   冻结模型的 token 异常分数
attention_graph/visualize.py  h0/hK 的论文式联合 t-SNE 投影
attention_graph/patterns.py  无训练的多层 prompt 溯源模式发现
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

## 无训练的溯源模式发现

`discover-patterns` 不训练 GNN，也不拼接 degree、entropy、lag 等异质统计量。
它只研究一个图机制：从每个 response-token 节点沿 layer-ordered attention 图
向后追溯时，质量多快到达 prompt，以及尚未到达 prompt 的 response ancestry
是否集中在一条窄链中。默认主节点表示**只含 prompt-absorption curve**；
live-response concentration 只能用 `SIGNATURE_VIEW=response_concentration`
作为独立实验运行，两者不会拼接。cache 未观察质量作为独立控制曲线，不进入
聚类或 t-SNE。模式、坐标和代表节点冻结后才读取 test token labels。

```bash
bash run_provenance_patterns.sh
```

输出包括全部 test token 的 landmark t-SNE、模式中心曲线，以及每个模式中
最接近中心的真实 token ego graph。非 landmark 节点使用原结构空间近邻插值，
因此每个 test token 都有二维坐标；模式发现仍在原始结构曲线上完成。另外输出
正确/幻觉响应图的节点模式占比与相邻模式转移图，标签只参与这一步事后解释。

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
