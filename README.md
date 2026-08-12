# Attention Graph Hallucination Detection

这个仓库只保留一条研究主线：**从 LLM attention 构图，学习 token 节点表示，再做无监督异常检测。**

```text
RAGTruth + observer LLM
        ↓
canonical sparse attention
        ↓
RP/RR attributed token graph
        ↓
relation/channel-aware GNN
        ↓
self-supervised masked reconstruction
        ↓
learned response-token embeddings
        ↓
train-only robust residual calibration
        ↓
held-out token anomaly scores
        ↓
labels.jsonl only for final evaluation / coloring
```

手工统计量不再作为 GNN 输入，也不再用于主 t-SNE。它们只由 `attention_graph/statistics.py` 在**全部样本**上生成，作用是验证假设、做 baseline 和解释模型。

## 模块职责

```text
cache.py                 canonical attention 格式与校验
ragtruth.py              RAGTruth 读取、tokenization、标签 sidecar 对齐
extract.py               observer LLM -> canonical attention
archive.py               旧 formal cache -> canonical attention
metadata.py              canonical index 的 RAGTruth metadata
research_dataset.py      canonical split 的 lazy data access + evaluation labels

attention_graph/
  graph.py               canonical attention -> RP/RR sparse attributed graph
  model.py               learned channel fusion + CHARM-style GNN + reconstruction losses
  train.py               label-blind train/validation/calibration
  score.py               frozen embedding + leave-one-token-out anomaly residual
  statistics.py          all-data scalar diagnostics; never used as GNN input
  evaluate.py            frozen scores后才读取 labels
  visualize.py           learned node embedding t-SNE; coordinates固定后才读取 labels

main.py                  唯一命令行入口
```

完整方法见 [`docs/method.md`](docs/method.md)。

## MART: non-GNN mechanism baseline

MART reads canonical CSR rows directly, without a second graph-support selection.
For every response token it retains the fraction of **retained** causal mass
coming from prompt, the entropy of retained source entries plus diagonal and one
censored-OTHER bucket, their anchor `q * (1 - H)`, retained/diagonal/OTHER mass,
channel mean/std, signed late-minus-early layer drift, and causal EMA innovation. A detector
fits robust position-bin normalization, PCA whitening, and kNN novelty only on
the canonical train split; relative position conditions calibration but is not a
kNN feature. Test data are transformed by the frozen detector. The deterministic
reference set is capped by `--reference-size` (default 100000) to bound kNN memory.

```bash
python main.py fit-mart --train-split /data/RAGTruth/model_traces/llama31_8b/train \
  --output outputs/mart/model.npz --device cuda
python main.py score-mart --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --checkpoint outputs/mart/model.npz --output outputs/mart/test_scores.npz --device cuda
python main.py evaluate --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --scores outputs/mart/test_scores.npz --output outputs/mart/evaluation.json
```

MART is the primary non-GNN baseline.  Prior statistics select a representation,
not an anomaly direction.  A GNN is necessary only if it beats MART and a
no-message baseline, while source-shuffling its graph support reduces results.

## Canonical attention

每个样本只需要六个 attention 字段：

```text
token_ids
response_idx
attention_diagonal        [L,H,N]
response_row_ptr          [L*H*R+1]
response_column_indices
response_values
```

`response_values` 是 `attention > attention_floor` 且 `source < target` 的 response-query sparse trace。弱于 floor 的 channel 是 **censored / 未观察到**，不能当成精确 0。

`labels.jsonl` 与 attention 文件分离：

```json
{"sample_id":"10071","positive_runs":[[81,84],[85,87]]}
```

`positive_runs` 是 response-relative `[start,end)`；只在 `evaluate`、`evaluate-statistics` 和 `visualize` 的最后着色阶段读取。

## 1. 数据准备

从原 RAGTruth + observer 模型抽取：

```bash
python main.py extract \
  --model-path /models/Meta-Llama-3.1-8B-Instruct \
  --dataset-path /data/RAGTruth/dataset \
  --output-dir /data/RAGTruth/model_traces/llama31_8b/train \
  --split train \
  --generator-model llama-2-7b-chat \
  --floor 0.01 \
  --device cuda
```

已有 formal cache：

```bash
python main.py archive-attention \
  --formal-root /path/to/formal_cache \
  --output-root /data/RAGTruth/model_traces/llama31_8b
```

校验：

```bash
python main.py verify-attention \
  --archive-root /data/RAGTruth/model_traces/llama31_8b
```

## 2. 全数据统计诊断

这一步不训练模型，也不筛选错误样本。默认处理传入 split 的**所有 response token / 所有 response**：

```bash
python main.py statistics \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --output outputs/statistics.json \
  --device cuda
```

冻结统计结果后才打开标签计算单特征 AUROC：

```bash
python main.py evaluate-statistics \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --statistics outputs/statistics.json \
  --output outputs/statistics_evaluation.json
```

这里回答的是“哪些可解释 attention 图统计与错误有关”，不是主检测器。

## 3. 无监督 GNN 训练

只在 canonical **train** split 上训练。source_id 被拆为 mutually exclusive train / validation / calibration 三组；三组都不读取 hallucination label。

```bash
python main.py train \
  --train-split /data/RAGTruth/model_traces/llama31_8b/train \
  --output-dir outputs/model \
  --device cuda \
  --embedding-dim 64 \
  --message-steps 2 \
  --epochs 50
```

默认图是 threshold-union：只要某 layer/head retained trace 存在，该 causal token pair 就形成边。每条边仍保存所有 retained `(layer, head, value)` trace；不会先平均成一个 edge feature。也支持 `typed_mass_cover`：分别在 RP/RR 内选择覆盖指定 retained relation mass 的最小 source support，用作构图方式消融。

训练时随机选 response targets，**同时遮蔽该 token 的 attention diagonal 与全部 incoming RP/RR edges**，再要求 GNN 重建：

1. incoming edge support；
2. retained layer/head attention weight；
3. attention-row distribution + censored OTHER mass；
4. masked node attention diagonal。

因此训练信号来自图自身，不来自 `positive_runs`。

## 4. 冻结模型打分

正式打分默认 `target-block-size=1`，即逐 token leave-one-out：

```bash
python main.py score \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --checkpoint outputs/model/model.pt \
  --output outputs/test_scores.npz \
  --device cuda
```

输出包含每个 response token 的：

- learned GNN embedding；
- RP/RR support reconstruction residual；
- RP/RR attention-weight residual；
- row-distribution residual；
- node-diagonal residual；
- train-calibration median/MAD 标准化后的 anomaly score。

该文件不含 label。

## 5. 最终评估

```bash
python main.py evaluate \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --scores outputs/test_scores.npz \
  --output outputs/evaluation.json
```

只有这一步打开 `labels.jsonl`。token-level 是主指标；response-level 仅作为 secondary aggregation。

## 6. Learned embedding 可视化

```bash
python main.py visualize \
  --canonical-split /data/RAGTruth/model_traces/llama31_8b/test \
  --scores outputs/test_scores.npz \
  --output-dir outputs/visualization
```

t-SNE 输入是 **GNN learned node embedding**，不是 degree/entropy 等手工统计。节点采样、标准化和 t-SNE 都先完成，之后才读取 label 给同一坐标着色。

## 后续核心消融

主实验应围绕真正的问题逐项加入消融；当前核心代码已经支持 `message_steps=0` 和四种 support selection，其余作为下一步实现：

- `message_steps=0`：没有消息传递；
- layer/head channel mean：验证逐 channel 信息是否必要；
- source shuffle：保持边数量/target/relation，破坏 source 对齐；
- collapse RP/RR：验证 prompt grounding 与 response-history 路由的关系类型是否必要；
- threshold / top-k / typed mass-cover support selection：验证构图方式。

这些消融应比较同一 train/test protocol 下的 token AUROC/AUPRC，而不是通过“t-SNE 是否好看”判断方法有效。
