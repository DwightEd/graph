# Attention Graph Hallucination Detection

本项目从观察模型的压缩稀疏 attention 直接得到每个 response token 的无标签表征，并检验真实 RP（prompt→response）/RR（response→response）图是否提供额外信息。主实验不训练 GNN，也不把 hallucination 标签用于投影、构图或异常检测器。

```text
compressed CSR attention
  -> [token, layer, head, 4 validated mechanisms]
  -> train-only robust PCA -> token-only embedding
  -> fixed RP/RR propagation -> token+graph embedding
  -> train-only prototype novelty score
  -> labels opened only for AUROC/AUPRC and coloring
```

## 一键运行

在仓库根目录执行：

```bash
git pull origin main && CUDA_VISIBLE_DEVICES=0 bash run_token_representation.sh
```

正式缓存路径和研究环境已写成默认值，也可显式覆盖：

```bash
FORMAL_ROOT=/path/to/formal_cache \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda \
SAMPLE_IDS=11289,42 \
bash run_token_representation.sh
```

脚本会持续显示六个阶段及逐样本进度条，不转换或复制原 attention。输出包括：

- `token_representations_label_free.npz`：全部 test token 的 `token_only`、`token_graph`、`no_rp`、`no_rr` 表征和冻结分数，不含标签；
- `sample_graphs/*.npz`：每个 test 样本的 token、邻接边、边类型/权重、逐 hop 可达统计、带 block 边界的多尺度传播特征，以及两种主要 token 表征；
- `token_representation_report.json`：四个视图的总体、task、data source AUROC/AUPRC，以及 graph 相对 token-only 的增益；
- `population_token_representations.png`：冻结 PCA 坐标与分数分布；
- `sample_*_token_graph.png`：指定样本的真实 RP/RR 图和每个 response token 的二维表征。未指定时只按无标签 embedding dispersion 选一条，不按 AUROC 挑图。

## 表征具体是什么

每个 response token、每层、每个 head 只保留四个互补机制：

1. `routing_balance`：长度归一化的保留 RP 与 RR/diagonal 路由平衡；
2. `effective_support_fraction`：有效支撑数占所有合法 source 的比例，反映“连接少且集中”；
3. `dominant_edge_strength`：该 channel 最强的保留连接，反映“保留边更强”；
4. `response_locality`：RR attention 更偏向近邻还是远历史。

这些值不会先跨 layer/head 求均值，而是形成 `[L,H,4]` 张量并展平。只用 train token 拟合 median/MAD 和 PCA，得到初始 token embedding。低于 cache floor 的 unresolved mass 单独保存为 control，不进入表征。

图增量使用真实 source endpoint。令按 target 归一化的 RR 邻接为 (P_{RR})，初始 token 表征为 (H^{(0)})，默认显式保留三阶传播：

\[
H^{(k)}=P_{RR}H^{(k-1)},\quad k=1,2,3.
\]

RP source 位置先形成直接 prompt provenance (B^{(0)})，再通过 (B^{(k)}=P_{RR}B^{(k-1)}) 传到没有直接连接该 prompt 的后续 token。传播没有参数、没有反向传播。每个样本还保存各 hop 的可达祖先数和有效影响质量；单样本图分别显示直接边、由 (P^2/P^3) 得到的非相邻有效关系、最终 token embedding。报告同时比较：

- `token_only`：只使用 `[L,H,4]` 机制张量；
- `token_graph`：机制 token embedding + 完整 RP/RR 固定传播；
- `no_rp`：去掉 RP endpoint message；
- `no_rr`：去掉 RR endpoint 与邻居 message。

因此结果可以直接回答：有效机制本身是否足够，以及精确构图是否真的增加区分度，而不是把 t-SNE 是否“看起来分开”当结论。

## 数据接口

读取层原生支持当前正式稀疏 PT cache 和 canonical NPZ split。每条样本只要求：

```text
token_ids
response_idx
attention_diagonal        [layers, heads, tokens]
response_row_ptr          [layers * heads * response_tokens + 1]
response_column_indices
response_values
```

`labels.jsonl` 或正式 PT 内的 `y_token` 在表征和分数冻结前保持封存。

## 保留的诊断

仓库只保留 `statistics`/`evaluate-statistics` 作为回溯已有发现的诊断入口。需要反向传播的旧 GNN、MART 特征汤、Lookback/GMM、`discover-patterns` 和重复 graph-validation runner 均已删除；当前研究主入口只有 `represent-tokens`。
