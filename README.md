# Attention Graph Hallucination Detection

本项目直接读取已有压缩稀疏 attention，为每条样本生成逐 token 的无监督图表征。当前唯一主线不训练 GNN、不反向传播，也不把 32×32 个 layer-head 通道提前平均。

```text
compressed response-query CSR
  -> windowed Lookback [token, layer, head]
  -> complete L×H node vector (32×32 = 1024 dimensions)
  -> strongest-head voting builds one sparse route per layer
  -> prompt moments propagate over layer-level RR routes for multiple hops
  -> compact graph state [token, mechanism, layer]
  -> train-only calibration/PCA; labels opened only for final evaluation
```

## 一键运行

```bash
git pull --ff-only origin main && CUDA_VISIBLE_DEVICES=0 bash run_token_representation.sh
```

指定样本图：

```bash
SAMPLE_IDS=11445,11289 CUDA_VISIBLE_DEVICES=0 bash run_token_representation.sh
```

程序分八个阶段显示进度，直接使用原始压缩缓存，不转换、不复制、不重新抽取 attention。可用 `ROUTE_TOP_HEADS=4`、`PROVENANCE_HOPS=2` 和 `LOOKBACK_WINDOW=8` 修改三个核心参数。

## 为什么只为 Lookback 保留 1024 维

Lookback Lens 对每个 token 分别计算每个 layer-head 的 Lookback，只沿时间窗口聚合。对于 32 层、32 头模型，节点表征因此是完整的 1024 维：

\[
L_{t,l,h}=\frac{\frac1P\sum_{p<P}a_{l,h,t,p}}
{\frac1P\sum_{p<P}a_{l,h,t,p}+\frac1{t+1}\left(\sum_{r<t}a_{l,h,t,r}+a_{l,h,t,t}\right)}.
\]

这 1024 个坐标代表不同层、不同头的直接路由偏好，提前平均会消掉“少数有效层头”的信号。旧的单标量 `direct_lookback_anomaly` 只保留为兼容诊断，不进入主节点表征。

缓存只保存高于 `attention_floor` 的边；程序不会猜测或重建被截断的原始 attention，所有质量、路由和传播都明确解释为 retained attention。

## 为什么传播不再保留 1024 维

完整 1024 维用于保留直接 Lookback 信号；把十几个结构机制也全部扩成 1024 维，会重复保存同一批边并放大噪声。现在每层先对同一 `(query, source)` 边做 strongest-head voting：

\[
B_{l,t,s}=\frac1K\sum_{h\in\operatorname{TopK}}a_{l,h,t,s},
\]

缺失 head 按 0 计，默认 `K=4`，且仅保留 `B>=attention_floor` 的路由边。这不是 32 个 head 的普通均值：单个很强的 head 可以保留，一批仅略高于缓存阈值的弱边不会因取并集而充满整张图。

紧凑结构状态只保留每层的：

- prompt 保留质量、覆盖率、跨度、质心及质心变化；
- history 保留质量、可达历史覆盖率、RP/RR 边比例、加权 lag、lag 变化和 far-history 比例；
- 一跳/两跳 prompt provenance 的路径质量、prompt 质心和范围。

默认一共 `17×32=544` 维，并单独落盘，不与 1024 维 Lookback 拼接。对 73,994 个 token，结构文件约 0.08 GiB；旧的 `19×1024` 设计约 2.68 GiB。

## 多跳传播如何建模不直接相邻关系

当前 token 的 prompt attention 先写成位置矩：

\[
S^{(0)}_{t,l}=\sum_{p<P}B_{l,t,p}[1,u_p,u_p^2].
\]

再沿同一层的 response-history 路由传播：

\[
S^{(k)}_{t,l}=\sum_{r<t}B_{l,t,r}S^{(k-1)}_{r,l}.
\]

因此两跳量表示“当前 token 经由历史 response token 间接继承了多少 prompt 证据、证据来自 prompt 的哪个位置、范围多宽”。传播不做行归一化，所以连接少、路径弱和连接强会产生不同的路径质量。

## 无监督评分与信号审计

训练集只建立无标签 Lookback reference：按 response 相对位置拟合每维 median/MAD，并拟合 PCA。测试时报告：

- `robust_tail`：聚合绝对偏差最大的 5% 坐标，避免 1024 维全部平均；
- `subspace_residual`：到 train-only PCA 子空间的重构误差。

表征和分数落盘后才读取测试标签。报告分别审计 1024 个 Lookback 坐标和每层紧凑结构机制，输出最佳层头/层、raw AUROC、方向无关 separability，以及超过 0.60/0.65/0.70 的维数。窗口化 Lookback 同时报告当前 token 标签和 Lookback-Lens 风格的窗口标签。

逐坐标 separability 是后验机制发现，不是可部署分数；从中选择层头或机制后，必须在 validation 冻结选择，再在新 held-out test 上报告。

## 输出

- `token_node_representations.float16.npy`：全部 test token 的完整 `L×H` Lookback 节点向量；
- `compact_layer_structure.float16.npy`：`[mechanism, token, layer]` 的紧凑图状态；
- `train_reference_model.npz`：train-only median/MAD 与 PCA 参数；
- `token_representations_label_free.npz`：metadata、兼容统计、无监督分数和二维坐标，不含标签；
- `sample_graphs/*.npz`：每条样本实际用于传播的多层稀疏 COO 路由及全局节点行区间；
- `token_representation_report.json`：无监督分数、Lookback 通道信号和紧凑结构信号；
- `population_token_representations.png`：train-only PCA 与无监督分数分布；
- `sample_*_token_graph.png`：直接边、多跳 prompt provenance、所有 token 投影、Lookback 层轨迹和结构热图。
