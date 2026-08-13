# Attention Graph Hallucination Detection

本项目直接读取已有压缩稀疏 attention，为每条样本生成逐 token 的无监督图表征。当前唯一主线不训练 GNN、不反向传播，也不把 32×32 个 layer-head 通道提前平均。

```text
compressed response-query CSR
  -> windowed Lookback [token, layer, head]
  -> complete L×H node vector (32×32 = 1024 dimensions)
  -> strongest-head voting builds one sparse route per layer
  -> fixed multiscale graph filter: X-M1 and M1-M2
  -> prompt-bin evidence flow + graph-connected anomaly components
  -> incoming-route/weight-matched randomized topology control
  -> four train-only one-class references; labels opened only for evaluation
```

## 一键运行

```bash
git pull --ff-only origin main && CUDA_VISIBLE_DEVICES=0 bash run_token_representation.sh
```

指定样本图：

```bash
SAMPLE_IDS=11445,11289 CUDA_VISIBLE_DEVICES=0 bash run_token_representation.sh
```

已有完整输出不需要重新计算特征，可以只重画某条样本。以下命令针对已发现信号最强的 layer 4；省略 `--display-layer` 时，每对 source-target 使用所有层中的最大路由权重：

```bash
python main.py render-token-graph \
  --test-split /path/to/formal_cache/test \
  --output-dir /path/to/existing/token_representation/output \
  --sample-id 12471 \
  --display-layer 4
```

程序分八个阶段显示进度，直接使用原始压缩缓存，不转换、不复制、不重新抽取 attention。核心参数是 `PROMPT_BINS=16`、`GRAPH_HEAD_COMPONENTS=8` 和 `LOOKBACK_WINDOW=8`。

## 为什么只为 Lookback 保留 1024 维

Lookback Lens 对每个 token 分别计算每个 layer-head 的 Lookback，只沿时间窗口聚合。对于 32 层、32 头模型，节点表征因此是完整的 1024 维：

\[
L_{t,l,h}=\frac{\frac1P\sum_{p<P}a_{l,h,t,p}}
{\frac1P\sum_{p<P}a_{l,h,t,p}+\frac1{t+1}\left(\sum_{r<t}a_{l,h,t,r}+a_{l,h,t,t}\right)}.
\]

这 1024 个坐标代表不同层、不同头的直接路由偏好，提前平均会消掉“少数有效层头”的信号。旧的单标量 `direct_lookback_anomaly` 只保留为兼容诊断，不进入主节点表征。

缓存只保存高于 `attention_floor` 的边；程序不会猜测或重建被截断的原始 attention，所有质量、路由和传播都明确解释为 retained attention。

## 图传播不是几个手工标量

完整 1024 维仍作为直接节点状态保存。每层采用所有 retained head 的并集构图，同一 `(query, source)` 的边权取最强 head：

\[
B_{l,t,s}=\max_h a_{l,h,t,s}.
\]

仅保留 `B>=attention_floor` 的路由边。这里不做 layer/head 平均，因此单个明确的功能 head 不会被其他 head 稀释。

传播前只在每层内部把 32 个 head 投影到 8 个固定正交通道。投影由随机种子唯一确定，不读取数据或标签；原始 1024 维节点文件保持不变。对 response 路由定义原始质量矩阵 $W_l$，并计算条件邻居消息：

\[
M^{(1)}_l=\frac{W_lX_l}{W_l\mathbf1},\qquad
M^{(2)}_l=\frac{W_l^2X_l}{W_l^2\mathbf1}.
\]

图滤波器输出保留符号、并由原始路径质量连续门控的两级创新：

\[
\Psi_1=(W\mathbf1)(X-M^{(1)}),\qquad
\Psi_2=(W^2\mathbf1)(M^{(1)}-M^{(2)}).
\]

因此它检测的是节点相对真实邻居的变化和跨尺度变化；当路径质量趋近 0 时传播也连续趋近 0，不会让一条极弱边获得完整消息。它不是把 degree、mass、entropy 等标量简单平均。

## 多跳传播如何建模不直接相邻关系

Prompt 不再只压缩成质量、质心和方差。Prompt 被划分为 16 个位置区间，直接证据是：

\[
G^{(0)}_{t,l,b}=\sum_{p\in\mathcal B_b}B_{l,t,p}.
\]

它通过同一 RR 路由形成 $G^{(1)}$ 和 $G^{(2)}$，并保存 $G^{(0)}-G^{(1)}$、$G^{(1)}-G^{(2)}$。这样可以区分“直接关注了哪个 prompt 区域”和“经历史 response 间接继承了哪个区域”。原始 $W\mathbf1$、$W^2\mathbf1$ 单独保留路径可靠质量，极弱路径不会因为条件归一化被伪装成强路径。

## 如何证明图结构确实有效

程序在标签打开前冻结六个分数：

- `token_only`：仅完整 Lookback；
- `direct_edges`：直接 prompt-bin 路由与 RR 一跳质量；
- `true_propagation`：真实 attention 拓扑上的两级图创新；
- `randomized_propagation`：保持每个 RR 目标、层、路由条目数和每条边权，只随机合法历史源节点（允许 null 中出现平行路由）；
- `evidence_flow`：前三个真实视图的 train-calibrated 组合；
- `randomized_topology_control`：把真实传播替换为随机传播的完整对照。

每个视图只使用无标签 train，按相对位置拟合 median/MAD 和 PCA 正常子空间。主验证要求同时满足：

- `evidence_flow` 优于 `token_only`；
- `true_propagation` 优于 `randomized_propagation`；
- AUROC 和 AUPRC 都报告；
- 对整条 response 做 200 次 paired cluster bootstrap，不能把同一样本的 token 拆散重采样。

如果真实传播没有稳定优于随机传播，就不能声称构图有效；这时应修改路由，而不是继续调整可视化。超过 train 95% 分位数的节点再按照 RR 边合并为异常连通分量。

逐坐标 separability 是后验机制发现，不是可部署分数；从中选择层头或机制后，必须在 validation 冻结选择，再在新 held-out test 上报告。

## 输出

- `token_node_representations.float16.npy`：全部 test token 的完整 `L×H` Lookback 节点向量；
- `evidence_flow_node_embeddings.float16.npy`：真实拓扑扩散创新的 train-only PCA 节点表征；
- `compact_layer_structure.float16.npy`：`[mechanism, token, layer]` 的紧凑图状态；
- `train_reference_model.npz`：train-only median/MAD 与 PCA 参数；
- `token_representations_label_free.npz`：metadata、兼容统计、无监督分数和二维坐标，不含标签；
- `sample_graphs/*.npz`：每条样本实际用于传播的多层稀疏 COO 路由及全局节点行区间；
- `token_representation_report.json`：六视图指标、真实/随机拓扑增益、样本级 bootstrap、分任务结果和固定阈值检测；
- `population_token_representations.png`：train-only PCA 与无监督分数分布；
- `sample_*_attention_structure_*.png`：四个相互对应的结构视图，不再把 token 压在两条水平线上：
  1. 全部 response 节点的真实拓扑扩散空间，节点颜色为无监督异常分数，标签只作为青色外圈；
  2. RP 加权邻接矩阵，横轴是 prompt source、纵轴是 response target，并叠加 hop-1 prompt provenance 的质心与范围；
  3. RR 加权邻接矩阵，离对角线的距离直接表示历史跨度；
  4. 固定 train 阈值下的逐 token 异常分数，以及由 RR 边连接出的异常分量。
