# Structural Attention Routing Fingerprints

当前实现把一条 prompt-response 样本构造成 multiplex attention graph，然后在**构图阶段**把边和一跳邻居结构聚合进每个 response token 的节点特征。之后的无监督 detector 只读取节点特征，不再读取邻接关系，也不预测下一个 token。

旧 HoloRoute masked reconstruction 和 P-Cut 已退役，结果保存在：

- [`iclr/HOLOROUTE_BASELINE_QA_RESULT.md`](iclr/HOLOROUTE_BASELINE_QA_RESULT.md)
- [`iclr/PCUT_SMOKE_RESULT.md`](iclr/PCUT_SMOKE_RESULT.md)

## 代码入口

```text
graph.py       sparse attention -> multiplex token graph
features.py    edges + one-hop neighbour provenance -> node features
detection.py   node features -> train-only normal subspace -> anomaly score
pipeline.py    fit, score and export per-sample graph data
evaluate.py    scores frozen 后读取标签
run.py         命令行入口
```

## 节点特征

对每个 response token、layer 和 head，构造三组固定基展开：

```text
direct prompt distribution
response-history lag distribution
one-hop inherited prompt distribution
```

再加入 diagonal 和 unresolved mass。所有层均保留；heads 通过固定正交投影压缩，而不是简单求平均。最终每个 token 得到一个固定维度结构向量：

```python
graph = build_graph(sample, config.graph)
features = build_node_features(graph, config.feature)

features.token_layer   # [response, layer, feature]
features.node          # [response, node_feature]
```

无监督检测只在 `features.node` 上拟合位置条件化的 robust PCA normal subspace，主分数是节点特征到该子空间的残差能量。

## 一键运行 QA

```bash
bash experiments/holoroute/run_qa.sh
```

烟测：

```bash
TRAIN_LIMIT=30 TEST_LIMIT=10 \
bash experiments/holoroute/run_qa.sh
```

输出目录：

```text
experiments/holoroute/outputs/routing_fingerprint_qa/
├── method.pt
├── reference.npz
├── scores.npz
├── graphs/<sample_id>.npz
└── evaluation/
```

每个 `graphs/*.npz` 保存 sparse edges、`token_layer_feature` 和最终 `node_feature`，可以直接用于其他节点无监督检测器。
