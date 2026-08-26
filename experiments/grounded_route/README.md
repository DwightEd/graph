# GroundedRoute

GroundedRoute 把一条 prompt-response 样本构造成带 layer/head 类型的因果 token 图，并把边与邻居信息聚合到每个 token 的 `node_embedding` 中。后续检测器只读取节点表征，不再读取邻接关系。

代码规范见 [`iclr/ENGINEERING_RULES.md`](iclr/ENGINEERING_RULES.md)。

## 图

```text
node       prompt 或 response token
edge       source token -> response token
edge type  (layer, head)
edge value retained attention weight
```

稀疏 cache 中未保存的边不会被当作零。每个 response row 保存 retained edge mass、self diagonal mass 和 unresolved mass，三者归一化后总和为 1。

## 节点聚合

每条边先把 source state、layer/head、source role、lag 和 lineage 编码为 message。attention weight 不再被重复编码，而是直接作为加权矩和总质量。

对同一个 `(target, layer, head)`，聚合器分别计算 prompt-source 与 response-source 的：

```text
weighted mean      邻居内容的中心
weighted spread    邻居内容是否冲突或分散
total mass         当前路由对该类 source 的依赖量
```

随后加入 diagonal self message 和 unresolved message，在 heads 间做 target-conditioned pooling，并用 GRU 更新 token state。全部 Transformer layers 处理完后得到：

```python
output.node_embedding      # [all_tokens, hidden_dim]
output.response_embedding  # [response_tokens, hidden_dim]
```

核心代码：

```text
graph.py         attention cache -> TokenGraph
lineage.py       prompt / response-closed / unresolved 路径来源
aggregation.py   role-aware weighted moments
model.py         多层 token 状态更新
learning.py      无标签 endpoint prediction objective
pipeline.py      build / fit / encode / detect
evaluation/      node-only detector、probe 与构图控制
```

## 自监督训练

训练任务是从当前 token 之前的因果 prefix 区分真实 source endpoint 与 role/lag 匹配的 causal non-edge。它只用于学习节点表征，不是最终异常分数。

## 一键运行 QA

```bash
bash experiments/grounded_route/run_qa.sh
```

输出：

```text
experiments/grounded_route/outputs/qa/
├── model.pt
├── calibration/index.npz
├── calibration/graphs/*.pt
├── test/index.npz
├── test/graphs/*.pt
├── detector.npz
├── scores.npz
└── evaluation.json
```

## 评估节点表征

只使用已有 real embedding：

```bash
bash experiments/grounded_route/evaluation/run_qa.sh
```

完整构图控制：

```bash
bash experiments/grounded_route/evaluation/run_controls_qa.sh
```

控制含义：

```text
no_message       不读取 source node state
endpoint_rewire  保留 role/layer/head/coarse lag，替换 exact source
weight_shuffle   保留 endpoints 和 row mass，打乱 weight-endpoint pairing
```

无监督 detector 与监督 readability probe 都只读取 `node_embedding`。监督 probe 只用于判断表征中是否存在可读信号，不属于主方法。
