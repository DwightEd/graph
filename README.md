# GroundedRoute

GroundedRoute 是当前主实验。每条 prompt-response 样本构造成一张 typed attention graph，图编码器输出每个 token 的 `node_embedding`，后续检测器只读取节点表征。

```text
sparse attention
-> token graph
-> neighbour aggregation
-> frozen node embeddings
-> node-only detection
```

节点是 token。每条 retained attention 保留 source、target、layer、head 和 weight。layer/head 不在消息传递前平均，未保存边作为 unresolved mass。

当前聚合器分别计算 prompt-source 与 response-source 邻居 message 的 weighted mean、weighted spread 和 total mass，再融合 diagonal、unresolved 和 head identity。

## 目录

```text
research_dataset.py                统一数据接口
experiments/grounded_route/        当前 token-graph 表征方法
  aggregation.py                   邻居加权矩
  lineage.py                       路径来源
  evaluation/                      node-only 评估与构图控制
  iclr/ENGINEERING_RULES.md        项目代码规则
experiments/dbgnn_reference/       DBGNN / GCN 参考实现
experiments/holoroute/             历史基线
docs/EXPERIMENT_HISTORY.md         历史实验记录
```

## 运行

```bash
bash experiments/grounded_route/run_qa.sh
bash experiments/grounded_route/evaluation/run_qa.sh
bash experiments/grounded_route/evaluation/run_controls_qa.sh
```

四种构造分别独立训练和编码：real、no_message、endpoint_rewire、weight_shuffle。所有下游 detector 与 probe 只读取 `node_embedding`。

## 测试

```bash
python -m compileall -q experiments/grounded_route
bash -n experiments/grounded_route/run.sh
bash -n experiments/grounded_route/evaluation/run.sh
pytest -q experiments/grounded_route/tests experiments/grounded_route/evaluation/tests
```
