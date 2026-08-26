# Node-embedding evaluation

这个目录只评估 GroundedRoute 已经保存的 `node_embedding`。它不读取边，也不再做消息传递。

## 模块

```text
data.py       读取并对齐 real / control 节点表征
models.py     小型 node-only 神经模型
detectors.py 无标签 detector 集合
probes.py    source-disjoint 监督可读性上限
metrics.py   AUROC、AUPRC 和 source bootstrap
controls.py  real 与三种构图控制的成对差异
run.py       一次完成评分和报告
```

## 单一 real 表征

```bash
bash experiments/grounded_route/evaluation/run_qa.sh
```

它比较 PCA-kNN、Isolation Forest、LOF、Autoencoder 和 Deep SVDD，并运行只作为诊断的 linear / MLP supervised probes。

## 完整构图控制

```bash
bash experiments/grounded_route/evaluation/run_controls_qa.sh
```

四条管线分别训练和编码：real、no_message、endpoint_rewire、weight_shuffle，然后使用完全相同的 node-only readers 比较。

主报告位于：

```text
experiments/grounded_route/outputs/qa_controls/evaluation/report.json
```
