# Architecture

```text
attention cache
    -> graph.py: exact multiplex token graph
    -> features.py: deterministic structural aggregation
    -> node_feature [response_token, feature]
    -> detection.py: unsupervised node detector
    -> evaluate.py: label-posthoc metrics
```

模块边界：

- `graph.py` 只读取和规范化 sparse edges；
- `features.py` 把边分布和一跳邻居 provenance 固化到节点特征；
- `detection.py` 只接收节点特征和 nuisance conditions；
- `pipeline.py` 负责 train/test 数据流和图文件导出；
- `evaluate.py` 只在分数冻结后读取标签。

当前 active path 中没有神经消息传递、next-token prediction、masked graph reconstruction、P-Cut 或多 residual fusion。
