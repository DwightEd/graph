# Experiment plan

## 主实验

```text
Structural routing fingerprint + robust PCA residual
```

必须与以下方法在相同 token、source split 和评价协议下比较：

```text
absolute / relative position
simple prompt/history mass statistics
RR spectral residual
layer/head mean fingerprint
no inherited-prompt block
matched endpoint rewire
```

## 判定门槛

1. 主分数必须超过 absolute-position AUPRC；
2. `no inherited prompt` 退化，才说明邻居 provenance 有用；
3. `layer/head mean` 退化，才说明保留 channel structure 有用；
4. matched rewire 退化，才说明 exact endpoints 有用；
5. score-position Spearman 必须报告；
6. 结果按 source bootstrap 给出置信区间；
7. 同一节点特征至少用 PCA residual、kNN 或 LOF 做 detector sensitivity 对照。

该实现仍是图节点表示基线，不提前声称新的幻觉机制。
