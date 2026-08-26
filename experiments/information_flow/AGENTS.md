# Information Flow coding contract

修改本目录前先读取：

```text
experiments/grounded_route/iclr/ENGINEERING_RULES.md
```

本实验额外遵守以下边界：

- 原论文的 value/residual-aware contribution 与本目录的 attention-only proxy 必须明确区分；
- 不把未保存的 sparse attention 当作零；
- 不根据 test label 选择层序、方向、视图或 detector；
- `extract.py` 不读取标签；
- downstream reader 只读取固定的节点表征；
- 先验证 ordered all-layer flow，再考虑加入新的神经网络；
- 若 ordered flow 不优于 last-layer、layer-mean、reverse-order 和 identity 控制，应停止该 attention-only 假设，而不是增加手工分数。
