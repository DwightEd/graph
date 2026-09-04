# Internal Routing Rhythm Audit

当前主线位于 `experiments/reanchor_flow/`。它不使用标点预设重锚定边界，而是对每个
response token 自动计算：

- 整体 source route 是否突然变化；
- 分给全部 prompt token 的消息份额是否相对近期回升；
- 消息是否连续地从邻近上下文转向更非局部的 source；
- 当前 token 生成后是否成为未来 token 反复读取的 anchor。

Prompt-revisit、nonlocal-review 与 anchor peaks 独立发现，再比较短时滞耦合和
circular-shift null。Nonlocal review 使用连续距离权重，不设置“必须距离至少多少 token”
的硬门槛。报告同时给 pooled coupling 与按 source 计算的 sample-lift 置信区间，用来判断
现象是否在多数样本中重复，而不是由少数长回答驱动。

## 运行

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main
conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh --smoke --query-chunk 32
```

完整 test split：

```bash
conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh --query-chunk 64
```

输出包括每任务的关键总体统计、population rhythm 图，以及默认一条样本的路由图。详见
`experiments/reanchor_flow/README.md`。
