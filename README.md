# Internal Routing Rhythm Audit

当前主线位于 `experiments/reanchor_flow/`。它不再用标点预设重锚定边界，而是对每个
response token 自动计算：

- 整体 source route 是否突然变化；
- 是否重新读取远处 prompt / RAG evidence；
- 这次回看是少数 source 的选择性重锚定，还是较广的 prompt 审查；
- 当前 token 生成后是否成为未来 token 反复读取的高影响 anchor。

Revisit peak 与 anchor peak 独立发现，再比较短时滞耦合和 circular-shift null。标签只在
捕获结束后用于比较 hallucination onset 与同一回答中的匹配 clean token。默认只做一次
前向，不做删证据或多分支反事实。

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

输出只有关键统计、每任务一张 population rhythm 图，以及默认一条样本的直观路由图。
详见 `experiments/reanchor_flow/README.md`。
