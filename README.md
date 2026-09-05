# Re-Anchor Mechanism Audit

当前主方法位于 `experiments/reanchor_flow/`，采用 Evidence-to-Target Causal Corridor
(ETCC) 审计：

1. passage、sentence、field 或 evidence span 构成 source-unit candidates；
2. 在保留 layer/head/source/target 的 unrolled graph 上计算 `C(u→t)` 和
   `T(v|u,t)`；
3. 可选择 raw attention 或 target-specific true-message backend；
4. 以固定 `z_q(a)-z_q(b)` 对 root、carrier 和 corridor 做 exact cut/patch/block；
5. 同一条 clean message 删除后必须能原位补回，否则该因果样本无效。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main

conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_corridor.sh \
    --pair /path/to/paired_world.npz \
    --flow-signal message \
    --query-chunk 8
```

旧 schema-v8 `analyze/evaluate/detect` 管线保留为 frozen baseline；最新较弱的 held-out
detection 结果不再被解释为证据重锚机制。

详见：

- `experiments/reanchor_flow/METHOD.md`
- `experiments/reanchor_flow/SCHEMA.md`
- `experiments/reanchor_flow/RESEARCH_PLAN.md`
