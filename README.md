# Re-Anchor Mechanism Audit

当前主方法位于 `experiments/reanchor_flow/`，采用 hub-aware、head-resolved 机制审计：

1. passage、sentence、field 或 evidence span 构成 source-unit candidates；
2. 在保留 layer/head/source/target 的 unrolled graph 上允许 `evidence → hub → target`；
3. `HeadResolvedRouteModel` 分开记录 provenance routing、target action、source-cut
   integration 和 exact intervention 四账，全程不平均 heads；
4. 可选择 raw attention 或 target-specific true-message backend；
5. 以固定 `z_q(a)-z_q(b)` 对 root、carrier 和 corridor 做 exact cut/patch/block；
6. 同一条 clean message 删除后必须能原位补回，否则该因果样本无效。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main

conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run subset \
    --split test --task all \
    --samples-per-task 1 --targets-per-sample 1 \
    --flow-signal message --carrier-scope all \
    --query-chunk 8 \
    --output experiments/reanchor_flow/outputs/native_mechanism_v2 \
    --plot
```

结果写入 `experiments/reanchor_flow/outputs/native_mechanism_v2/test/`。`carrier_scope` 默认是
`all`，以便跨层追踪 prompt hub；缩减 scope 时，未建模的 prompt destination 会进入
`unobserved` provenance，而不会沿用其初始 source 身份。

四联图显示 connected root→target backbone、逐 head selected-root/response-origin action、
residual-attention-MLP integration 和精确 cut/patch/block。高一致性只表示 functional alignment；
它本身既不证明 correctness，也不
证明 hallucination attractor。native source cut 识别的是 observed target 对指定 Value-message
operator 的依赖；grounded factual-effect 结论仍需 aligned clean/counterfactual pair。

详见：

- `experiments/reanchor_flow/METHOD.md`
- `experiments/reanchor_flow/SCHEMA.md`
- `experiments/reanchor_flow/MECHANISM_AUDIT.md`
