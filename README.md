# Re-Anchor Mechanism Audit

当前主线位于 `experiments/reanchor_flow/`，以每个 response token 的因果预测位置
`q=p-1` 为单位，依次检验：

1. 正确回答中，校正可见 source 数量与 `||W_OV||` 容量后，直接路由是否从 prompt
   转向 response history；
2. 模型自行发现的 route-transition 位置是否重新接入 prompt/RAG evidence，并随后形成
   future anchor；
3. hallucination onset 是否出现更弱的 evidence entry 或更强的错误复用；
4. evidence-conditioned state 是否进入、与 question/instruction group 联合控制、跨层保留，
   以及最终是否对 target-versus-runner readout 可见。

浅层现象只需一次前向。`--mechanism-limit N` 对每个任务前 N 条样本追加四组
post-softmax、pre-Value-sum source cuts，并生成逐层机制图。标点只作为可视化参考，
不用于定义内部事件。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main

conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh \
    --query-chunk 64 \
    --mechanism-limit 30 \
    --output experiments/reanchor_flow/outputs/mechanism_v6_30
```

详见 `experiments/reanchor_flow/README.md` 和 `experiments/reanchor_flow/METHOD.md`。
