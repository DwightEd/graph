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

完整 train/test 运行还会在不读取 train 标签的条件下校准 transport/adoption failure score，
先冻结 test token score，再打开 test 标签报告 token 与 hallucination-onset AUROC/AUPRC。

每个样本都执行一次 context cut，直接得到候选、采纳和分布影响；
`--mechanism-limit N` 只控制额外三组 source cuts 与逐层机制图。标点只作为匹配变量，
不用于定义内部事件。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main

conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh \
    --split all \
    --query-chunk 64 \
    --mechanism-limit 30 \
    --output experiments/reanchor_flow/outputs/reanchor_v8_all
```

详见 `experiments/reanchor_flow/README.md`、`experiments/reanchor_flow/METHOD.md` 和
`experiments/reanchor_flow/DETECTOR.md`。
