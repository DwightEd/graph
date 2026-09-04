# Claim Re-Anchor Flow for Hallucination Analysis

当前现象发现主线位于 `experiments/reanchor_flow/`。它使用同一个冻结的
Llama observer，在一次 teacher-forced forward 中构造 attention-only 与
`A||W_OV||` token DAG，并研究新的事实 claim 是否重新接入 evidence：

```text
normal prompt -> response-history drift
                    ↓
claim-boundary evidence reread
                    ↓
evidence-seeded global flow through boundary anchors
                    ↓
claim sink
```

对每个 claim，代码计算 FlowTracer-style backward potential、全局
`evidence -> boundary -> sink` 路径质量，以及 direct、edge-bag、middle-layer、
attention-only 和 role/lag-preserving rewire 对照。小规模 label-blind audit 会在
同一模型中真实删除 graph-selected backbone，并与非连通强边和 matched endpoint
比较。该实验先验证图结构是否必要，不预设最终 hallucination detector。

`experiments/constraint_routing_rhythm/` 保留为 evidence-channel sensitivity 与
两跳 relay baseline。它的 `ConstraintDeficit` 不作为 re-anchor-flow 的主分数。

## 一键运行

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main
conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh --smoke
```

Pilot：

```bash
conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh \
    --limit 20 \
    --audit-limit 3 \
    --plot-limit 3 \
    --output experiments/reanchor_flow/outputs/pilot
```

完整 test split 运行时去掉 `--limit`。

## 模块责任

| 路径 | 唯一责任 |
|---|---|
| `research_dataset.py` | 统一数据入口与 evaluation-only labels 边界 |
| `experiments/common/ragtruth_alignment.py` | RAGTruth task/source/evidence token 对齐 |
| `experiments/common/llama_message_intervention.py` | registry-free Llama 前向、消息观察、post-softmax Value-message gate 与 suffix rerun |
| `experiments/reanchor_flow/claims.py` | label-free claim proxy |
| `experiments/reanchor_flow/routes.py` | 同一次前向归约 attention 与 `A||W_OV||` 路由 |
| `experiments/reanchor_flow/graph.py` | `source token -> predicted token` DAG 与结构对照 |
| `experiments/reanchor_flow/potential.py` | sink-conditioned path potential |
| `experiments/reanchor_flow/flow.py` | re-anchor flow 与 dominant backbone |
| `experiments/reanchor_flow/capture.py` | 单样本图视图和真实 path-cut 编排 |
| `experiments/reanchor_flow/analyze.py` | label-free 遍历、保存与样本图 |
| `experiments/reanchor_flow/evaluate.py` | 冻结 artifacts 后读取标签并评价 |
| `experiments/reanchor_flow/run.py` | CLI |

共享干预实现直接调用 Llama 层的 `q_proj/k_proj/v_proj/o_proj`、RMSNorm、residual
和 MLP，不再依赖 Hugging Face 不同版本中变化的 `AttentionInterface`、
`AttentionMaskInterface` 或 `ALL_ATTENTION_FUNCTIONS` 注册表。首次样本会用短前缀
验证手写完整前向与模型原生前向闭合。

## 测试

```bash
python -m pytest -q experiments/common/tests
python -m pytest -q experiments/reanchor_flow/tests
python -m pytest -q experiments/constraint_routing_rhythm/tests
python -m compileall -q experiments/common experiments/reanchor_flow \
  experiments/constraint_routing_rhythm
bash -n experiments/reanchor_flow/run_all.sh
```

合成测试只验证实现语义，不等于真实机制结论。正式结论仍取决于 full graph 是否
稳定优于 attention、direct、bag、rewire，并且 graph-selected 连通路径的真实删除
是否比同质量散边造成更大的输出变化。
