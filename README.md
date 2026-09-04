# Re-Anchor Phenomenon Audit

当前主线位于 `experiments/reanchor_flow/`。目标不是先调一个幻觉检测分数，而是依次检验：

1. 正常生成是否从 prompt evidence 逐渐转向 response history；
2. 非幻觉内容边界是否相对同句伪边界重新读取 evidence；
3. 真实 hallucination span 的首 token 是否出现更弱的重锚定。

旧版本的 `source token -> predicted token` 压平图已经退出主方法。预测 token `p` 的
logit 来自 query state `q=p-1`，但该 state 不会在 teacher forcing 中写入 `p` 的
hidden state；将 `p` 继续作为下一跳 carrier 会形成不存在的计算路径。跨 layer
平均后再做 path rollout 也会拼接出违反层序的路径。

v3 保存每个 target token、每层的 evidence / other-prompt / history attention
share、对应的 `A * ||W_O V||` message-magnitude share，以及两者的可见源 null。
主变量是 `log(observed/null)`，因此 history token 数量机械增长不能伪造 H1；
`evidence lift - other-prompt lift` 进一步排除了泛化 prompt 回看。H3 只用错误
token 尚未进入上下文的 offset 0，并将 exact-boundary、near 与 late onset 分开。

## 一键运行

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main
conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh --smoke
```

先跑 20 个样本/任务的现象 pilot：

```bash
conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh \
    --limit 20 \
    --query-chunk 64 \
    --output experiments/reanchor_flow/outputs/pilot20_v3
```

`--query-chunk` 现在作用于 attention score/softmax 本身，真实降低显存峰值。GPU
被共享时可改成 `--query-chunk 32`。

只有当 H1-H3 值得继续时，再加入两次证据干预：

```bash
conda run --no-capture-output -n research \
  bash experiments/reanchor_flow/run_all.sh \
    --limit 20 \
    --query-chunk 64 \
    --causal-cuts \
    --output experiments/reanchor_flow/outputs/causal_pilot
```

新默认结果目录为 `experiments/reanchor_flow/outputs/<model>/phenomenon_v3/`。
availability null 是新的捕获量，旧 NPZ 无法可靠补算，因此 v1/v2 会被明确拒绝，
不会静默混用。

## 模块责任

| 路径 | 责任 |
|---|---|
| `experiments/common/llama_message_intervention.py` | Llama 前向、真正的 query chunk、message gate 和 rerun |
| `experiments/reanchor_flow/routes.py` | 逐层、逐事件、逐 source-role 的消息统计 |
| `experiments/reanchor_flow/claims.py` | label-free sentence-like boundary proxy |
| `experiments/reanchor_flow/events.py` | 事件窗口、幻觉起点与同响应匹配对照 |
| `experiments/reanchor_flow/capture.py` | 单样本 baseline 与可选 evidence cuts |
| `experiments/reanchor_flow/analyze.py` | 流式遍历、逐样本落盘和显存释放 |
| `experiments/reanchor_flow/signals.py` | artifact 一致性校验与 availability-adjusted 信号 |
| `experiments/reanchor_flow/hypotheses.py` | H1-H3 对照、scope gate 与统计报告 |
| `experiments/reanchor_flow/evaluate.py` | 流式加载 NPZ、标签开启与报告写盘 |
| `experiments/reanchor_flow/metrics.py` | source-cluster 统计 |
| `experiments/reanchor_flow/visualize.py` | layer/event 曲线与置信带 |
| `experiments/reanchor_flow/run.py` | CLI |

## 验证

```bash
python -m pytest -q experiments/reanchor_flow/tests \
  experiments/common/tests/test_llama_message_intervention.py
python -m compileall -q experiments/common experiments/reanchor_flow
bash -n experiments/reanchor_flow/run_all.sh
```

详见 `experiments/reanchor_flow/METHOD.md`。若 generator 与 observer 不同，报告只属于
teacher-forced observer 对固定答案的处理机制，不能声称恢复了原生成模型的电路。
