# Layer-Ordered Attention Graphs for Hallucination Detection

当前 active experiment 是 `experiments/directed_route_hypergraph/`。它把每个 prompt-response 样本构造成保留 layer、head、source、target 和 unresolved mass 的有向 attention-row hypergraph，并用三个无标签目标训练 64D token encoder：

```text
clean sparse attention
├─ local typed rows
├─ ordered P/R/U provenance
└─ ordered exact-endpoint layout
            |
corrupted graph -> neural hypergraph encoder -> frozen 64D token embeddings
                                             -> PCA-kNN -> frozen scores
                                             -> post-hoc labels only
```

新增的 endpoint layout 使用真实 Transformer 层序组合 sparse attention transitions，并将 cache 未解析质量保守地送入 absorbing sink。它受 Information Flow 的矩阵路径组合启发，但不含 hidden state、`W_V/W_O` 或真实 residual contribution，因此只是 attention-transport proxy，不是论文算法复现或因果贡献估计。

GroundedRoute 是稳定的 typed token-graph baseline；Information Flow 目录是冻结 GCN 上的 deterministic transport control；HoloRoute、P-Cut 和其他旧路线只保留为历史与对照。旧 P-Cut 的 full-QA closure AUROC 为 `0.4209`，本次优化没有重新启用 route cuts 或翻转其方向。

## 目录

```text
research_dataset.py                     unified data interface
experiments/directed_route_hypergraph/  current ordered-layout experiment
experiments/grounded_route/             typed token-graph baseline and evaluator
experiments/information_flow/           deterministic attention-transport control
experiments/dbgnn_reference/            DBGNN / GCN reference
experiments/holoroute/                   historical baselines and failed P-Cut record
docs/RESEARCH_STATUS.md                 current claims, gates and next experiment
docs/INFORMATION_FLOW_METHOD_AUDIT.md   paper audit and design boundary
docs/EXPERIMENT_HISTORY.md              durable experiment history
```

## 运行

```bash
bash experiments/directed_route_hypergraph/run_qa.sh
```

关键无标签目标消融：

```text
local only        FLOW_WEIGHT=0 LAYOUT_WEIGHT=0
local + P/R/U     FLOW_WEIGHT=0.5 LAYOUT_WEIGHT=0
local + endpoint  FLOW_WEIGHT=0 LAYOUT_WEIGHT=0.25
all objectives    FLOW_WEIGHT=0.5 LAYOUT_WEIGHT=0.25
reverse target    LAYOUT_ORDER=reverse
```

基线与结构控制仍可独立运行：

```bash
bash experiments/grounded_route/run_qa.sh
bash experiments/grounded_route/evaluation/run_controls_qa.sh
bash experiments/information_flow/run_qa.sh
```

## 测试

```bash
python -m compileall -q \
  experiments/grounded_route \
  experiments/directed_route_hypergraph \
  experiments/information_flow

bash -n experiments/directed_route_hypergraph/run.sh
bash -n experiments/directed_route_hypergraph/run_qa.sh

python -m pytest -q \
  experiments/grounded_route/tests \
  experiments/grounded_route/evaluation/tests \
  experiments/directed_route_hypergraph/tests \
  experiments/information_flow/tests \
  experiments/holoroute/tests
```

当前仓库没有记录 ordered-layout 的正式 RAGTruth 结果。synthetic tests 只验证质量守恒、层序、因果前缀、exact endpoint、梯度和 artifact 边界，不能替代真实任务评估。
