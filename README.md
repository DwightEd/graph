# Attention graph research

The new hypothesis-stage, fully modular typed-path method is in
`experiments/causal_typed_path_debruijn/`. It keeps every observed
`[layer, head]` attention row and emits a route state per channel, models the
generation-time route grammar with a fixed second-order De Bruijn process, and
couples recent rupture with response-closed lock-in. Cross-layer head transport
is explicitly an equal-head proxy because `W_O/W_V` are unavailable. See its
`METHOD.md`; no real-cache performance claim has been made from the
implementation-only smoke tests.

The active spectral method is in `experiments/spectral_feasibility/`. It keeps
signed RR causal-prefix modes independently for every layer/head, fits a robust
PCA on one set of unlabeled source groups, and calibrates anomaly tails on a
disjoint set. Run it with:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/spectral_feasibility/run.sh
```

Its mechanism audit is in `experiments/rr_topology_dynamics/`. The earlier
causal-topology encoder remains a research baseline; the reported true-graph
versus rewired/token-only comparisons do not establish a useful topology gain.

Frozen scores from these and other experiments can be compared under the same
`task_type`, token/response unit, response-position range, and positive
prevalence with `experiments/conditioned_benchmark/`. The benchmark aligns all
methods on identical token rows and never refits a detector after labels open:

```bash
bash experiments/conditioned_benchmark/run.sh \
  TEST_SPLIT OUTPUT_DIR rr_spectral=TEST_SCORES.npz
```

## Attention-only dynamic multiplex construction

`attention_multiplex/` 只研究压缩 attention 本身，将
layer 建模为有序深度、head 建模为关系类型，并联合分解
`(layer,response-query) × (head,prompt+response-source)` 稀疏矩阵。它不读取
hidden state 或标签，也不输出 AUROC、异常分数和 t-SNE。

所有新方法必须通过根目录 `research_dataset.py` 读取 attention；禁止在实验
子目录直接解析 `.pt/.npz`。中央接口能够逐 channel 恢复带 censoring 掩码的
`[R,N]` 因果矩阵：保留边使用真实值，合法未保留边默认填 cache floor
（正式缓存为 `0.01`），未来位置为 0。PP 非对角边从未保存，因此明确不采用，
不会伪造成完整 `[N,N]` 矩阵。

一键运行：

```bash
bash attention_multiplex/run_attention_multiplex.sh
```

详见 `attention_multiplex/README.md` 和 `attention_multiplex/METHOD.md`。

## Previous hidden-state experiment

此前的 `trajectory_geometry/` 无标签图状态转移实验把压缩
attention 当作有向算子，把逐层 hidden state 当作节点信号，并用真实邻接
预测下一层状态更新：

\[
\widehat{\Delta X_l}
=f_l\!\left(X_l,\{A_{l,h}X_l\}_{h=1}^{H},q_l\right),
\qquad
E_l=\Delta X_l-\widehat{\Delta X_l}.
\]

`f_l` 是 train-only trimmed ridge 的闭式解，不使用标签、不训练 GNN、
不反向传播。每个 response token 的主表征是跨层残差 `E_l` 的固定 DCT
系数，而不是一组手工图指标的平均值。

## 一键运行

hidden sidecar 根目录需要包含 `train/` 和 `test/`，并与 attention 文件按
sample id 对齐：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main && \
HIDDEN_ROOT=/实际的/hidden_state_cache \
bash trajectory_geometry/run_graph_state_model.sh
```

也可以把 hidden 根目录作为第一个位置参数：

```bash
bash trajectory_geometry/run_graph_state_model.sh \
  /实际的/hidden_state_cache \
  /实际的/output_directory
```

先跑小规模接口检查：

```bash
LIMIT_TRAIN=10 LIMIT_TEST=5 \
HIDDEN_ROOT=/实际的/hidden_state_cache \
bash trajectory_geometry/run_graph_state_model.sh
```

脚本会逐样本打印 projection、state-fit、train/test encode 进度，并把完整
stdout/stderr 保存到 `${OUTPUT_DIR}.log`。

## 输入

attention 直接读取已有的压缩 response-CSR，不转换、不恢复低于阈值的边：

```text
attention_diagonal       [layers, heads, tokens]
response_row_ptr
response_column_indices
response_values
attention_floor
```

hidden sidecar 接受 `.pt`、`.npz` 或 `.npy`：

```text
hidden_states            [layers + 1, tokens, hidden_dim]
token_ids                [tokens]                         optional
sample_id / response_id                                  optional
```

也支持 `[tokens,layers+1,hidden_dim]`，以及只有连续 block 输出的
`[layers,tokens,hidden_dim]`。只含 response 或只保存少数非连续层会直接报错，
因为那样无法计算 prompt-to-response 消息或忠实的逐层状态转移。

## 同一个模型中的三个严格视图

三种视图使用相同 hidden projection、控制变量、拟合器和样本级
fit/calibration 划分：

- `node_control`：目标 token 状态、位置以及最小 RP/RR/self/unresolved 控制；
- `true_graph`：在 `node_control` 上加入真实 `A_{l,h}X_l`；
- `rewired_graph`：保持 target、layer、head、边权、RP/RR 类型和粗距离桶，
  只因果重连 source 后重新计算消息。

训练报告的 `gate_passed` 只依据无标签 held-out sample 的状态预测误差：

```text
true_graph MSE 至少比 node_control 低 1%
true_graph MSE 至少比 rewired_graph 低 1%
```

两项改善还必须通过 held-out 完整样本的配对 bootstrap，95% 区间下界大于
0。它检验邻接是否能解释 hidden 更新，不是幻觉 AUROC。只有该门通过，才值得
对冻结的 token residual embeddings 进行后续无监督异常检测。

## 输出

```text
graph_state_model.npz
manifest.json
train/index.jsonl
train/state_<sample_id>.npz
test/index.jsonl
test/state_<sample_id>.npz
```

每个节点文件包含：

- `true_graph_embedding`：主节点表征；
- `node_control_embedding`：不使用 source 邻接身份的基线；
- `rewired_graph_embedding`：随机拓扑对照；
- 三种视图的逐层原始预测 MSE；
- `graph_gain = MSE(node_control)-MSE(true_graph)`；
- `rewire_gap = MSE(rewired_graph)-MSE(true_graph)`；
- 仅用于审计的五个 route controls。

所有文件都标记 `labels_included=false`。本阶段不输出 AUROC、t-SNE 或挑选的
样本图，因为它的唯一任务是构造和验证节点表征。

旧的 `run_token_representation.sh`、attention scalar statistics 和 Stage-A
route dynamics 暂时保留为复现实验基线，不再作为当前主模型入口；待真实
远端缓存完成对照复现后再安全删除。

## Label-free causal topology experiment

Run the current foreground experiment on the canonical Llama-3.1-8B split:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
bash run_token_representation.sh
```

`DATA_ROOT` defaults to
`/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b`.
The baseline runner creates a timestamped directory under `outputs/causal_topology/`,
prints progress in the foreground, and writes the same stream to
`${OUTPUT_DIR}.log`. Use the named runner directly; the historical Lookback
compatibility alias has been removed.

The encoder preserves the 32 x 32 layer-head channels while separating retained
attention marginals from source-sensitive topology. Missing CSR edges remain
censored rather than being assigned the threshold as an exact weight; the
undefined Lookback balance alone falls back to `attention_floor`. It represents prompt
source position with Fourier moments and response history with normalized
one-hop mean, absolute difference, variance, and two-hop messages. A
coarse-lag-bin RR rewire is the fixed topology null; it preserves target,
channel, weight, and log2 lag bin, but not source in-degree, exact lag, or
collisions. Independent atomic one-class references are fitted and calibrated
on source/sample-disjoint unlabeled train groups, then combined by a calibrated
hierarchy.

The fixed token scores are:

- `attention_marginals`, `retained_support`, and `balance_scale`;
- `prompt_topology`;
- `rr_one_hop_exact`, `rr_two_hop_exact`, and `rr_multihop_exact`;
- `rr_multihop_lag_rewired`;
- `causal_topology_exact` and `causal_topology_lag_rewired`;
- the primary score, `full_signal`.

The completed output directory contains four files:

```text
topology_one_class_model.npz
topology_label_free.npz
topology_label_free_report.json
topology_experiment_report.json
```

Only scalar scores, token metadata, and two compact score coordinates are
stored for test tokens. The experiment does not write the former population
1024-D `X` or 3072-D `[X,Fp,Fr]` matrices.

During train-reference construction,
`train_reference_checkpoint.npz` is updated every `CHECKPOINT_INTERVAL`
completed samples. Resume using the same output directory and unchanged
configuration:

```bash
OUTPUT_DIR=/path/from/the/interrupted/run bash run_token_representation.sh
```

The checkpoint signature covers the train inventory, encoder settings, and
atomic block contract. It is removed after a successful run.

Four paired sample-bootstrap comparisons are fixed before evaluation:

1. `full_signal` versus `attention_marginals`;
2. `causal_topology_exact` versus `attention_marginals`;
3. `rr_multihop_exact` versus `rr_multihop_lag_rewired`;
4. `rr_multihop_exact` versus `rr_one_hop_exact`.

"Unsupervised" describes representation learning and anomaly scoring: the
algorithm does not expose or consume labels while encoding, fitting references,
or freezing scores. The completed experiment still opens the evaluation labels
afterward to report AUROC/AUPRC and paired uncertainty. A canonical cache file
may physically contain an embedded `y_token` field when deserialized; the
scientific contract is that this field is not supplied to the algorithm, not
that its bytes can never enter process memory.

See [docs/method.md](docs/method.md) for the mathematical contract.
