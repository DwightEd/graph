# Attention-conditioned graph state modeling

当前主线是 `trajectory_geometry/` 中的无标签图状态转移模型。它把压缩
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
