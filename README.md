# Constraint Routing Rhythm for Hallucination Analysis

当前正式研究实现位于 `experiments/constraint_routing_rhythm/`。它不再把
attention graph 本身当作创新点，而是按下面的证据链研究 evidence constraint
是否真正控制输出：

```text
functional local/global route visualization
                    ↓
evidence uptake → response carrier → later delivery
                    ↓
post-softmax evidence Value-message deletion + downstream rerun
                    ↓
ConstraintDeficit = cut margin - baseline margin
```

baseline 使用逐 query-head、GQA 对齐的
`A * ||W_O[head] V[kv(head), source]||`。它产生两个描述量：窗口化
`FunctionalReach` 与严格层序的 evidence-conditioned `RelayCapacity` 两跳瓶颈。
二者只用于可视化和提出 carrier；唯一主检测量是全 evidence-source cut 后固定
target-versus-runner margin 的有符号变化。没有 GNN、代理模型、最小电路搜索、
artifact 身份摘要链或多特征分类器。

carrier 只由 `RelayCapacity` 提出，`FunctionalReach` 不参与筛选；两者是否真的
形成 preplan–anchor 相位关系必须用 held-out circular-shift null 检验。若要进一步
声称“约束整合”，还必须增加 support/conflict evidence polarity swap，而不能只凭
attention 或一次删除干预命名机制。

完整方法、近邻工作边界与停止条件见
[`experiments/constraint_routing_rhythm/METHOD.md`](experiments/constraint_routing_rhythm/METHOD.md)。

## 一键运行

先跑每任务一例的 smoke：

```bash
bash experiments/constraint_routing_rhythm/run_all.sh --smoke
```

建议先跑 20×3 个样本的 pilot：

```bash
bash experiments/constraint_routing_rhythm/run_all.sh \
  --limit 20 --audit-limit 2 --plot-limit 2 \
  --output experiments/constraint_routing_rhythm/outputs/pilot
```

正式运行：

```bash
bash experiments/constraint_routing_rhythm/run_all.sh \
  --output /path/to/output
```

脚本已有当前机器的模型、attention cache 与 RAGTruth `source_info.jsonl` 默认
路径；其他机器可用 `--model`、`--cache`、`--source-info` 覆盖。主路径逐样本、
逐分支顺序运行，每个样本约两个完整前向；`--audit-limit 0` 可关闭小子集的额外
机制诊断。

GPU 上若已有其他进程占用数 GB 显存，8B 模型仍会 OOM。先运行：

```bash
nvidia-smi
```

确认目标卡基本空闲后再执行。allocator 设置只能缓解碎片，不能释放其他进程的
显存。

## 代码结构

| 路径 | 责任 |
|---|---|
| `research_dataset.py` | 统一数据与 evaluation-only labels 边界 |
| `experiments/constraint_routing_rhythm/routes.py` | 流式功能消息路由 |
| `experiments/constraint_routing_rhythm/rhythm.py` | local/global 与 evidence-carrier 图 |
| `experiments/constraint_routing_rhythm/intervene.py` | 真实 Value-message gate 与重前向 |
| `experiments/constraint_routing_rhythm/capture.py` | 单样本主干预及 audit |
| `experiments/constraint_routing_rhythm/analyze.py` | label-free 遍历、释放、保存、画图 |
| `experiments/constraint_routing_rhythm/evaluate.py` | 最后才读取标签并分任务评价 |
| `experiments/constraint_routing_rhythm/run.py` | `analyze/evaluate/all` 三个入口 |

GroundedRoute、Information Flow、directed route hypergraph 等目录只作为既有基线或
历史实验，不属于这套实现。

## Claim re-anchor flow 发现实验

`experiments/reanchor_flow/` 使用同一次冻结模型前向同时构造精确
`A||W_OV||` 图与 attention-only 对照，将 query 行转换成
`source token -> predicted token` 的因果 DAG，再用目标条件化全局路径势能研究：

```text
正常 prompt -> response-history 漂移
             ↓
claim 边界 evidence re-read
             ↓
evidence-seeded flow 是否经 boundary anchor 到达 claim sink
```

它还比较 direct、edge-bag、attention-only、middle-layer 与 role/lag-preserving
rewire，并在 label-blind 小子集上真实删除 functional-flow backbone、attention
backbone、capacity bag 和 matched endpoint 后重跑模型，以检验图连接是否比边属性
本身更重要。这是现象发现和图必要性实验，不替代 `ConstraintDeficit` 的已有结果。

运行：

```bash
bash experiments/reanchor_flow/run_all.sh --smoke

bash experiments/reanchor_flow/run_all.sh \
  --limit 20 --audit-limit 3 --plot-limit 3 \
  --output experiments/reanchor_flow/outputs/pilot
```

## 测试

```bash
python -m pytest -q experiments/constraint_routing_rhythm/tests
bash -n experiments/constraint_routing_rhythm/run_all.sh

python -m pytest -q experiments/reanchor_flow/tests
bash -n experiments/reanchor_flow/run_all.sh
```

合成测试只验证实现闭合，不等于真实机制结论。当前尚未记录正式 RAGTruth
`ConstraintDeficit` 或 re-anchor-flow 结果；主分数、功能节律、全局闭合和路径删边
必要性必须由真实运行及预注册对照决定。
