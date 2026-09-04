# Constraint Routing Rhythm

本实验研究一个简单、可直接重跑验证的问题：删除所有 evidence-source Value
messages 后，冻结模型对已生成 token 的固定 target-versus-runner margin 如何
变化。

唯一主检测分数是：

```text
ConstraintDeficit = cut_margin - baseline_margin
```

分数越高，表示 evidence transport 对当前输出的支持越弱或更像 veto。它测量指定
post-softmax intervention 下的 constraint sensitivity，不等于无条件 factuality。

一次 baseline 前向还会用精确
`A * ||W_O[query_head] V[gqa_kv_head, source]||` 构造功能路由图，并收敛为
两个可视化描述量：

- `FunctionalReach`：通常局部的 heads 在何处越过窗口回看；
- `RelayCapacity`：`evidence → response carrier → future query` 严格前后层序
  两跳路径的 normalized bottleneck，并以 absolute `relay_mass` 作数值质量门。

这两个量只解释或提出 relay 候选，不进入检测分数。图上的 route availability 与
真实 output control 会并排显示，不能合成第二 detector。U/D/UD 四分支只在读取标签前
固定的小子集上验证严格 layer-ordered carrier，同样不进入分数。本包不使用
GNN、surrogate、minimum-circuit search、ICG 或监督 feature combiner。

`FunctionalReach` 不参与 carrier 筛选，避免把两条曲线的节律关系预先写进规则；
没有完整未来窗口的末尾 token 记为未观测。正式的 rhythm 主张还需在 held-out
sources 上通过 circular-shift 时滞零分布，constraint-integration 主张还需通过
support/conflict evidence polarity swap。当前代码先实现最小可证伪主干与 U/D audit。

完整定义、相关工作边界、停止条件与 teacher-forcing 限制见
[METHOD.md](METHOD.md)。

## 一键运行

先从仓库根目录跑一条每任务一例、每例最多 8 个 response token 的 smoke：

```bash
bash experiments/constraint_routing_rhythm/run_all.sh --smoke
```

建议随后跑一个可控 pilot，而不是直接把整个 test split 全开：

```bash
bash experiments/constraint_routing_rhythm/run_all.sh \
  --limit 20 --audit-limit 2 --plot-limit 2 \
  --output experiments/constraint_routing_rhythm/outputs/pilot
```

smoke 默认写入独立的 `outputs/<model>/smoke/`，不会让截断结果污染正式目录。
正式运行整个 test split：

```bash
bash experiments/constraint_routing_rhythm/run_all.sh \
  --output /path/to/output
```

若本机路径不同，直接覆盖三个输入：

```bash
bash experiments/constraint_routing_rhythm/run_all.sh \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --cache /path/to/attention_cache \
  --source-info /path/to/RAGTruth/source_info.jsonl \
  --output /path/to/output
```

只冻结无标签结果或只做后验评价：

```bash
python -m experiments.constraint_routing_rhythm.run analyze --limit 10
python -m experiments.constraint_routing_rhythm.run evaluate
```

`--limit` 是每任务样本数；`--audit-limit 0` 可关闭额外 U/D 与 direct-response
诊断。`--max-events` 会截断 response，只能用于 smoke。它缩短 response 部分，
但不会缩短长 prompt，所以不能保证单样本峰值显存下降。

`--mass-floor` 是 carrier 候选的 absolute message 数值下限，默认 `1e-6`；它和
其它 proposal 参数都会写入 manifest，必须在读取标签前固定。

主路径每样本只有一次 baseline 和一次 evidence cut，即约两个完整前向。只有
显式设置 `--audit-limit N` 时，才会在每任务 label-blind、source-disjoint 的固定
样本上额外运行 direct-response 与可行的 matched non-evidence control；若确实找到
carrier，再顺序运行 U、D、U∪D 三个分支。
最重的 audit 样本共 7 次前向，但分支从不沿 batch 维并行。

若 GPU 上已有其他进程，仅靠 allocator 设置无法容纳 8B 模型；运行前应先用
`nvidia-smi` 确认卡基本空闲。脚本已设置
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 以降低碎片影响。
当前 attention 仍是 dense eager，峰值随 `heads × sequence_length²` 增长；
`--query-chunk` 只降低功能路由归约的临时量，不能降低原生 attention 峰值。
实现固定在 `transformers>=4.57,<4.58` 的 Llama full-sequence、无 padding/KV-cache
路径，首次样本会用最多 8 个 token 自动验证原生 eager 与全一门后端闭合。

## 模块责任

| 模块 | 唯一责任 |
|---|---|
| `data.py` | 读取 label-free tokens、response boundary 与 evidence mask |
| `routes.py` | 当前层内计算并流式归约精确 `A||W_OV||` |
| `capture.py` | 编排一个样本的 baseline、主 cut 与小子集 audit |
| `rhythm.py` | local/global、uptake、delivery 图和 U/D 候选 |
| `intervene.py` | 全 evidence-source cut 与 U/D 四分支真实重前向 |
| `analyze.py` | label-free 数据遍历、逐样本释放、保存与画图 |
| `artifacts.py` | 每样本轻量结果保存与恢复 |
| `evaluate.py` | artifacts 冻结后唯一允许读取 hallucination labels 的阶段 |
| `run.py` | 单一 CLI 编排，不复制核心算法 |
| `run_all.sh` | 唯一公开的一键入口 |

四维 attention/message tensor 从不跨 layer 保存；二维 route map 只活到当前样本
PNG 写完。baseline 默认只缓存 layer 0；启用 U/D audit 时再缓存 early/late split，
而不是缓存全部层。NPZ 不含 dense route map、标签、身份摘要链或 learned embedding。
输出为：

```text
<output>/results/<Task>/<sample>.npz
<output>/figures/<Task>/<sample>.png
<output>/reports/<task>/report.json
<output>/run_manifest.json
```

`run_manifest.json` 只是可读配置和完整样本清单，不包含文件摘要链。配置、模型或
cache 不一致时必须换输出目录；中断的 run 在补齐前不能评价，避免 smoke、OOM 残片
或不同模型被静默合并。

后验报告除主分数 AUROC/AP 与位置、长度、置信度控制外，还单列
`route_control_dissociation`：高 `RelayCapacity` 区间里主分数是否仍有效，以及
route proposal 与 signed evidence support 是否真正耦合。它不产生复合分数。

## 验证

```bash
python -m pytest -q experiments/constraint_routing_rhythm/tests
python -m ruff check experiments/constraint_routing_rhythm
bash -n experiments/constraint_routing_rhythm/run_all.sh
```

合成测试覆盖 GQA、`W_O` head block、路由归一化、`q=p-1`、no-renorm gate、
suffix 重算、标签防火墙、逐样本释放、smoke 前缀评价与 PNG 关闭。真实数据结果
尚未运行，因此当前是完整的可执行研究实现，不是已经成立的论文结论。

## 运行成功的最低含义

端到端命令完成只说明 artifacts 已生成。正式机制结论还要求：

- 全一 gate 与原生 logits 闭合；
- `query_position = prediction_position - 1`；
- GQA 与 `W_O` query-head block 对齐；
- evidence cut 不重归一化且下游确实重算；
- `ConstraintDeficit` 在标签打开前冻结；
- functional maps 优于 attention-only / `A||V||` 对照；
- U/D/UD 在 label-blind 子集和 matched carrier 对照上成立；
- QA、summarization、data-to-text 与不同模型分别报告。

任一项未完成时，应在报告中缩小 claim，而不是增加新 detector。
