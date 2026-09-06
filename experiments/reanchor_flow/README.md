# 逐 head 的时间轴重锚定机制审计 v3

v3 直接审计这个生成机制：模型在位置 \(q\) 是否从最近少数 response token，切换为读取原始
prompt 或更早的 response relay；读到的具体位置又是否真正写入 residual，并对 \(q+1\) 的生成有用。

方法分成两层，不能混为一个分数：

1. **结构发现**：逐 layer、逐 head 扫描 response 时间轴。每个完整 causal source row 在任何 sparse
   edge 裁剪前，先聚合为四个互斥位置桶：`prompt_evidence`、`other_prompt`、
   `remote_response` 和 `recent_local`。用真实 post-\(W_O\) message transport 找出“上一位置 local
   占优、当前位置 long-range 占优”，并且同时出现 local 下降与 long-range 上升的切换点。
2. **接纳审计**：结构坐标冻结后，再附上具体 source token/unit、evidence lineage、signed
   grad-message action、residual/attention/MLP integration，以及可选的少量 exact confirmation。

整个流程保留 `(layer, head, source, destination)`；绝不先对 head 平均。attention 作为路由图的辅助
对照，结构选点使用 \(W_O(A V)\) message transport。程序也不常驻保存完整 \(L\times H\times T^2\)
下三角矩阵：完整 row 按 `query_chunk` 流式计算，只保留 \(L\times H\times P\times4\) 桶总量和每桶
最强 source，所以图上的事件三角视图是桶聚合/最强 source 轨迹，不是未经压缩的全 \(T^2\) tensor。

完整算法见 [METHOD.md](METHOD.md)，预注册假设与结论边界见
[MECHANISM_AUDIT.md](MECHANISM_AUDIT.md)，artifact 契约见 [SCHEMA.md](SCHEMA.md)。

## 为什么这次才是完整的重锚定审计入口

此前版本没有把这一机制闭环：早期版本画过 response 下三角路线变化和 prompt revisit，但混合或平均
了 heads，也没有把“local 下降 + long-range 上升 + 信息接纳”绑定在同一个事件；后来的版本加入了
真实 message、gradient、hub 与 intervention，却改成固定 target 的静态 route，丢掉了在完整生成时间轴
上先找切换点的步骤。

v3 现在补齐的是“全时间轴逐-head结构发现 → source 身份/lineage → 当前 token 的 action/integration
→ 可选因果确认”。这是一条可审计的测量链，不代表这个机制已经在数据上被验证，也不自动证明它能
检测幻觉。

## 代码组织

| 责任 | 模块 |
|---|---|
| 输入坐标与 source units | `units.py`、`worlds.py`、`native_world.py` |
| full-row / sparse message capture | `flow.py`、`native_flow.py`、`attribution.py`、`message_norm.py` |
| 结构事件与 route 分析 | `reanchor_timeline.py`、`route_model.py`、`throughput.py`、`route_plan.py` |
| 少量冻结干预 | `native.py`、`corridor.py`、`audit.py` |
| artifact 契约、写入与校验 | `artifact_schema.py`、`artifact_payload.py`、`artifact_validation.py` |
| 数据选择、运行、评价与作图 | `subset_data.py`、`subset.py`、`subset_report.py`、`mechanism_plot.py`、`run.py` |

`reanchor_timeline.py` 是唯一的时间事件定义；`route_plan.py` 只负责固定 target 的候选拓扑，
`route_model.py` 只负责 provenance/action/integration ledger。`audit.py` 与 `corridor.py` 并非旧版重复物：
它们保留 aligned clean/corrupt world 的双向 patch 路径，后续 fact×confidence matched audit 仍需要它。
artifact 的 schema、payload 和 validator 已分开，避免在运行流程里手工搬运一长串字段。

## 默认运行语义

- 默认 `--target-policy reanchor`：在当前模型的 clean full-row transport 中按最强单 head 的结构分数
  排序，经时间 NMS 后选择最多 \(N=\)`targets-per-sample` 个事件中心。
- `--target-policy reanchor-window`：\(N\) 是 target row 的硬总预算；先冻结最多
  \(\lceil N/3\rceil\) 个中心，再按 rank 补中心的 \(-1/+1\) 上下文。`N=3` 即最强事件的
  `[-1, 0, +1]` 窗口。完全没有事件时才 label-free 地回退到 evenly-spaced rows，并写入
  `contrast_origin`。
- 即使 \(N\) 已覆盖短 response 的全部 rows，仍执行完整事件扫描：真实 center 保留 event metadata，
  其余 rows 明确记为 non-event，而不是把整个短样本误判为无事件。
- target selector 重新计算当前模型的完整 causal rows，不使用外部稀疏 attention cache、label、
  gradient 或 intervention outcome。
- 对每个冻结 target 再构建 target-conditioned root/hub/corridor AuditPlan。plan 冻结后固定做一次
  selected-root cut 以生成 integration diagnostic；它不是 confirmation。
- 只有显式传入 `--confirm`，才对预算内冻结候选做 necessity/sufficiency/restore/block；不会逐 route
  消融，也不会用 exact outcome 重选失败候选。

一个 artifact 的时间轴覆盖该 target 的 teacher-forced prefix。只有 event 的 `position ==
query_position` 时，其 action 才能称为该事件对紧随 \(q+1\) token 的 immediate action；更早 event 的
action 只是“对当前 artifact 晚期 target 的 downstream action”。

## 一键运行：事件窗口机制审计

下面在 QA、Summary、Data2txt 各取一个样本，并给最强重锚定事件分配 3 个 target rows。它会生成
逐-head时间轴、事件三角视图和接纳诊断，不对所有路线逐条剪除：

~~~bash
PROJECT=/share/home/tm902089733300000/a903202310/lys/research/graph
MODEL=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
CACHE=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
SOURCE_INFO=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl
OUTPUT="$PROJECT/experiments/reanchor_flow/outputs/temporal_reanchor_v3"

cd "$PROJECT" &&
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run subset \
    --split test \
    --task all \
    --samples-per-task 1 \
    --targets-per-sample 3 \
    --target-policy reanchor-window \
    --flow-signal message \
    --carrier-scope response \
    --max-response-tokens 128 \
    --max-route-rows 256 \
    --edges-per-head 2 \
    --root-candidates 4 \
    --hub-candidates 8 \
    --corridor-edges 64 \
    --edge-coverage 0.90 \
    --query-chunk 8 \
    --local-window 10 \
    --model "$MODEL" \
    --cache "$CACHE" \
    --source-info "$SOURCE_INFO" \
    --output "$OUTPUT" \
    --plot
~~~

运行时有 tqdm 的 sample、target 及绘图/评价阶段进度；每个 target 完成后立即落盘，相同 scientific
config 可以恢复。旧的 `native_mechanism_v2` 或其他配置目录不能与本次 schema/config 混用，请使用
新的 output。

若进程被系统直接打印 `Killed`，通常是 OS/cgroup 的 OOM，不是 `corridor_ok=False` 导致的异常。
先保持 `carrier-scope=response` 和有限的 `max-response-tokens`，再降低 `query-chunk`；
`max-route-rows`、`edges-per-head` 与 `corridor-edges` 分别限制常驻 rows、稀疏边和确认子图。full-row
扫描仍会在 chunk 内临时计算 causal row，但只持久化四桶统计，不持久化完整 \(T^2\)。

## 一键运行：少量因果确认

确认模式仍先按同一 transport-only 规则冻结事件与 AuditPlan，随后只验证预算内 root/hub/corridor：

~~~bash
PROJECT=/share/home/tm902089733300000/a903202310/lys/research/graph
MODEL=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
CACHE=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
SOURCE_INFO=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl
OUTPUT="$PROJECT/experiments/reanchor_flow/outputs/temporal_reanchor_v3_confirm"

cd "$PROJECT" &&
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run subset \
    --split test --task all --samples-per-task 1 \
    --targets-per-sample 3 --target-policy reanchor-window \
    --flow-signal message --carrier-scope response \
    --max-response-tokens 128 --max-route-rows 256 \
    --edges-per-head 2 --root-candidates 4 --hub-candidates 8 \
    --corridor-edges 64 --edge-coverage 0.90 \
    --query-chunk 8 --local-window 10 \
    --model "$MODEL" --cache "$CACHE" --source-info "$SOURCE_INFO" \
    --output "$OUTPUT" --confirm --plot
~~~

`--confirm` 会增加有限的 forward reruns。确认失败是有效的 negative result；程序不会自动换 root、
扩大 corridor 或回头修改结构事件。

## 阶段 B：AUROC/AUPRC

全部 label-free capture 完成后，独立读取 hallucination labels：

~~~bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph &&
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run subset-evaluate \
    --split test \
    --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
    --output experiments/reanchor_flow/outputs/temporal_reanchor_v3
~~~

结果写入 `mechanism_evaluation.json`，三个不训练的 raw axes 分开报告，不能拼成一个事后挑选的分数：

| raw axis | 定义 | 预注册 hallucination-risk 方向 |
|---|---|---|
| `route_origin_competition` | query row 的 response-origin action 相对 all-evidence-origin action | 越高风险越高 |
| `temporal_switch_score` | 只在 event-center artifact 上，冻结 layer/head/query 的 transport-only switch score | 中性探索轴：同时报告 raw-higher 与 negated 方向 |
| `evidence_adoption` | 只在 `prompt_evidence` event center 上，同一 layer/head/query 的 full-row prompt-evidence bucket signed action | 越低风险越高 |

每个 axis 输出自己的 evaluated target 数、positives、prevalence、AUROC 和 AUPRC；中性 temporal
axis 另输出 `negated_auroc/negated_auprc`，并在每个 task group 下按
`prompt_evidence/other_prompt/remote_response` 输出 `temporal_switch_by_source_kind`；不能看完 test
labels 后挑方向或 source kind。窗口上下文与 fallback
不伪装成 event center，非 prompt-evidence center 也不进入 evidence-adoption 分母。只有一个 label
类别时指标为 `null`。route baseline 可评价所有冻结 targets；temporal/adoption 只把 artifact 自己的
\(q+1\) label 连接到合格的 query-matched event center。prefix 内更早 candidate 的 downstream action
不会继承这个 label。`evidence_adoption` 仍是对 observed token
contrast 的局部支持，不是事实正确性。

每任务 1 个样本只是端到端 smoke test，几乎不能判断检测效果。正式评价必须增加 held-out 样本，并
保持 test labels 不参与 target/event/root/hub/corridor、方向或阈值选择。还要注意 reanchor policy
本身按结构事件抽样：当前 AUROC/AUPRC 是 event-conditioned 关联，不是所有自然 response token 的
总体检测性能；报告会标记 `selection_is_not_population_evaluation=true`。要声称 population detector，
还需预注册并加入非事件对照/覆盖抽样，而不能只看被 selector 选中的中心。

## 关键预算

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--target-policy` | reanchor | 用 full-row true-message switch 选事件中心 |
| `--targets-per-sample` | 1 | target row 硬预算；事件窗口审计建议 3 |
| `--carrier-scope` | response | 只展开 response destinations；prompt 仍可作 source |
| `--max-response-tokens` | 128 | clean 时间轴 pilot 的 response horizon |
| `--max-route-rows` | 256 | 每个 target 的 represented destination rows 硬上限 |
| `--edges-per-head` | 2 | sparse capture 为 transport top-k ∪ absolute-functional top-k，最坏 2k；corridor 再限 k |
| `--root-candidates` | 4 | root 候选上限 |
| `--hub-candidates` | 8 | hub 候选上限 |
| `--corridor-edges` | 64 | frozen corridor 显式 message 上限 |
| `--query-chunk` | 8 | full-row 临时计算的 query chunk |
| `--local-window` | 10 | `recent_local` 的 response 距离上限，包含对角线 |

`prompt_evidence`、`other_prompt`、`remote_response`、`recent_local` 是按 source position/unit 定义的
互斥集合。`remote_response` 只是较早 response 候选；只有其保存的 evidence-lineage fraction 大于零，
才可进一步称 evidence-bearing relay candidate。这个 fraction 来自 coverage-pruned sparse route
provenance，预算外质量由 `route_row_total - route_row_retained` 现算为 unknown/unobserved 且不重归一；它不是 full-row exact lineage，更不是
语义真值。bucket kind 是输入位置/unit 身份，不代表该 layer 的 node state 是纯 evidence 或纯
other-prompt；经过更新后任何 winner 都可能混合 provenance。高 attention 只说明 gate，高 message transport 说明实际写入
强，高 signed action 说明对当前 contrast 的局部一致性；三者不能互换。

## 图怎么读

| 面板 | 内容 | 结论上限 |
|---|---|---|
| frozen route | selected-root backbone、corridor、候选 hub 与未观察质量 | 固定 target 的候选 topology |
| switch timeline | 所有 `(layer, head)` 独立轨道 × response predictor，显示 full-row 四桶 transport 切换 | local→long-range 结构事件；不是语义或因果结论 |
| event triangles | top 事件的 prompt 区、recent-local band、每桶最强 source/token/unit、rise/fall 与 action | 桶聚合后的具体读取候选；不是完整 \(T^2\) 矩阵 |
| integration | selected-root cut 下 residual/attention/MLP 与 head/layer coherence/action | post-selection 接纳诊断 |
| exact ladder | 少量 cut/restore/block；未请求时为 not run | 冻结计划的 operator-specific 验证 |

高 head agreement 只表示当前 contrast 下抵消较少，不等于 correctness、稳定状态或“推理谷底”。只要
较早 response hub 仍携带 evidence lineage 且被有效接纳，target 没有 direct prompt attention 也可能
保持 grounded；但这一点仍需 matched factual pair 和因果验证。

已有 artifact 可单独重画：

~~~bash
python -m experiments.reanchor_flow.run mechanism-plot \
  --artifact /path/to/audit.npz \
  --output /path/to/mechanism.png
~~~

subset 的 `--plot` 用当前 tokenizer 解码 artifact token IDs；单独运行 `mechanism-plot` 时可用
`--tokens-json` 提供显示字符串。两种绘图都不读取 hallucination labels。

## 当前结论边界

- attention 下三角图足以提出“路由切换”候选，但不能证明读入内容、接纳或因果必要性。
- residual difference 只说明 intervention 改变 state；gradient/action 只说明固定 contrast 附近的局部
  sensitivity/consistency，均不能从 residual 中自动解耦出纯事实语义。
- native observed-token support 不是 factual correctness。区分 fact、confidence、词法/语法需要
  fact value × confidence 的 \(2\times2\) matched worlds、nuisance controls、跨模板/实体泛化及 selective
  causal swaps。
- response history 的使用是正常生成；teacher-forced DAG 不能证明幻觉 self-reinforcement、stable
  basin 或 attractor，这需要 free-running 双向扰动实验。
- [*How do LLMs Compute Verbal Confidence?*](https://arxiv.org/abs/2603.17839) 只提供
  cache→retrieve 的实验模板，不提供本项目的固定层、节点或 groundedness 结论。

## 测试

~~~bash
python -m pytest -q \
  experiments/common/tests/test_llama_message_intervention.py \
  experiments/reanchor_flow/tests
~~~
