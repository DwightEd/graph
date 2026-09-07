# 逐 head 的时间轴重锚定机制审计 v3

## 已完成扫描后：直接得到检测指标与事件对照

`scan_analyze` 读取已有 `train/test/run_manifest.json` 和 `scans/*.npz`，在 CPU 上
拟合逐 head 时序路由密度，评分 **test 全部已扫描 token**，包括首 token。
它不需要 functional targets，因此 `functional_targets=0` 不再阻止这条结构检测评估路径。
无需重跑 LLM forward、gradient 或逐路由消融。标签仍需原 cache；formal cache 的标签读取会
反序列化单个 attention 文件，不加载模型权重。

在项目根目录执行（`--scans` 指向你本次全量扫描的实际输出根目录）：

```bash
git pull --ff-only origin main
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.scan_analyze \
  --scans experiments/reanchor_flow/outputs/mechanism_all_v3 \
  --output experiments/reanchor_flow/outputs/routing_detection_v1 \
  --supervised-probe
```

`--supervised-probe` 额外给出一份明确使用 train 标签的线性诊断及同预算位置基线，
用于区别“扫描没有判别信息”和“当前无监督假设不合适”。它不属于无监督主方法；只跑
无监督检测时去掉该参数。两条结果分开汇报，绝不根据 test 指标翻转分数方向或选 head。
若 cache 移动了，加 `--cache /新的cache根目录`，其下应有 `train/`、`test/`。

首先打开 `routing_detection_v1/detection_summary.md`，其中包括 ALL 和三个任务的
AUROC、AUPRC（average precision）、幻觉比例、按 source 成簇 bootstrap 的 95% 区间，
以及主方法相对静态路由、位置基线的**配对差值区间**。AUPRC 应结合幻觉比例看；时序方法若
没有稳定超过静态路由与位置对照，不能声称重锚定机制改善了检测。当前实现未在真实扫描上
实测检测效果；单凭已经上传的 cohort 聚合值无法还原 token 排名或 AUROC。

| 结果 | 用途 |
|---|---|
| `detection_report.json`、`detection_summary.md` | 全 token 检测结果、覆盖率、基线与不确定性 |
| `detection_curves.png` | ALL/各任务 ROC、PR 曲线 |
| `models/*.npz`、`source_partition.json` | 固定模型、校准参数、fit/calibration/test source 名单 |
| `frozen_detector.json`、`predictions/*.npz` | 逐样本全部 token 分数、q→q+1、最高异常贡献 head、事后标签 |
| `events/event_audit.json`、逐 head PNG | train 选 head 后，test 的正常/幻觉事件及邻近非事件对照轨迹 |
| `events/event_pairs.jsonl` | 具体事件/对照位置、各桶 winner source position/unit，可继续定位人工或功能审计 |

拟合、校准、全 token 评分、标签连接、bootstrap、事件统计均有 tqdm。按样本读取，fit 每样本
最多 128 行；评分保留全部行，内存随单样本长度而变，不保存所有样本的 head tensor。
`--no-events` 可只跑检测，`--no-plot` 关闭图片。详细建模与尚未验证的假设见
[METHOD.md](METHOD.md#扫描后的逐-head-时序路由检测)。

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

## 当前审计覆盖哪些步骤

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
| 独立完整时间轴与跨样本比较 | `sample_scan.py`、`cohort_plot.py` |
| 扫描/标签边界、无监督时序模型 | `scan_dataset.py`、`routing_transition.py` |
| 检测流程/校准、独立评估、事件对照 | `scan_analyze.py`、`detection_metrics.py`、`routing_events.py` |
| 可选监督读出诊断 | `routing_probe.py` |
| 数据选择、运行、评价与作图 | `subset_data.py`、`subset.py`、`subset_report.py`、`mechanism_plot.py`、`run.py` |

`reanchor_timeline.py` 定义底层 local→long-range 分数与原功能审计选点；离线检测不使用其中
读取右邻居的 peak mask。`routing_events.py` 在同一底层分数上用固定 train 阈值和左侧 refractory
定义可因果定位的报告事件。`route_plan.py` 只负责固定 target 的候选拓扑，
`route_model.py` 只负责 provenance/action/integration ledger。`audit.py` 与 `corridor.py` 并非旧版重复物：
它们保留 aligned clean/corrupt world 的双向 patch 路径，后续 fact×confidence matched audit 仍需要它。
artifact 的 schema、payload 和 validator 已分开，避免在运行流程里手工搬运一长串字段。

## 默认运行语义

- `audit-all` 默认扫描 train+test、QA/Summary/Data2txt 的全部可用样本（`samples-per-task=0`），
  保留完整 response（`max-response-tokens=0`），不根据正确/幻觉标签筛样本。默认功能审计是
  `reanchor-window`、每样本最多 3 个 target；`--scan-only` 可先只完成全部样本的结构审计。
  `subset` 仍默认每任务 1 个样本、128 response tokens。
- `subset` 默认 `--target-policy reanchor`：在当前模型的 clean full-row transport 中按最强单 head 的结构分数
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

`negative candidate is not the frozen native runner` 的原因是旧流程在完整 response 上冻结 runner，
截取 target prefix 后又重新 argmax。近并列候选可能因数值舍入换位。现在 prefix capture 显式接收
冻结的 runner，gradient、margin 和 cut 共用同一 contrast，仍检查 target identity。
结构事件也保留原 discovery identity/score；prefix 重算结果作为独立诊断，不要求两次排序或分数完全相等。

## 一键运行：先比较全部正确/幻觉样本的结构

在项目根目录运行。模型、cache、source-info 默认采用本项目现有服务器路径，可用同名参数覆盖：

~~~bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph &&
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run audit-all \
    --scan-only --query-chunk 4 \
    --output experiments/reanchor_flow/outputs/mechanism_all_v3
~~~

每个 split 自动保存完整样本 scan 与时间轴图，并在 capture 完成后生成 cohort 汇总图/报告。
该阶段不计算 gradient 或 root cut，适合先观察普遍模式；它没有验证 MLP 接纳或事实因果贡献。
这里的“全部”指所配置 cache 中的全部可用样本；尚未在本地实际跑完全部 RAGTruth。

随后用相同 output、相同配置去掉 `--scan-only`，复用 world/scan 补每样本最多 3 个 target 的功能审计：

~~~bash
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run audit-all \
    --query-chunk 4 \
    --output experiments/reanchor_flow/outputs/mechanism_all_v3 --plot
~~~

`--plot` 额外绘制每个 target 的详细机制图；`audit-all` 不加它仍有样本时间轴和 cohort 图。
运行时有 tqdm 的 sample、target 及绘图/评价阶段进度；每个 target 完成后立即落盘，相同 scientific
config 可以恢复。已跑过的 v3 `subset` 在同一配置下可补建缺失 scan，保留原 target plan；改为
全量/完整 response 后配置不同，请使用新的 `mechanism_all_v3` output。旧 schema 不能混用。

若进程被系统直接打印 `Killed`，通常是 OS/cgroup 的 OOM，不是 `corridor_ok=False` 导致的异常。
先保持 `carrier-scope=response` 并降低 `query-chunk`。功能审计的单层 autograd 仍可能保存
\(O(H T^2)\) 中间量；destination 预算不是完整 prefix 的显存上限。如必须限制
`max-response-tokens`，该次只能称截断 response 审计，coverage 会记录差异。
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

`audit-all` 在每个 split 完成 label-free capture 后自动独立读取 hallucination labels。
已有结果也可单独汇总，无需重新加载模型：

~~~bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph &&
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run subset-evaluate \
    --split all \
    --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
    --output experiments/reanchor_flow/outputs/mechanism_all_v3 --plot
~~~

`cohort_summary.json` 与 `cohort_{ALL,QA,Summary,Data2txt}.png` 比较完整 scan 上的四类来源
transport 占比与切换率，保留每个 layer/head。先在每样本、每标注组内平均 tokens，再让样本等权；
差值列只用同时包含两类 token 的 mixed 样本做样本内“幻觉−非幻觉”差。
少于 3 个贡献样本的格子显示灰色。此处是描述性比较，尚未控制位置、claim 类型或同 source 样本依赖，
不报告显著性或 bootstrap CI；未标幻觉也不等于独立核验事实正确。

`cohort_functional_*.png` 仅比较选中 target 上的 action/integration。完整结构覆盖和选点功能覆盖
分别报告；`--scan-only` 没有功能 AUROC，不能把缺失值当成零效果。

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

下表是 `subset` 默认值；`audit-all` 覆盖为全部样本、完整 response、`reanchor-window` 与 3 targets。

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--target-policy` | reanchor | 用 full-row true-message switch 选事件中心 |
| `--targets-per-sample` | 1 | target row 硬预算；事件窗口审计建议 3 |
| `--carrier-scope` | response | 只展开 response destinations；prompt 仍可作 source |
| `--max-response-tokens` | 128 | clean 时间轴 pilot 的 response horizon |
| `--max-route-rows` | 256 | 超限时保留靠近 target 的最近连续 destinations；不裁剪 causal sources |
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

destination 截尾不改变完整 sample scan。未展开的 response 节点从 layer 1 起记为 `UNOBSERVED`，
不能把它们沿用为“纯 response-origin”，也不能由缺失的 lineage 推断其没有 prompt 证据。

## 图怎么读

输出按 `train/`、`test/` 分目录：`scans/<task>/*.npz` 与 `.timeline.png` 保存样本完整结构时间轴；
`cohort_summary.json`/`cohort_*.png` 保存跨样本比较；`audits/<task>/<sample>/` 保存已选 target
的详细功能结果。完整结构 scan 不等于完整事实接纳验证。

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
