# 信息形成与后续使用：研究定位和现有工具

当前研究主线是：**被后续生成复用的中间状态怎样形成，以及它对输入约束的响应是否实际影响后续输出。**
论文对照、具体审计顺序、通过条件与实现缺口统一记录在 [MECHANISM_AUDIT.md](MECHANISM_AUDIT.md)。

目前还没有完成这条机制的验证。四桶监督读出用于检查判别信息，`discover` 的 PCA／聚类用于
探索写入组合；两者均不是已经验证的机制驱动检测器。下面保留现有工具的复现命令，不能把运行成功
或输出 AUROC 当成新主线已经完成。下一项研究产物应是逐 head 原图和同一载体的形成／复用过程。

## 复现已有向量探索工具（不等于机制确认）

以下命令覆盖已有扫描中的 train/test 全部样本与完整 response；在项目根目录启动，避免
`ModuleNotFoundError: No module named 'experiments'`：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph && \
git pull --ff-only origin main && \
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.discover \
  --scans experiments/reanchor_flow/outputs/mechanism_all_v3 \
  --output experiments/reanchor_flow/outputs/native_discovery_v1 \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --query-chunk 8 --sketch-dim 16 --fit-rows 4096 \
  --components 8 --patterns 6 --plots-per-task 4
```

每个样本两次无梯度前向；首次加载模型另有一次小规模实现一致性检查。采集按样本保存并可续跑。
同一输出目录重复执行会复用已完成的原生 trace，再运行分析；修改模型、投影或截断配置须换输出目录。
`--phase capture` 只采集，`--phase analyze` 只分析已有 trace，不加载大模型权重。
模型／dtype 默认可从旧扫描配置读取；原标签 cache 移动后用 `--cache /新cache根目录`。

默认 `--samples-per-task 0 --max-response-tokens 0` 表示全部样本／完整回答。
`--plots-per-task 4` **只控制示例图数量**，按样本 ID 选取，不筛统计；设为 `0` 画全部已采集测试样本。
采集、拟合、预测、标签审计和绘图都有 tqdm。需要先验证服务器环境时，使用单独输出目录加
`--samples-per-task 1 --max-response-tokens 32`，不能把这次小样本结果解释为完整实验。

## 先看哪些结果

| 输出（相对 `native_discovery_v1/`） | 用途 |
|---|---|
| `summary.md`、`mechanism_report.json` | 所有模式的覆盖、来源配对差异、区间、多重比较及数值闭合 |
| `mechanism_cohort.png` | 各任务中每个模式的正常／幻觉出现率差异与来源数 |
| `mode_writes_<task>.png` | 各模式的逐 head 和 MLP 带符号读出表现；不平均 head |
| `figures/<task>/<sample>.png`、同名 JSON | 全回答模式轨迹、局部 head／残差／MLP 变化和具体来源位置 |
| `train,test/traces/<task>/<sample>.npz` | 无标签原生向量投影、逐 head 写入、来源边和模块阶段 |
| `train,test/examples/<task>/<sample>.json` | 每个样本的无标签变化候选：前后模式、具体 head／MLP 层及带符号向量差 |
| `train,test/predictions/<task>/<sample>.npz` | 冻结模式、坐标、变化距离和重构误差，覆盖所有采集行 |
| `native_patterns.npz`、`pattern_geometry.npz` | 联合模式模型、保留 head 轴的模式中心与分量载荷 |
| `source_partition.json`、`plot_selection.json` | 模型拟合来源／行、测试来源、重叠排除与无标签选图记录 |
| `detection_report.json`、`detection_curves.png` | 所有标注 token 的 AUROC／AUPRC、来源 bootstrap 区间和基线 |

模式距离、相邻变化、低 observed-token margin 和位置都输出检测诊断，但**模式稀有／变化大没有被
假定为已证实的幻觉机制**。分数方向预先固定；不根据 test 翻转方向。首先检查模式能否复现、涉及什么
运算，再判断与幻觉是否有稳定差异，不能只按最高 AUROC 给机制命名。

## 已有向量工具的组织

| 模块／对象 | 责任 |
|---|---|
| `scan_dataset.py` | 复用已有扫描的输入坐标；`ScanLabelStore` 在模式与预测保存后才读取标签 |
| `native_trace.py`：`FrozenReadout`、`NativeTraceObserver` | 两次原生前向，逐 head 向量、模块更新与固定读出记账 |
| `native_patterns.py`：`NativePatternModel` | 不读标签的联合向量模式学习、冻结、全 token 转换 |
| `native_report.py`：`NativeCohort` | 流式来源级归纳、关联统计与样本／总体可视化 |
| `discover.py` | CLI、阶段编排、续跑和数据流；不重复模型运算 |
| `experiments/common/llama_message_intervention.py` | 已有 Llama／GQA 前向与 observer 接口，供新旧流程共用 |

现有向量字段见 [SCHEMA.md](SCHEMA.md)，研究步骤与尚未验证的假设见 [MECHANISM_AUDIT.md](MECHANISM_AUDIT.md)。
默认固定 16 维投影有损；全部 head 保留不等于全部信息无损。`--save-head-codes` 可另存每个 head
在 W_O 前的完整净写入，明显增加磁盘占用。展示边只存每个 head 最多 `--display-edges 2` 条，
其余来源仍参与完整净写入和统计，同时保存遗漏的带符号总量与绝对总量。

## 以下为旧四桶基线与有限 target 审计

这些入口用于复现已有结果，不生成上面的原生向量模式。

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
没有稳定超过静态路由与位置对照，不能声称重锚定机制改善了检测。用户已运行的 ALL 结果中，
routing_joint AUROC 约 0.530，监督 routing 约 0.835；这证明有可读出的标签关联，未证明重锚定机制。

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

## 监督读出成功后：定位实际贡献的 head

如果已运行 `--supervised-probe`，在项目根目录执行：

```bash
python -m experiments.reanchor_flow.probe_heads \
  --analysis experiments/reanchor_flow/outputs/routing_detection_v1
```

入口自动读取已保存的模型、source 划分和原扫描路径。扫描或 cache 移动后，分别用
`--scans`、`--cache` 覆盖其含 `train/`、`test/` 的根目录。有 tqdm，不重新拟合检测器，
不运行大模型或逐路由干预。

`HeadReadout` 将完整监督 logit 精确分解为每个 head 的当前状态项、相邻变化项，以及
上下文项和截距；保留全部 layer/head。参数 L2 排名仅用于查看系数尺度，不能代替实际贡献。
实际筛选在原 train 的 calibration sources 上比较同一样本、同 log2 位置区间内的
幻觉−正常贡献差，再按样本、source 平衡。取正差值最大的最多 20 个 head 作为检查预算，
先保存选择，再读取 test 进行同样的贡献核查。fit 来源的结果单独报告。

输出 `head_audit/frozen_head_selection.json` 和 `head_audit/head_report.json`。后者保存所有
head 的 `parameter_rank`、校准选择的 `selected_rank`，以及 fit/calibration/test 的贡献差、
正方向 source 比例和配对样本量。缺少可配对 calibration source 时，选择为空，不从 test 补选。
这仍是**使用标签的监督读出诊断**，不是无监督 head 发现或因果重要性；当前/差分相关、head
相互补偿、粗位置区间及有限配对来源都会影响解释，也没有验证这些 head 子集能保持完整 AUROC。

v3 的原审计目标是：模型在位置 \(q\) 是否从最近少数 response token，切换为读取原始
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

现有实现见 [METHOD.md](METHOD.md)，修订后的研究设计与结论边界见
[MECHANISM_AUDIT.md](MECHANISM_AUDIT.md)，artifact 契约见 [SCHEMA.md](SCHEMA.md)。

## 当前审计覆盖哪些步骤

此前版本没有把这一机制闭环：早期版本画过 response 下三角路线变化和 prompt revisit，但混合或平均
了 heads，也没有把“local 下降 + long-range 上升 + 信息接纳”绑定在同一个事件；后来的版本加入了
真实 message、gradient、hub 与 intervention，却改成固定 target 的静态 route，丢掉了在完整生成时间轴
上先找切换点的步骤。

v3 已提供全时间轴逐 head 结构扫描，以及有限 target 的 action/integration 和可选干预。
扫描本身没有向量与 MLP 状态；target 的 observed-vs-runner 目标也没有事实语义。
因此尚未完成“限定条件选择 → 事实整合 → 后续调用”的机制闭环。

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
| 可选监督读出与逐 head 贡献核查 | `routing_probe.py`、`probe_heads.py` |
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
