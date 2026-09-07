# Mechanism audit data contracts

## Native computation discovery (`discover`, schema 1)

当前入口的产物与下文旧 v3 target/route schema 分开。逐样本 trace 与预测不保存幻觉标签；
标签只在模型、预测、选图名单冻结后用于报告和图的事后覆盖。所有 NPZ 支持 `allow_pickle=False`。

记 L=layers、H=query heads、Q=实际 response predictor 数、R=sketch 维数、K=展示边数、D=残差维数。
绝对 predictor `row_position[q]` 预测 `token_ids[row_position[q]+1]`；不平均 head。

| 字段 | 形状／语义 |
|---|---|
| `native_trace_schema`, `dataset_sample_id`, `source_id`, `task_type` | schema 与输入身份 |
| `token_ids`, `token_text`, `response_start`, `row_position` | 输入、可读 token 文本、response 起点、绝对 predictor |
| `sketch_projection`, `sketch_seed`, `sketch_dim` | [D,R] 固定共享投影与可复现参数 |
| `head_sketch`, `mlp_sketch` | [L,H,Q,R]、[L,Q,R] 实际净写入的线性投影 |
| `residual_sketch` | [L+1,Q,R] 每层输入与最后一层后的 raw residual |
| `attention_sketch`, `post_attention_sketch` | [L,Q,R] attention 写入与加法后的状态 |
| `head_norm`, `residual_norm`, `attention_norm`, `post_attention_norm`, `mlp_norm` | 原始完整向量范数，非四桶范数之和 |
| `head_margin`, `attention_margin`, `post_attention_margin`, `mlp_margin`, `stage_margin` | 同一 q 使用冻结方向的直接读出项；保留各自 head／layer 轴 |
| `observed_token_ids`, `runner_token_ids`, `final_margin`, `observed_logprob`, `entropy` | [Q] 第一次原生前向的实际读出 |
| `final_rms_denominator`, `readout_bias`, `readout_remainder` | [Q] raw final RMS、输出偏置与最终舍入余项 |
| `edge_source_position`, `edge_attention`, `edge_margin` | [L,H,Q,K] 显示边；空位置 -1、数值 0 |
| `edge_sum_margin`, `edge_total_absolute_margin` | [L,H,Q] 所有 source 边的带符号和／绝对和 |
| `omitted_margin`, `edge_omitted_absolute_margin`, `edge_rounding_remainder` | 显示遗漏与 source→head 舍入；omitted_margin 含该舍入 |
| `head_remainder_*`, `attention_add_remainder_*`, `mlp_add_remainder_*` | `_sketch`／`_margin` 中 head 聚合（含输出偏置）和实际残差相加余项 |
| `forward_repeat_max_abs_error` | [Q] 两次原生 raw final 向量的最大重复误差 |
| `head_code`（可选） | [L,H,Q,head_dim] float32 的 W_O 前净写入，不是全 source 路由 |

`train,test/native_manifest.json` 保存配置、输入身份、每个 trace 的相对路径与行数、完整状态。
续跑检查配置与 token 坐标，不复用不一致 trace；输出变更需新目录。读取旧扫描只取输入 token，
不把旧四桶数据当成新向量。

`native_patterns.npz` 保存 `NativePatternModel` 的 shape、scale、mean、components、centers、
explained_variance_ratio、fit_rows、seed。`pattern_geometry.npz` 保存恢复原 sketch 单位的
`center_head_sketch`／`center_mlp_sketch` 与 `loading_head_sketch`／`loading_mlp_sketch`，均保留 head 轴。

逐样本 predictions 保存 `coords[Q,C]`、`pattern_id[Q]`、`distance[Q]`、`transition[Q]`、
`has_previous[Q]`、`reconstruction_error[Q]` 以及绝对／response 坐标。C 是保留 PCA 维数，
不是类别数。首行或缺失前行的 transition=0 且 has_previous=False。

逐样本 `train,test/examples/<task>/<sample>.json` 在标签读取前保存。无论是否绘图，均记录最多 8 个
间隔至少 3 个 predictor 的高变化候选、前后模式、变化最大的 head／MLP 层和完整 signed sketch delta。
该数量只限制人工检查清单，全部 token 的预测、转移和统计不受影响。

`mechanism_report.json` 包括所有模式、全部转移（对角线为持续）、模式条件下的逐 head 有符号均值、
位置匹配 H−N 差、source 区间、全任务×模式 BH q、未配对计数及数值闭合。
`detection_report.json` 单独报告固定诊断的所有标注 token AUROC／AUPRC 与 source 区间。
图和具体变化示例的预算不减少模式拟合的 head 轴或全量审计分母；示例不是已确认的机制事件。

以下为旧 v3 复现路径的数据契约。

## Mechanism-audit v3 data contract

本文定义 v3 artifact 的语义边界。实现可以增加诊断字段，但不能改变以下不变量：时间轴结构事件先
由 clean full-row transport 冻结，target 与 AuditPlan 随后冻结；候选选择不读取 label 或 exact
intervention outcome；所有 message 保留 head；未观察质量不能重新分配到已保存的 route。

## 1. 坐标与基本不变量

所有 token 坐标都是绝对 position。对 predictor 位置 \(q\)，prediction position 是 \(q+1\)，
causal prefix 长度为 \(N=q+1\)。

| 轴 | 含义 |
|---|---|
| L | decoder layers |
| H | query heads；始终保留，不平均 |
| N | 当前 target 的 causal source positions |
| P | capture 中实际表示的 destination rows |
| E | artifact 中为绘图/诊断保存的 sparse 逐 head message edges |
| E_c | frozen AuditPlan 中的完整 corridor message rows |
| A | stage trace positions |
| U | source units |
| R | frozen root candidates |
| K | frozen hub candidates |
| B=4 | prompt_evidence / other_prompt / remote_response / recent_local 位置桶 |
| C=4 | all registered evidence / other_prompt / response / unobserved provenance |

NPZ 必须能用 allow_pickle=False 读取，只含标量、字符串和数值数组。hallucination/correctness label 不得
进入 world、AuditPlan、mechanism NPZ 或机制图。

默认 carrier_scope=response。represented rows 从 response-side predictor rows 中产生；所有早于该
destination 的 causal prompt positions 仍可作为 source。若要搜索 prompt 内部 destination/hub，必须显式
使用 carrier_scope=all，并同样受 rows budget 约束。超过 max_route_rows 时保留最近连续 destinations，
所有 causal sources 仍参与读取；实际窗口由 route_row_position 给出，不改变完整 sample scan。

## 2. Native world：label-free target contract

native_world_schema=2 是当前 label-free world；loader 只为迁移兼容读取 schema 1，新的 reanchor run
必须写 schema 2：

| 字段 | 形状 | 含义 |
|---|---:|---|
| sample_id, tokenizer_id | scalar | 安全样本 ID 与 tokenizer identity |
| token_ids | [N_full] | prompt 与 teacher-forced response |
| response_start | scalar | 第一个 response token 的绝对 position |
| token_unit_id | [N_full-1] | 每个 causal source position 的 unit |
| unit_name, unit_kind | [U] | passage/sentence/field/response 表 |
| evidence_unit_id | [R_world] | 可成为 root 的预注册 source units |
| query_position | [T] | 冻结 predictors |
| positive_token_id | [T] | observed token |
| negative_token_id | [T] | native run 后冻结的 runner |
| contrast_origin | [T] | target 选择与 contrast 语义 |

该 world 只定义 observed-token contrast，不定义 factual correctness。negative_token_id 在完整
discovery run 冻结后，prefix baseline 用 `fixed_runner` 显式保留；cache 的 runner/readout/margin、
gradient 与 cut 必须共用它，不以 prefix 的新 argmax 覆盖。

### 2.1 Label-free target selection

`target_selection_policy=reanchor` 时，`targets_per_sample=N` 是最多 \(N\) 个 structural event centers；
事件按最强单 head 的 full-row true-message score 排序，并做时间半径 1 的 NMS。
`reanchor-window` 把 \(N\) 作为 target-row 硬总预算：先分配最多 \(\lceil N/3\rceil\) 个中心，再按 rank
补中心 \(-1,+1\) 邻居。完全无事件时才 evenly-spaced fallback。
当预算覆盖全部短 response rows 时仍必须运行 scan，并给命中的中心保存 event record；其他已选 row
保存 non-event record，`fallback=false`。

selector 必须重算当前 clean model 的完整 causal rows，并在 sparse pruning 前聚合第 6.1 节四桶；不得
使用外部缓存的稀疏 attention 值、gradient、label 或 ablation outcome。`contrast_origin` 必须包含
event 的 layer/head/source kind/center/offset/score/support，或明确的 no-event fallback origin。
每个最终 artifact 仍只对应一个 frozen query/contrast。

world 与 target artifact 还必须持久化相应的 `target_reanchor_*` selection record：

| 字段 | 含义 |
|---|---|
| selection_recorded/policy/scan_signal | 是否有记录、策略与 `exact_full_row_message_transport` selector |
| has_event/is_center/fallback | 该 row 来自事件中心、窗口上下文还是 no-event fallback |
| center_position/window_offset | 注册中心与当前 query 的相对 offset |
| layer/head/source_kind/source_position/source_unit_id | 选中结构事件的精确 head 与 winner source |
| score/support | 最强单 head structural score 与同时间 peak-head 数 |
| previous_anchor_fraction/anchor_fraction | dominance flip 的两侧份额 |
| relative_anchor_rise/relative_local_fall | transport-only score 的注册分量 |

非 reanchor policy 使用 `selection_recorded=false` 和文档化的空 sentinel；reanchor no-event fallback
必须 `selection_recorded=true, fallback=true, has_event=false`，不得与非 reanchor run 混淆。

### 2.2 独立样本时间轴

`scans/<task>/<sample>.npz` 的 `sample_scan_schema=1` 在 target selection 前保存，不含 labels。
`route_row_position` 覆盖所配置 response horizon 的全部 predictors；`audit-all` 默认 horizon 为完整 response。

| 字段 | 形状/含义 |
|---|---|
| dataset_sample_id/source_id/task_type | 样本与来源 identity |
| token_ids/response_start | 完整采集 token 序列与 response 起点 |
| full_response_tokens/processed_response_tokens | 原始/实际采集 response 长度 |
| route_row_position | [P_scan]，包含首个 response token 的 predictor |
| reanchor_bucket_attention/transport | [L,H,P_scan,4]，裁剪前完整 source-row 四桶 |
| reanchor_bucket_source_position/source_unit_id | [L,H,P_scan,4]，桶内最强 source |
| reanchor_score | [L,H,P_scan]，纯 transport 切换几何 |

对应 `.timeline.png` 保留每个 head 的完整 raster，另显示最多 4 个单 head 的四桶轨迹。
扫描第一行没有前一个观测 predictor，不能当作已证实的“无切换”纳入 cohort 切换率分母。

## 3. Native mechanism artifact identity

v3 使用 subset_audit_schema=3、method_version=budgeted_head_resolved_route_audit_v3，每个 target 保存
一个 NPZ。

identity 至少包括：

- dataset_sample_id、sample_id、source_id、split、task_type、generator_model；
- model_id、model_dtype、tokenizer_id；
- response_start、query_position、prediction_position；
- positive_token_id、negative_token_id、contrast_origin；
- target_selection_policy、target_selection_rank；
- target_reanchor_* selection record（reanchor policies 时）；
- flow_signal、carrier_scope、edge_coverage、query_chunk、local_window；
- token_ids、token_unit_id、unit_name、unit_kind、evidence_unit_id。

同一 target 若改变 target contrast、source-unit table、destination scope 或任何 route budget，就是不同的
scientific identity，不得复用旧 artifact。

## 4. 冻结 AuditPlan

AuditPlan 必须在 selected-root paired diagnostic、任何 confirmation rerun 及 label access 之前生成。
计划的逻辑 identity 由上一节的 target/run identity、represented rows、预算和下列有序候选共同确定；
confirmation flag 与 outcome 不进入
候选选择。实现若保存 plan hash，hash 输入也必须遵守这个边界。

### 4.1 预算字段

| 字段 | 默认 | 语义 |
|---|---:|---|
| route_budget_edges_per_head | 2 | capture 每分支的 k 与 frozen corridor 每 head-row 的硬 cap |
| route_budget_max_rows | 256 | 最近连续 represented destination rows 上限；sources 不截断 |
| route_budget_root_candidates | 4 | 保存的 root candidates 上限 |
| route_budget_hub_candidates | 8 | 保存的 hub candidates 上限 |
| route_budget_corridor_edges | 64 | frozen corridor 显式 messages 的硬上限 |
| route_budget_confirm | false | 是否在计划冻结后执行 validation ladder |

capture 对每个 head-row 分别取 transport top-k 与 \(|\text{functional}|\) top-k 的并集，因此最坏
保留 \(2k\) 条不同边；frozen corridor 再限制为每 head-row 至多 \(k\) 条。edge_save_limit 只控制
绘图/诊断 sparse table 的保存量，不得改变 AuditPlan；edge_coverage 是 transport 分支在 top-k 内的
覆盖目标，也不得突破 capture 2k、corridor k、rows 或全局 corridor budget。

### 4.2 Root plan

下列数组第一维均为 R，且顺序就是冻结 rank：

| 字段 | 含义 |
|---|---|
| root_unit_id | source unit ID |
| root_route_mass | 与固定 target 相连的 norm-based route mass |
| root_signed_action | route-weighted signed grad-message |
| root_absolute_action | 逐 head action absolute budget |
| root_functional_agreement | absolute signed sum / absolute budget；是抵消惩罚，不是 head 平均 |
| root_selection_score | 非负 magnitude：route mass × absolute action budget × agreement |

selected_root_unit_id 必须来自该冻结数组。若执行了注册的 zero-action fallback，artifact 必须明确记录
selected_root_selection_fallback=true；root_signed_action 单独提供方向，agreement 不在净 signed action
上重复乘第二次。exact effects 不得作为 tie-break。

### 4.3 Hub plan

hub_layer、hub_position、hub_route_mass、hub_signed_action、hub_graph_score 的第一维均为 K。
hub_graph_score 是 route mass × 两侧 absolute-action-budget bottleneck × agreement bottleneck 的非负
magnitude；hub_signed_action 单独保存方向。若另外保存 hub_functional_agreement，它是 score 的
可审计分量，不是 head 平均。相邻层同 position 的 NMS 顺序是计划的一部分。

### 4.4 Corridor plan

完整 frozen corridor 使用独立的 `frozen_corridor_*` 表权威保存，不依赖绘图 sparse `edge_*` 表，也不受
edge_save_limit 影响：

| 字段 | 形状 | 含义 |
|---|---:|---|
| frozen_corridor_edge_index | [E_c] | capture 的完整内部 edge table 中冻结的全局 index；不是 sparse `edge_*` row index |
| frozen_corridor_layer/head/source/target | [E_c] | corridor 的完整逐 head endpoint |
| frozen_corridor_source_unit | [E_c] | source position 所属 unit |
| frozen_corridor_attention_native, frozen_corridor_attention_root_cut | [E_c] | native 与固定 selected-root-cut world 的 QK gate |
| frozen_corridor_native_functional_score, frozen_corridor_root_cut_functional_score | [E_c] | native/root-cut message 的 signed target action |
| frozen_corridor_source_evidence_lineage_fraction | [E_c] | source node 的 all-evidence lineage share |
| frozen_corridor_evidence_lineage_action | [E_c] | all-evidence lineage share × native signed functional score |
| frozen_corridor_native_message_norm, frozen_corridor_root_cut_message_norm, frozen_corridor_delta_message_norm | [E_c] | matching W_O 后的 native、root-cut 与差分 message norm |
| frozen_corridor_root_throughput | [E_c] | selected-root-conditioned route mass；与 all-evidence lineage 是不同 estimand |
| corridor_edge_count | scalar | frozen plan 的显式 message 数，必须等于 E_c |
| edge_in_frozen_corridor | [E] | 可视 sparse `edge_*` 表的 corridor membership；true count 必须等于 E_c |
| edge_on_backbone | [E] | 绘图 sparse table 中是否属于 connected backbone |

connected backbone 可以用 residual continuation 跨过没有显式 message 的层，但 message 数、补充边和
每个 head-row 都必须满足冻结预算。若预算小于 backbone 所需显式 message 数，应失败并要求新配置，
不能静默扩大。所有 `frozen_corridor_*` 数组必须具有相同的 E_c 行数并与
frozen_corridor_edge_index 一一对应；恢复验证以这张表为准。绘图表必须保留完整 backbone 与
corridor，因此它们可使实际保存数超过 edge_save_limit；其余 marginal edge 才受该限制。若跨 head
分组，仍使用 absolute budget × agreement 而不是 head 平均。

## 5. 真实逐 head message 与 sparse capture

一条边的语义是 post-W_O residual write：

\[
m^{l,h}_{j\to i}=W_O^{l,h}\left(
A^{l,h}_{i,j}W_V^{l,g(h)}\operatorname{LN}(r_{l,j})
\right).
\]

实现可以在 pre-W_O code 上用 \(W_O^TW_O\) 计算 norm，也可以在等价的 pre-W_O endpoint 做精确
干预；artifact 中的 message norm 与 functional score 始终解释为 matching W_O slice 后的 residual
message。不能把 attention × V 或 attention weight 本身称为 message。

以下字段第一维均为 E，且行一一对应：

| 字段 | 含义 |
|---|---|
| edge_layer, edge_head, edge_source, edge_target | 精确逐 head 坐标 |
| edge_source_unit | source position 所属 unit |
| edge_attention_native | QK gate，仅作对照 |
| edge_native_message_norm | 真实 post-W_O message norm |
| edge_native_functional_score | \(\langle\nabla_{m_e}F_t,m_e\rangle\)，保留符号 |
| edge_root_throughput | selected-root-conditioned route mass |
| edge_source_evidence_lineage_fraction | source node 的 all-evidence lineage share |
| edge_evidence_lineage_action | all-evidence lineage share × native signed functional score |
| edge_on_backbone | 是否在保存的 connected backbone 上 |
| route_edge_origin | [E,C] provenance ledger |

由于 EVIDENCE 是 all-evidence provenance，schema 3 使用
edge_source_evidence_lineage_fraction 与 edge_evidence_lineage_action；旧的 root_lineage 字段名会把
该量误解为 selected root，不属于本契约。

edge_attention_root_cut、edge_root_cut_functional_score、edge_root_cut_message_norm 与
edge_delta_message_norm 只在相应 source-cut world 被显式生成时有意义；未生成时必须以 evaluated/status
字段或缺失值区分，不能用零冒充 negative result。

保存的 sparse edge 表不是全图。候选 capture 先做 transport top-k 与 absolute-functional top-k 的
并集，故 E 中一个 head-row 最坏有 2k 条边；进入 frozen corridor 后同一 head-row 最多 k 条。每个
完整 head-row 在裁剪前计算 total mass，并保存或可由 ledgers恢复以下诊断：

| 字段 | 形状 | 含义 |
|---|---:|---|
| route_row_position | [P] | destination slot 到绝对 position |
| route_row_total | [L,H,P] | 裁剪前完整 row mass |
| route_row_retained | [L,H,P] | 显式保留 edge mass |

`unobserved = clamp(route_row_total - route_row_retained, min=0)`，对应 fraction 为
`unobserved / route_row_total`（零 total 时取零）；这两个量由保存的 total/retained 现算，不重复持久化。

scope 外 prompt destination 另作为 unrepresented-row limitation 记录，不能伪造为某个 represented
row 的质量。它没有 attention row，只以 residual probability 1 保留已有 all-evidence/other-prompt
provenance，使后续 represented row 仍可读取该 prompt state；这不表示 prompt 内部更新已被审计。
未 represented 的 response state 从 layer 1 起为 UNOBSERVED，不能沿 residual 复制成确定的
response-origin；其零 evidence register 表示未观察，不能解释为无证据。
若实现只在 provenance ledger 中表示缺失质量，则
route_head_transport[..., unobserved] 是最低限度的 completeness 诊断。无论采用哪种持久化方式，都
必须满足非负和质量守恒；未保存质量不得重新归一到 retained edges。高 unobserved fraction 会降低
路线完备性，不能解释为“其余路径不存在”。

## 6. Connected backbone 与完整 ledgers

backbone 是 layer-unrolled DAG 中从 selected-root prompt position 到
(layer_count, query_position) 的一条 widest candidate path。它保证可视化连通，不保证最小性或因果性。

| 字段 | 形状 | 含义 |
|---|---:|---|
| backbone_node_layer, backbone_node_position | [L+1] | 每层边界 node 坐标 |
| backbone_node_throughput | [L+1] | node 的 selected-root throughput |
| backbone_node_origin | [L+1,C] | node provenance |
| backbone_step_throughput | [L] | message/residual step throughput |
| backbone_step_is_residual | [L] | 是否为隐式 residual continuation |
| route_node_throughput | [L+1,N] | selected-root-conditioned node throughput |

完整分析 ledgers 保留 head 轴：

| 字段 | 形状 | 含义 |
|---|---:|---|
| route_origin_name | [C] | provenance channel 顺序；evidence 指 world 中全部注册 evidence units 的 lineage |
| route_node_origin | [L+1,N,C] | node provenance |
| route_head_transport | [L,H,P,C] | 每 head 的来源 transport |
| route_head_action | [L,H,P,C] | provenance-conditioned signed grad-message action |
| route_head_direct_evidence | [L,H,P,2] | selected-root direct transport/action；不代表 all-evidence channel |
| route_cross_head_vector_coherence | [L,P] | 合成向量方向一致性 |
| route_cross_head_functional_agreement | [L,P] | signed target action agreement |
| route_evidence_source_reuse | [L,H,N,2] | all-evidence-origin future transport/action |
| route_response_source_reuse | [L,H,N,2] | response-origin future transport/action |

### 6.1 生成时间轴上的重锚定发现

重锚定发现不从稀疏 route edges 或固定-target action 选点。在 edge top-k 前，完整 causal row 被流式
聚合成四个互斥 source-location buckets；持久内存为 \(O(LHPB)\)，不保存 \(O(T^2)\) attention
matrix。对 destination \(q\)，四个 source 集合为：prompt 内 evidence unit、prompt 内非 evidence、
距离超过 `local_window` 的 response、距离不超过 `local_window` 的 response（含 \(s=q\) 对角线）。
它们必须互斥并穷尽 causal sources。

| 字段 | 形状 | 含义 |
|---|---:|---|
| reanchor_bucket_name | [4] | prompt_evidence、other_prompt、remote_response、recent_local |
| reanchor_bucket_attention | [L,H,P,4] | 裁剪前的完整 attention mass |
| reanchor_bucket_transport | [L,H,P,4] | 裁剪前真实 W_O(A V) message transport；不随 flow-signal 改变 |
| reanchor_bucket_downstream_action | [L,H,P,4] | 各桶对当前 artifact 固定 target 的 signed action |
| reanchor_source_position, reanchor_source_unit | [L,H,P,4] | 每桶 transport 最强 source 的 token/unit |
| reanchor_source_attention, reanchor_source_transport | [L,H,P,4] | 该最强 source 的 attention/message transport |
| reanchor_source_downstream_action | [L,H,P,4] | 该最强 source 对固定 target 的 signed action |
| reanchor_anchor_fraction | [L,H,P] | 三个 long-range 桶相对 long-range+recent-local 的份额 |
| reanchor_switch_delta | [L,H,P] | 当前与前一位置的 anchor-fraction 差，仅作几何诊断 |
| reanchor_relative_anchor_rise | [L,H,P] | 相邻生成位置的归一化 long-range rise |
| reanchor_relative_local_fall | [L,H,P] | 相邻生成位置的归一化 recent-local fall |
| reanchor_dominance_flip | [L,H,P] | 前一位置 local-dominant、当前位置 anchor-dominant，且 raw rise/fall 都为正 |
| reanchor_score | [L,H,P] | dominance flip 处两项相对变化的几何均值，其余为 0 |
| reanchor_anchor_source_kind/position/unit | [L,H,P] | transport 最大的 long-range 桶及其 winner source |
| reanchor_anchor_source_transport | [L,H,P] | 上述 winner source 的真实 message transport |
| reanchor_anchor_source_evidence_fraction | [L,H,P] | 所有 winner 在该 layer/source node 的 coverage-pruned evidence lineage；bucket kind 不决定纯度 |
| reanchor_local_source_position/unit | [L,H,P] | recent-local 桶的 transport winner |

候选坐标完全由 full-row true-message transport 冻结，保留 layer 和 head，不平均 head，也不依赖标签、
梯度或逐 route 消融。结构定义要求同一 head 上同时发生 local-dominant→anchor-dominant、raw
long-range rise 与 raw local fall；score 不含 action。每个 head 独立找 temporal peak，时间 NMS 后
按最强单 head score 排名；同一时间的 `support` 只作诊断，不能替代 head 轴。

`reanchor_candidate_*` 是数量受限的展示/排序表，至少分成以下字段组：

| 字段组 | 含义 |
|---|---|
| layer/head/position/source_kind/source_position/source_unit | 事件坐标和当前最强 long-range source |
| score/switch_delta/previous_anchor_fraction/anchor_fraction/relative_anchor_rise/relative_local_fall/dominance_flip/support | 冻结的纯 transport geometry |
| prompt/other_prompt/remote_response/previous_local/local_transport | 事件前后的可审计 transport 分量 |
| source_evidence_fraction | sparse-route source provenance；remote response 只有该值大于零才可称 evidence-bearing candidate |
| long_range/bucket/source/local_downstream_action | 对当前 artifact 固定 target 的分层 signed action 注释 |
| current_target_match | candidate position 是否等于该 artifact 的 query_position |
| long_range/bucket/source_immediate_action | 只在 current_target_match=true 时有限；否则必须为 NaN |
| selected_root_integration_budget/coherence/action | 只在 winner=prompt_evidence 且 winner unit=selected root 时有限 |

只有 `reanchor_candidate_current_target_match=true` 的行，immediate-action fields 才是该行紧随
\(q+1\) token 的局部效用；更早行的 action 只能叫“对 artifact 所审计晚期 target 的 downstream
action”。selected-root integration 不适用于其他 source，必须为 NaN，不能泛化成该事件已经整合。
winner-source evidence fraction 来自 coverage-pruned sparse route provenance；预算外质量保留在
unknown/unobserved 且不重归一。`prompt_evidence`/`other_prompt` 是输入 source unit 的位置身份，经过
layer 更新的 node state 可以混合多种 provenance；所以 fraction 不必分别为 1/0。它不是 full-row
exact lineage 或语义真值。结构候选本身也不预设信息类型。

每个 target artifact 仅覆盖其 teacher-forced prefix 的 represented 窗口；完整 response 事件频率来自
独立 sample scan/cohort。候选表可被截断，dense `reanchor_score` 是 prefix 重算的值；冻结的
`target_reanchor_score` 才是选择事件的 discovery score。两次低精度计算可能改变分数/近并列 winner，
不以数值相等校验 plan identity，也不据此重选 target。teacher forcing 不等于自由生成反馈动力学。

## 7. Residual / attention / MLP diagnostics

AuditPlan 冻结后，native discovery 固定对 selected root 做恰好一次 cut，形成 native/root-cut paired
state；这次 run 只生成 stage/integration diagnostics，不参与 root/hub/corridor selection，也不设置
evaluated/confirmed。stage fields 的差分语义为：

| 字段 | 形状 | 含义 |
|---|---:|---|
| route_stage_position | [A] | 被比较的绝对 positions |
| route_stage_displacement | [L,A,3] | residual input、attention write、MLP write 的 \(\|native-cut\|\) |
| route_stage_action | [L,A,3] | 对应 \(\langle g,d\rangle\) |
| route_module_vector_cosine | [L,A] | attention 与 MLP delta 的向量方向关系 |
| route_module_functional_agreement | [L,A] | attention/MLP signed action agreement |
| route_state_continuity | [L,A] | 相邻层 source-conditioned displacement cosine |
| route_head_integration | [L,H,P,4] | delta-message budget、net norm、coherence、action |
| route_layer_integration | [L,P,4] | residual addition 后的 layer budget/net/coherence/action |

displacement 只说明这个 post-selection operator 改变了 state；gradient/action 只说明当前基点附近对固定 contrast 的局部
sensitivity。二者结合是一阶 screen，仍不等于 exact causal effect，也不能从 residual 中自动排除
词法、语法、实体类型等混杂。

## 8. Confirmation：not run 与 failed 必须区分

route_budget_confirm=false 时，仍会执行第 7 节的一次 selected-root diagnostic cut，但不执行注册的
root/corridor/hub validation ladder。所有 validation outcome
必须有对应的 evaluated 状态；unevaluated 数值应为 NaN 或文档化 sentinel，不能进入均值或被当成零。

| 状态字段 | 含义 |
|---|---|
| selected_root_evaluated | 是否执行 selected-root exact necessity/sufficiency |
| corridor_evaluated | 是否执行 frozen corridor restore/block/mediation |
| carrier_evaluated_count | 实际验证的 frozen hubs 数 |
| full_chain_evaluated | root、corridor、至少一个 hub 是否都已验证 |
| selected_root_confirmed, corridor_confirmed, carrier_any_confirmed, full_chain_confirmed | 仅在对应 evaluated=true 后解释 |

root_value_necessity、root_conditional_sufficiency、root_causal_score 及 corridor/carrier exact effect、
restoration error 只在相应 evaluated=true 时属于 --confirm 验证。root_cut_margin、root_value_effect 与
stage displacement 即使由固定 diagnostic cut 生成，也不得当作 plan score 或 confirmed effect。
默认 discovery 中 root_value_effect = native_margin − root_cut_margin 是有限的诊断量；同时
selected_root_evaluated=false，selected_root_value_necessity、selected_root_conditional_sufficiency、
selected_root_causal_score 与其他未运行的 exact validation fields 必须为 NaN。
confirmation outcome 不得改变 root_unit_id 顺序、
selected_root_unit_id、hub coordinates、corridor coordinates、selection score 或预算。

analysis_stage=discovery 表示 validation ladder 未运行；analysis_stage=confirmed 表示至少一个注册的
exact test 已运行，不代表全部通过。not run、evaluated-but-failed、invalid restoration 是三个不同状态。

## 9. Visualization contract

机制图只读取 label-free artifact 与可选 display tokens：

| 面板 | 最低依赖 | 空状态语义 |
|---|---|---|
| route | frozen plan、edge_*、backbone_*、route_edge_origin、unobserved diagnostics | 无候选或预算受限，不等于无机制 |
| switch timeline | reanchor_score/switch_delta 与 route_row_position | 每个 layer/head 一条轨道；不得求 head/layer mean |
| event triangles | 四桶 totals/winners、candidate fields、token/unit table | 只画桶 winner 轨迹；不得伪装成完整 \(T^2\) matrix |
| integration | stage/integration fields；没有 paired world 时只画 native screen | 非 selected-root-applicable event 必须显示 not applicable |
| confirmation | evaluated flags 与 exact effects | --confirm 未请求时明确显示 not run |

switch timeline 的颜色/强度来自裁剪前四桶 transport geometry；attention 仅作辅助。event triangle
显示 prompt region、recent-local band、事件线、具体 winner token/unit、rise/fall、evidence lineage 和
action。只有 event position=query 才标 immediate next-token action，其他必须写 downstream action。
图不能读取 hallucination labels；不能把 sparse route 画成完整因果电路，或把 layer-unrolled DAG
画成 self-reinforcing feedback loop。

## 10. Controlled pair、2×2 内容契约与 ETCC

pair_schema=1 需要 aligned clean_token_ids/corrupt_token_ids、相同 response、source-unit table、
candidate_unit_id，以及运行前固定的 query/positive/negative/origin。load-time validation 拒绝 future
positions、response 差异和未注册的 prompt 改动。

matched_factorial_schema 是后续 controlled-audit 扩展，不是当前 native subset schema 3 的输出。要给
node/hub 分配操作性的内容标签，该扩展至少保存：

| 字段 | 含义 |
|---|---|
| fact_level, confidence_level | 四个 \(F\times C\) cells 的 0/1 水平 |
| template_id, entity_id, pair_id | 跨模板、跨实体与 matched-cell identity |
| node_activation | 对齐的 candidate node activation |
| fact_contrast | \(\tfrac12[(h_{10}-h_{00})+(h_{11}-h_{01})]\) |
| confidence_contrast | \(\tfrac12[(h_{01}-h_{00})+(h_{11}-h_{10})]\) |
| interaction_did | \((h_{11}-h_{10})-(h_{01}-h_{00})\) |
| nuisance_kind/effect | same-fact paraphrase、grammar/template、random-span、other-prompt controls |
| generalization_split | held-out template 或 entity split |
| swap_direction, answer_effect, confidence_effect | 双向 causal patch 的选择性结果 |

一个 hub 可以同时通过多个 contrast；schema 不提供互斥单标签。fact patch 的注册成功条件是改变答案
事实或 factual answer margin，而不只改变置信表达；confidence patch 应改变 confidence output 并尽量
保持 answer identity/factual margin。native schema 3 的 provenance 只有 evidence/other-prompt/
response/unobserved，不能单独生成 fact 或 confidence label。

etcc_schema=1 保存 controlled clean/corrupt route 与 intervention outcome。它与 native v3 的
observed-token estimand 不同；不得因字段名相似而合并。只有 token-aligned factual swap、same-fact
paraphrase、grammar/template controls 与双向结果都支持时，才可把 candidate carrier 升级为 factual
mediator。

## 11. Manifest、恢复与 label firewall

run_manifest.json 保存可恢复 config、target identity、artifact 相对路径和完成状态。恢复时至少校验第 3
节全部 identity 与第 4.1 节全部预算。改变 --confirm 也应使用独立 output 或显式拒绝复用，因为它改变
artifact stage，尽管它不能改变 AuditPlan selection。

subset-evaluate 必须等 manifest 标记 capture complete 后才加载 labels。它只读 mechanism artifact，
生成独立 mechanism_evaluation.json 与 cohort_summary.json，不回写 target、plan、score 或机制图。
同配置旧 v3 subset 可补建缺失的 scan 并给 sample manifest 增加 scan 路径，冻结 world/targets 不变。
`audit-all --scan-only` 的 analysis scope 为结构扫描；去掉该参数可在同 output 补功能审计。

cohort_schema=1 将完整 scans 与 selected-target 功能观察分开。结构报告覆盖率、每 layer/head 的
四桶份额与切换率；样本内按标注组平均后样本等权，mixed 样本另报组内差。贡献样本 <3 的格子置灰，
没有已实现的显著性/cluster CI。`cohort_*.png` 是结构图，`cohort_functional_*.png` 仅适用于已审计 targets。

## 12. 两阶段检测评价契约

Phase A 保存三个彼此独立、不依赖 exact confirmation 的 raw axes。静态 baseline
`route_origin_competition` 对 query row 的 origin
\(o\)，先在每层计算

\[
s_{o,l}=\sum_h a_{l,h,o},\qquad
b_{o,l}=\sum_h|a_{l,h,o}|,\qquad
Q_o=\sum_l s_{o,l}\frac{|s_{o,l}|}{b_{o,l}+\epsilon},
\]

再定义

\[
\operatorname{route\_origin\_competition}
=\frac{Q_{response}-Q_{evidence}}
{\sum_l b_{response,l}+\sum_l b_{evidence,l}+\epsilon}.
\]

其中 evidence 指 world 中全部注册 evidence units 的 lineage；selected root 不进入该 origin 定义。
其他 prompt token 仍属于 other_prompt，预算外质量仍属于 unobserved，因此 all-evidence 也不意味着
完整观测。该轴的 frozen direction 是 higher → higher hallucination risk。Phase B 才连接
hallucination labels。

另两个 raw axes 为：

| 字段 | 定义 | frozen hallucination-risk 方向 |
|---|---|---|
| temporal_switch_score | 只在 target_reanchor_is_center=true 时，读取冻结的 target_reanchor_score | 中性：同时报告 raw-higher 与 negated orientation |
| temporal_switch_recomputed_score / temporal_switch_score_delta | 相同坐标的 prefix dense score / 相对 discovery 的差值 | 仅复现诊断，不改变冻结事件 |
| target_reanchor_is_center / source_kind | selector 是否记录该 query 为事件中心及其 long-range bucket；只作分母/分层诊断 | 不单独作为连续轴拟合 |
| evidence_adoption | 只在 prompt_evidence event center 上，读取同一 layer/head/query 的 full-row prompt-evidence bucket signed action | lower → higher risk |

`evidence_adoption` 测 observed-token contrast 的局部支持，不是 factual accuracy。`temporal_switch_score`
只测结构切换，不把 action 混入 selector；prompt/other-prompt/remote-response switch 没有被假定为同一
风险方向。窗口上下文与 no-event fallback 的两个 axis 都为 NaN；非 prompt-evidence center 的
evidence_adoption 也为 NaN。三者不得与 route baseline 事后组合。

subset_evaluation_schema=5 对三个连续 raw axes 分别报告 evaluated_targets、positives、prevalence、
hallucination_direction、AUROC 与 AUPRC；中性 temporal axis 还报告
`negated_auroc/negated_auprc`，单类别时均为 null。报告同时保存 event-center/source-kind rate、query
coverage，以及可审计分量
route_evidence_origin_action/signed_sum/absolute_budget/head_agreement 与对应的 route_response_origin_*
字段。route baseline 可使用所有冻结 targets；temporal/adoption 只把该 artifact query 的
prediction-position label 连接到 selection record 指定的 event center；所有窗口上下文、fallback 与
更早 candidate 的 downstream action 必须排除。

每个 task group 还必须包含 `temporal_switch_by_source_kind`，对预定义的
`prompt_evidence/other_prompt/remote_response` 分别报告 evaluated count、raw-higher 及 negated
AUROC/AUPRC。该分层是结构语义的透明拆分，不允许根据 test labels 只挑最好看的 kind。

报告必须保存 `selection_is_not_population_evaluation=true`：reanchor policy 是 event-conditioned
sampling，当前指标不能表示所有自然 response tokens 的 population performance。只有另行预注册并
加入非事件对照/覆盖抽样后，才可定义 population-level detector evaluation。

selected_root_exact_bottleneck 是可缺失且独立的 secondary exact diagnostic：只有
selected_root_evaluated 与 corridor_evaluated 都为 true 且数值有限时才解释；各 effect 先对齐
root_value_effect 的实测方向再取最小值。它不与 all-evidence 主轴混成一个结论，也不进入
route_origin_competition。报告还应包含按 task/source 分组，并在实现后报告按 source 聚类的 bootstrap
interval。

这些指标只检验预注册的预测关联。它们不证明 route grounded、response action 构成 self-reinforcement，
也不允许用 test labels 选择 target、候选、方向、阈值或组合权重。若未来学习组合分数，必须在独立
train/calibration split 冻结后再评估 test。

## 13. 原始 attention / 原生载体审计（rhythm_schema=2）

`attention_rhythm_run` 的每样本主 NPZ 独立于上述旧 target/schema。记 R=N−P，所有曲线保存 R+1 行，
`row_position=P−1..N−1`。预测 token 标签对应曲线前 R 行，FAI 载体标签对应后 R 行；不得按同一个切片混接。
主 NPZ 的 `labels_used_for_capture=false`；标签独立存于 `.labels.npz`。

| 字段 | 轴与含义 |
|---|---|
| `distance`, `waad`, `message_distance`, `message_waad` | `[L,H,R+1]`；原始距离/幅值对照，不是事实贡献 |
| `attention_buckets`, `message_buckets` | `[L,H,R+1,4]`；旧压缩对照，所有来源均参与 |
| `fai`, `message_fai`, `fai_count`, `fai_full_horizon` | FAI 及真实未来窗口分母；缺失未来是 NaN |
| `past_source_position`, `past_source_attention` | 每个 query/head 最强的严格过去来源及系数；不存在为 -1/0 |
| `fai_best_query`, `fai_best_attention` | FAI 窗口中最强的严格后续 query，要求该 query 有 observed q+1；不改变 FAI 本身的完整分母 |
| `distance_mean`, `local_heads`, `global_heads` | 每个 head 的跨度及描述性排名；主同载体矩阵不按这两组筛 head |
| `map_heads`, `map_query_position`, `map_source_position`, `attention_maps` | 明确展示头/行的完整来源列；无 top-k/重归一化，展示窗口不截断上面的曲线 |
| `relay_paths`, `relay_selection_attention_product` | `[E,7]` 为 source/write_layer/write_head/carrier/read_layer/read_head/query；系数乘积仅为展示排序 |

无合法两跳候选时 `relay_paths` 是 `(0,7)`，不强造 hub。只有有候选且开启 detail 时才存在以下载体字段：

| 字段 | 轴与含义 |
|---|---|
| `relay_node_position` | K 个 carrier/query 的实际 token 位置 |
| `relay_residual` | `[L+1,K,D]` 完整残差，最后一行为 final norm 前状态 |
| `relay_post_attention`, `relay_attention_write`, `relay_mlp_write` | `[L,K,D]` 实际模块状态/写入；没有随机投影 |
| `relay_head_code` | `[L,H,K,d_head]` 所有 head 的完整 pre-W_O 净写入；没有跨 head 平均 |
| `relay_edge_index`, `relay_edge_columns` | `[2E,4]`：layer/head/source/query，layer 严格递增的两跳见证 |
| `relay_edge_attention`, `relay_edge_message` | `[2E]` 与 `[2E,D]`，真实系数及 post-W_O 消息 |
| `relay_readout_direction`, `relay_observed_token_ids`, `relay_runner_token_ids` | 每个 K 节点自己的 q+1 / frozen native runner 对照，不是真/假答案标签 |
| `relay_head_margin`, `relay_stage_margin`, `relay_attention_margin`, `relay_mlp_margin` | `[L,H,K]`、`[L+1,K]` 或 `[L,K]`，固定读出方向的带符号记账 |
| `relay_attention_rounding_margin`, `relay_residual_rounding_margin`, `relay_logit_rounding_margin` | 有限精度差单列，不当作信息源或 head 贡献 |

`.audit.npz` 的 `same_carrier_deeper_*` 保留全部物理 head 对，仅 `read_layer > write_layer` 且有完整未来窗口
及 FAI 峰时可定义；同时保留 local/global 的论文式描述对照。`head_pairs_<split>_<task>.npz` 的 `mean_lift`、
`valid_sources`、`positive_source_fraction` 对齐实际 head ID，先在 source 内平均样本，再在 source 间平均，
不常驻 `[sample,head,head]`。无效或未观测的 pair 为 NaN，计数为 0。

`index.json` 保存无标签样本范围与输入根目录；`--phase analyze` 使用该范围，不加载模型/tokenizer。
`summary.json` 明列 `detection_metrics_run=false` 和未检验事项；`gallery.html` 为图/坐标的导航页。
旧 v1 原始曲线仍可离线重分析，但缺少的向量/真实边不能补造；需要新采集时使用独立输出目录。
