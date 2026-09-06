# Mechanism-audit data contract

## 1. Coordinates

所有数组使用绝对 token position。对一个 predictor (q)，prediction position 是 (q+1)，
causal prefix 长度为 (N=q+1)。

| 轴 | 含义 |
|---|---|
| `L` | decoder layers |
| `H` | query heads；始终保留，不平均 |
| `N` | 当前 target 的 causal source positions |
| `P` | capture 中表示的 destination rows |
| `E` | artifact 中保存的高 root-throughput message edges |
| `A` | source-cut stage positions |
| `U` | source units |
| `R` | root candidates |
| `K` | 被精确测试的 carriers |
| `4` | `evidence / other_prompt / response / unobserved` provenance |

NPZ 使用 `allow_pickle=False` 可读取的标量、字符串和数值数组。hallucination/correctness label 不进入
world 或 audit artifact。

## 2. Native world

`native_world_schema=1` 是 label-free target contract：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `sample_id`, `tokenizer_id` | scalar | 安全样本 ID 与 tokenizer |
| `token_ids` | `[N_full]` | prompt 与 teacher-forced response |
| `response_start` | scalar | 第一个 response token |
| `token_unit_id` | `[N_full-1]` | 每个 source position 的 unit |
| `unit_name`, `unit_kind` | `[U]` | passage/sentence/field/response 表 |
| `evidence_unit_id` | `[R]` | 可作为 source root 的 units |
| `query_position` | `[T]` | 冻结 predictors |
| `positive_token_id` | `[T]` | observed token |
| `negative_token_id` | `[T]` | native run 中冻结的 runner |
| `contrast_origin` | `[T]` | target 选择与 contrast 语义 |

## 3. Native mechanism artifact

`subset_audit_schema=2`、`method_version=head-resolved-native-route/2`。每个 target 一个 NPZ。

### 3.1 Identity and claim coordinates

- dataset/sample/source/split/task/model/tokenizer identity；
- `response_start`, `query_position`, `prediction_position`；
- `positive_token_id`, `negative_token_id`, `target_token_ids`, `contrast_origin`；
- `flow_signal`, `edge_coverage`, `carrier_scope`, `query_chunk`, `local_window`；
- `token_ids`, `token_unit_id`, `unit_name`, `unit_kind`, `evidence_unit_id`。

这里的语义是 observed target 对 selected source Value-message cut 的依赖；schema 2 不新增 factual
correctness 声明。`carrier_scope` 默认是 `all`；缩减 scope 时，未表示的 prompt destination 在
下一层归入 `unobserved` provenance，不能把其初始 source 身份解释为跨层延续。

### 3.2 Root and exact intervention outcomes

| 字段 | 形状 | 含义 |
|---|---:|---|
| `root_unit_id` | `[R]` | candidate unit IDs |
| `root_route_mass`, `root_functional_score` | `[R]` | route 与一阶功能 screen |
| `root_value_necessity` | `[R]` | 删除该 unit Value messages 的 margin effect |
| `root_conditional_sufficiency` | `[R]` | 只保留该 unit 的 conditional rescue |
| `root_causal_score`, `root_evaluated` | `[R]` | 双向门槛和是否执行 rerun |
| `selected_root_unit_id`, `selected_root_confirmed` | scalar | 最终 selected root |
| `native_margin`, `root_cut_margin`, `root_value_effect` | scalar | native/source-cut 对照 |
| `corridor_*` | scalar | necessity、conditional/blocked/mediated rescue、restoration error/validity |
| `corridor_confirmed` | scalar bool | root、方向、mediation、restoration 共同通过 |

`carrier_*[K]` 保留 layer、position、unit、route throughput、state delta、target score、necessity、
rescue、block、mediated rescue、tolerance 和 `carrier_confirmed`。`full_chain_confirmed` 要求 corridor
与至少一个 carrier 同时通过；单个 carrier 正结果不是完整 source→target chain。

### 3.3 Sparse route edges

下列字段第一维均为 `E`，行序一一对应：

| 字段 | 含义 |
|---|---|
| `edge_layer`, `edge_head`, `edge_source`, `edge_target` | 精确 message 坐标 |
| `edge_source_unit` | source position 所属 unit |
| `edge_attention_native/root_cut` | 两个 world 中的 native gate |
| `edge_native_functional_score` | native gradient 与 native true message 的有符号内积 |
| `edge_root_cut_functional_score` | 冻结 native gradient 对 root-cut message 的投影 |
| `edge_native/root_cut/delta_message_norm` | matching head `W_O` 后的 message norms |
| `edge_root_throughput` | selected-root-conditioned candidate throughput |
| `edge_source_root_lineage_fraction` | source node 在 norm-based ledger 中的 selected-root lineage share |
| `edge_root_lineage_action` | 上述 share × native signed functional score；仅为一阶 route screen |
| `edge_on_backbone` | 该 message edge 是否属于保存的 connected backbone |
| `route_edge_origin` | `[E,4]`，每条边携带的 provenance |

artifact 保存 top root-throughput edges 供图使用；完整 head/position ledgers 单独保存，不能把 sparse
edge 表中未显示的边解释为不存在。保存预算先容纳 backbone 的 message edges，再以 throughput
补充 marginal edges。

### 3.4 Connected route backbone

backbone 是 unrolled graph 中从 selected-root prompt position 到 `(layer_count, query_position)` 的
一条 widest candidate path；逐层在 message edge 与隐式 residual continuation 之间选择，使路径
最小 throughput 最大。它保证图中路径连通，但仍是 norm-based routing candidate，不是 exact
causal chain。

| 字段 | 形状 | 含义 |
|---|---:|---|
| `layer_count` | scalar | audited decoder layer 数；target node 位于此 layer boundary |
| `backbone_node_layer`, `backbone_node_position` | `[L+1]` | backbone 的逐层 node 坐标 |
| `backbone_node_throughput` | `[L+1]` | 对应 node 的 selected-root throughput |
| `backbone_node_origin` | `[L+1,4]` | 对应 node 的 provenance ledger |
| `backbone_step_throughput` | `[L]` | 每个 message/residual step 的 throughput |
| `backbone_step_is_residual` | `[L]` | 该 step 是否为隐式 residual continuation |
| `route_node_throughput` | `[L+1,N]` | 全部 node 的 selected-root throughput，供 backbone/carrier 显示 |

### 3.5 Full route ledgers

| 字段 | 形状 | 含义 |
|---|---:|---|
| `route_origin_name` | `[4]` | provenance 通道顺序 |
| `route_node_origin` | `[L+1,N,4]` | 每层边界的 node provenance |
| `route_row_position` | `[P]` | destination slot 到绝对 position |
| `route_head_transport` | `[L,H,P,4]` | 每 head 的来源 transport |
| `route_head_action` | `[L,H,P,4]` | 每 head 的 provenance-conditioned signed action |
| `route_head_direct_evidence` | `[L,H,P,2]` | direct-root transport/action |
| `route_head_local_response` | `[L,H,P,2]` | local response-origin transport/action |
| `route_head_integration` | `[L,H,P,4]` | delta-message budget、net norm、coherence、action |
| `route_layer_integration` | `[L,P,4]` | residual addition 后的 layer budget/net/coherence/action |
| `route_cross_head_vector_coherence` | `[L,P]` | 合成向量一致性 |
| `route_cross_head_functional_agreement` | `[L,P]` | 有符号 target action 一致性 |
| `route_head_backward_distance` | `[L,H,P]` | transport-weighted source distance |
| `route_head_span` | `[L,H]` | response rows 上的 head span；仅用于逐 head 分组 |
| `route_evidence_source_reuse` | `[L,H,N,2]` | evidence-origin future transport/action |
| `route_response_source_reuse` | `[L,H,N,2]` | response-origin future transport/action |

`route_event_*[J]` 是 hub-aware selected-root action 的逐 head 局部峰：layer/head/position、score、
evidence transport/action、`direct_fraction` 与 local-response transport/action。event 是 intervention
candidate，不是因果结论。

### 3.6 Residual / attention / MLP integration

| 字段 | 形状 | 含义 |
|---|---:|---|
| `route_stage_position` | `[A]` | 被比较的位置 |
| `route_stage_displacement` | `[L,A,3]` | residual input、attention write、MLP write 的 `||native-cut||` |
| `route_stage_action` | `[L,A,3]` | 对应 gradient-dot-delta |
| `route_module_vector_cosine` | `[L,A]` | attention 与 MLP delta 的方向关系 |
| `route_module_functional_agreement` | `[L,A]` | attention/MLP signed action 的一致程度 |
| `route_state_continuity` | `[L,A]` | 相邻层 source-cut residual delta cosine |

displacement 只说明 selected source cut 改变了状态；action 是对固定 target margin 的局部一阶
作用；只有 exact rerun fields 是当前 operator 下的因果确认。两者都不证明存在可分离的事实表征。

## 4. Visualization contract

`mechanism_plot.save_mechanism_figure` 只读取 schema-2 mechanism fields 和可选 display tokens，不接受
labels。四个面板至少依赖：

- route：`edge_*`, `edge_root_throughput`, `edge_root_lineage_action`, `edge_on_backbone`,
  `backbone_*`, `route_edge_origin`, `route_node_throughput`；
- heads：`route_head_transport`, `route_head_action`, `route_row_position`；
- integration：`route_head/layer_integration`, `route_stage_*`, module/continuity fields；
- intervention：root/corridor/carrier exact effects。

## 5. Controlled pair and ETCC output

`pair_schema=1` 需要 aligned `clean_token_ids/corrupt_token_ids`、相同 response、source-unit table、
`candidate_unit_id`，以及固定的 query/positive/negative/origin。load-time validation 拒绝 future
positions、response 差异和未注册的 prompt 改动。

`etcc_schema=1` 保存完整 clean/corrupt sparse edge table，包括 pre-`W_O` clean/corrupt `AV`
codes、可选 post-`W_O` vectors、selector/content decomposition、residual transition、reverse/root
throughput、stage deltas，以及 root/corridor/carrier exact effects。它与 native schema 2 是不同
estimand；不得仅因字段名称相似而合并统计。

## 6. Manifest and label firewall

`run_manifest.json` 只记录可恢复的 config、冻结 selection、world/audit 相对路径和完成状态。同一
output 只恢复完全相同的 scientific config；改变 source、model、target policy、coverage、window
或 intervention limits 应使用新 output。

`subset-evaluate` 要求 capture 完成后才加载 labels，并生成独立 report。它不会修改 native world、
mechanism NPZ 或机制图。
