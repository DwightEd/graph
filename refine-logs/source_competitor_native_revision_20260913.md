# Source competitor native revision — 2026-09-13

## 范围与结论

这是最小方法扩展的接口复核：未修改实现、未运行 GPU，也不引入新图网络、训练或标签。我阅读了当前 surface-owner 方法说明、surface_verifier 的严格 B/A handoff、CausalOracle、freeze_pools、validate_origin、paired_role、mlp_interaction 和 message_gate_delta。

该扩展值得做，但应称为 **conditional source-occurrence competitor routing test**。它可以在同一个冻结 B/A 事件上检验：一个在当前 owner/条件下被有限判为不适用的 source occurrence，是否比经过验证的正确 occurrence 更能承载原错误分支。它能分别测量 source 内路径与已有 history 路径；它不能证明抽象 owner pointer、唯一原因、关系影响或错误聚合。

当前 paired_role 只实施 history→correct-source 的转移，不能回答 source A 与 source B 的竞争。直接把 B 塞进 paired_role 也不够：NativeGate.strength 是整个 group 的标量，两个方向移走的注意力质量可以不同。因此下列三个接口合同是实施前的 Required。

## Required 1 — 冻结、完整保留的 competitor roster

只在 surface_verifier 的五项严格检查得到 conditional_contrast_available 后构造 source_competitor_pool@1。输入必须冻结：

- B/A contrast、target slot、correct occurrence A、其 response/source graph SHA 和 tokenizer/observer identity；
- **全部** source occurrences，而不是 surface_owner 的 top-k、masked owner ranking 或之后 native 搜索的子集；
- 每个 occurrence 的 immutable ID、parent/record ID、raw/display span、surface type、source token span/keys、source graph seal 与 text SHA。

候选资格只能是：

    occurrence_id != A.occurrence_id
    surface_type == target.surface_type
    display_surface bytes == original erroneous target display_surface bytes
    source span/parent/record mapping is valid

这里的 display equality 是受控 competitor 构造，不能再进入 masked owner 主排序或用于报告 owner-locator 准确率。不得大小写折叠、单位换算、范围拆分、引用改写或把相同 literal 的不同 record 合并。None、unknown、bool、未闭合 quoted/range 等 surface policy 已关闭的值也保持关闭并计入状态。

roster 保存所有同值 occurrence，包括同 literal、不同 record 的每一个成员，以及每一条拒绝原因。若为限制有限 reader 调用而只能检查 K 个成员，则 K、稳定的 occurrence_id 排序、完整未检查尾部和 roster SHA 必须在任何 C/readout/native forward 之前冻结；不得因 I 失败、控制缺失或 native 无效而补搜。更保守而清楚的首次实现是对完整 roster 都作一次有限 I 检查，超预算就整体标为 competitor_pool_budget_unresolved，而非挑一个“有希望”的 B。

对每个冻结成员独立执行有限 SOURCE/OCCURRENCE 请求，完整 source 和 target 的 owner/条件可见，但该请求不得收到效果、masked embedding、owner score、gold 或“它应为错误 owner”的提示。只有 I >= .8 的成员才是 B_i；S/C/U、reader failure、坐标不一致或未检查都不是 B_i。这个 I 是“该 occurrence 对当前 owner/event/condition 不适用”的条件预测，仍非 ground truth。

最小输出：

    source_competitor_pool = {
      schema, bridge_sha256, correct_occurrence_id,
      target_original_display_sha256, frozen_roster, roster_sha256,
      finite_I_receipts, eligible_B_ids, rejected_or_unchecked_ids,
      selection_policy, native_budget_reservation,
      labels_used: false, target_conditioned_control: true
    }

## Required 2 — 真正等质量的 source A↔B 路由算子

每个 B_i 单独试验；不要将不同 record 的 B_i 合成一个 destination 后再宣称识别了某个 owner。对固定的同一 queries × layers，固定 source key domain：

    U_i = A.keys ∪ B_i.keys

并做两个镜像 route gates：

- A→B_i：从 A 的当前 attention mass 转到 B_i；
- B_i→A：从 B_i 的当前 attention mass 转到 A。

两次必须在每个可见 (layer, head, query) 具有同一绝对转移量：

    rho[l,h,q] = gamma * min(mass_A[l,h,q], mass_Bi[l,h,q])

其中 gamma 是预冻结的 1 或固定半强度。对应的 selected attenuation 是 1 - rho / mass_selected，而不是两次共用一个标量 strength。每个 row 的 U 内总 mass、U 外 weights、值 V、其它角色和其它 query 都必须保持不变；保存逐 row mass_A、mass_B、rho、mass_balance_error。任何真正 future edge 的权重仍须严格为零。某些早 query 看不见较晚 B_i 是合法的因果子图；这些 row 的 rho=0。若所有 row 均 rho=0、有效质量/可见 rows 未达到预冻结的最小条件，结果是 source_competitor_no_common_visible_mass，不是零效应。

这不能由目前的 CausalOracle.gates(... strength=float) 表达。需要一个只允许上述冻结 per-head/per-query transfer plan 的 route gate 参数，或一个等价的已验证 operator；不能以全组平均 oracle.mass 配平。operator 必须继续满足 native 的 GQA、动态 current V、full-shape sham、前缀不变、future-mask、dtype/finite 与 hook 清理合同。

以现有定义 delta = F_base - F_gated，预注册 directional witness 是：

    delta(A→B_i) <= -0.5
    delta(B_i→A) >= +0.5
    swap_contrast_i = delta(B_i→A) - delta(A→B_i) >= 1.0

并要求 exact sham、repeat、half-strength 同向且不超过 full 的两倍。阈值沿用当前 native certificate；不要依效果大小选 B_i、layer、query 或 strength。

每个 B_i 还需要两条在 roster 同时冻结的 source destination controls U_1/U_2：不同于 A 和全部 eligible B，类型、key 长度、可见性及 **逐 row 可转移质量 profile** 与 B_i 匹配，并各自 I>=.8。测量对应 A↔U_j 镜像 swap；只有 swap_contrast_i 超过两条控制的绝对值至少既有的 selectivity margin 时，才是 position-selective，而不是“把注意力移到任意 source 位置都有影响”。控制不足、全 future、I/U 或预算失败均应保留在分母且禁止 backfill。

## Required 3 — 固定 V 地址的 raw-origin 对照与归因边界

对每个获得 position witness 的 B_i，A 与 B_i 的 raw input scaling 必须使用同一 recipient 地址和同一 query/layer：

    V_address_i = sorted(A.keys ∪ B_i.keys)
    origin A.keys   -> V_address_i
    origin B_i.keys -> V_address_i

每一路均用现有 donor provenance（branch IDs、model identity/dtype、ordered keys、input scale、array digest）和其自己的预冻结、语义过滤后的 raw-origin controls。不得为 B_i 通过后再新建 origin pool，也不得把 A 的 origin controls 偷换给 B_i。两路都应有 exact donor sham、repeat、half、两 controls，并保持方向：

    origin_delta(A)   <= -0.5
    origin_delta(B_i) >= +0.5

这说明 A/B_i 的原始输入可经同一冻结 V-message 地址影响相反的 B/A event preference。它本身仍是 **mediated source effect**，不是 owner/address 的独立证明；只有和 Required 2 的同范围、等质量 route witness 合并时，才支持“在此固定 B/A contrast 中向不适用 source occurrence 的 conditional routing”。

现有 mediated 能捕获指定 group.keys 的 donor V，但实现时必须显式把 recipient group key domain 固定为 V_address_i，不能分别用 A-only/B-only receiver keys 后把两个不同地址的效果相减。

## 可报告的观察与禁止升级的解释

| 观察 | 可报告 | 不能报告 |
| --- | --- | --- |
| 全部 finite 检查、I、等质量 A↔B 路由、两个 route controls、同 V-address 双 raw-origin 通过 | 条件性 source wrong-occurrence routing witness | 抽象 owner pointer、唯一错误原因、模型已作错误正误决策 |
| 仅 B_i raw-origin 通过 | 不适用 source occurrence 的 mediated target dependence | routing 或 owner misbinding |
| route swap 通过而 raw-origin 未通过/未测 | 固定当前 V 下的 destination sensitivity | 该 raw source 输入造成效果 |
| A↔B 有效但 source controls 不通过 | 未受控 source destination effect | selective wrong-owner routing |
| B_i 的 I 未达 .8、U、缺失或坐标/hash 失败 | competitor unresolved | B_i 是错误 owner |
| source-competitor 与 history pairedR 都通过 | 同一 B/A 上有 source-internal 与 tested-history 的并存路径 | source 而非 history、唯一原因 |
| source-competitor 通过而 history 未通过 | 已认证 source-internal路径；tested history route 未认证 | history 没有作用或 source 独占 |
| 只有 MLP four-state interaction | 在已认证 route witness 条件下的 MLP modulation diagnostic | MLP 聚合了错误、MLP 单独导致错误 |

要严格区分“没有 history certificate”与“排除了 history reuse”：后者需要独立的、预冻结且同 F 的 history 反向证据；已有 paired_role 失败只是不确定。若两条路径都阳性，输出 mixed_source_competitor_and_history_paths。

## MLP 的最小定位

不需要增加新的 MLP 机制。可在 source A↔B_i route witness 已成立后，以同一 frozen MLP group 测量四状态 interaction，并把 source route contrast 作为 evidence arm。它最多说明该已经定义的 source-competitor contrast 受 MLP 条件调制。若 MLP scope 比 route scope 宽、context partition 主导、swap 未过或 raw origin 未过，维持 broad_MLP_event_interaction/unresolved，绝不可称“MLP 聚合错误”。

## 必要反例与回归合同

实现前应有 CPU/native-tiny 回归覆盖：

1. 同 display、同 type、不同 record 的两个 B_i 都保留；一个 I failure 不让另一个在事后替补，也不能被合并。
2. 候选值相似但 range/unit/quoted span 不 byte-exact 时不能进入 roster；None/unknown 不伪造 literal competitor。
3. B_i 位于部分早 query 的 future：这些权重严格零、rho 为零，但晚 query 的有效交换仍可测；全 future 则拒绝为 no-op。
4. A/B baseline mass 不等时，镜像 route 的每个有效 head/query 转移质量相等，U 内总 mass 保留，旧 scalar-strength 实现必须被测试拒绝。
5. route control、origin control、finite I receipt、graph/source text SHA、token span、model/reader identity 任一替换或缓存损坏都拒绝。
6. 仅 raw-origin 或仅 MLP interaction 通过时，输出状态不能含 wrong_owner_routing。
7. 已有 history pairedR 阳性与 source competitor 阳性同时出现时，输出为 mixed，不能按更大 effect 挑一个叙事。

该扩展不需要 GNN、数据标签训练或 free-QA。其价值是把现有严格 surface B/A 的 source 端从“正确证据 vs history”扩展到“正确 occurrence vs 冻结的不适用同值 occurrence”，并清楚保留无法区分、无共同可见质量、控制不足和预算未解决的分母。


## Targeted re-review — accepted one-way destination contrast

The proposed simplification is valid and supersedes the mirror requirement above.

The primary estimand may be a **one-way, single-layer source-B destination contrast**, rather than A↔B_i symmetry. For a fixed eligible B_i, a frozen query set Q and one frozen layer L, use one identical source-role key domain D that includes B_i, A, and both controls:

    B_i → A
    B_i → U_1
    B_i → U_2

Every branch has the same selected B_i keys, domain D, Q, L, strength and original forward input. The existing route operator then removes exactly the same current B_i attention mass in every head/query row and redistributes it only across the specified destination's current weights. Its mass-preservation contract is the desired estimand: same B, same outgoing mass, only destination changes. Running each measurement from the untouched baseline and restricting it to one layer avoids a preceding gated layer changing later-layer current mass.

This avoids a real power failure in the earlier mirror proposal: rho=min(mass_A,mass_B) collapses when a wrong routing state has almost no A mass, even when B has substantial removable mass. A→B is therefore an optional reverse-direction diagnostic, never a prerequisite for the primary source-competitor witness.

### Updated Required 2

Freeze before any B_i route effect:

- Q and L, the identical domain D, B_i selected keys, route strength, and the order of all three destination trials;
- A and U_1/U_2 as disjoint destinations; each control is source-role, I>=.8, not A or any eligible B, and structurally matched to A on key count, visibility and **destination receiving mass/profile at the fixed L×Q**;
- the complete per-head/per-query B_i selected mass and all candidate destination receiving masses, recorded from the common ungated baseline.

A row with B_i incoming mass nonzero but a chosen destination receiving mass zero is invalid under the current operator and must produce unresolved rather than fabricated mass. Mixed causal visibility remains valid: future B/destination edges stay exactly zero, and only rows on which the operation is defined contribute. If no defined B_i→A row remains, return source_competitor_no_common_visible_mass. The implementation must preserve D's per-row mass, leave D-external weights/values/roles/queries untouched, retain GQA/current-V/future-mask/full-sham contracts, and reject any failure of those checks.

With delta = F_base - F_gated, the preregistered primary pattern is:

    delta(B_i→A) >= +0.5
    repair_selectivity_i =
       delta(B_i→A) - max(abs(delta(B_i→U_1)), abs(delta(B_i→U_2))) >= .25

It also needs exact sham, repeat, and a fixed half-strength trial with the existing directional/stability rules. This says relocating B_i's **actual current outgoing mass** specifically to the correct A occurrence repairs the B-vs-A preference more than relocating that same mass to matched inapplicable source destinations.

The earlier raw-origin conditions remain Required: B_i raw-origin through the fixed union V address supports B (delta>=+.5), A raw-origin through that same address opposes B (delta<=-.5), with independently frozen origin controls. Together they provide a conditional occurrence-level witness: B_i is semantically inapplicable, carries a mediated B-supporting path, and its existing route mass has an A-specific repair. They do not identify an abstract owner representation; the B value/payload itself can still be the transported content.

### What mirror A→B would and would not add

No material confound is removed only by requiring A→B:

- Generic removal of B is held constant by B→A versus B→U_j.
- Generic destination sensitivity is bounded by the two matched U_j controls.
- Unequal B/A baseline mass is irrelevant because selected B mass, not A mass, is held fixed.
- Source payload versus abstract address remains unresolved even if A→B passes; the existing fixed-V raw-origin limitation still applies.

A→B may be logged as an optional robustness observation if A has sufficient visible mass under a predeclared rule. Its failure, absence, or no-common-mass state cannot downgrade an otherwise complete one-way witness. Conversely, B→A without B raw-origin, A raw-origin, finite I, controls, or the frozen input/identity/coordinate chain remains only an effect and cannot receive the wrong-owner-routing label.

History remains separately measured. A one-way witness plus a failed history certificate means tested-history-unresolved, not history-excluded; simultaneous history pairedR evidence still requires the mixed-source-and-history output.

