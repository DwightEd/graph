# v3 机制审计预注册

## 1. 状态

本文注册 v3 的研究问题、候选选择、可选因果确认和 hallucination 评价。文中明确区分：

- **measurement**：代码直接保存的量；
- **candidate**：由冻结规则选出的 structural reanchor、root、hub 或 corridor；
- **confirmed native effect**：--confirm 下对 native source operator 的因果结果；
- **factual mediator**：还通过 matched factual pair 的更强结论；
- **hypothesis**：尚未由当前实验验证的机制或检测关系。

native teacher-forced audit 不知道 observed token 是否正确。它最多研究“什么内部 route 支持这个已
出现的 token”。未完成 matched factual contrast 时不得写 grounded route；未完成 free-running
intervention 时不得写 hallucination attractor 或 self-reinforcing loop。

历史状态必须明确：此前没有一个版本完整审计“全 response 时间轴的逐-head local→prompt/remote
切换 → 具体 source → 接纳 → 下一 token”这条链。早期三角图包含 route change/prompt revisit，但
混合或平均 heads，且没有同时要求 local 下降、long-range 上升与 adoption；后期版本加入真实 message、
gradient、hub 与 intervention，却从固定 target 出发，未先在时间轴上发现切换。v3 首次把这条测量
链放进同一 artifact；**首次实现不等于已验证机制**。

## 2. 预注册顺序

每个样本/target 严格按以下顺序处理：

1. 在不读取 hallucination labels 的情况下选择 sample，冻结 source-unit table、response horizon、
   local window 与 target-row 预算；
2. 若使用 reanchor policy，扫描当前 clean model 的 full causal rows，只用逐-head true-message
   transport 冻结结构事件；
3. 按 `reanchor` 或 `reanchor-window` 分配 target rows，冻结 query、observed token、runner 与 target
   margin；无事件的 evenly-spaced fallback 必须写入 origin；
4. 冻结 destination scope、rows/edges/roots/hubs/corridor 预算及稳定 tie-break；
5. 对每个 target 捕获逐 head 的真实 message、route mass 与 target gradient；
6. 只用注册候选分数生成并冻结 AuditPlan；
7. 对 frozen selected root 固定做一次 cut，生成 native/root-cut integration ledger；该结果不参与选择，
   也不构成 confirmation；
8. 若指定 `--confirm`，只对 frozen plan 增加 necessity/sufficiency/restore/block validation；
9. 完成全部 capture 后，评价进程才可读取 labels。

AuditPlan 的 identity 必须覆盖 target identity、model/tokenizer、source units、represented rows、
预算、候选坐标、rank 和选择分数。confirmation_requested 与 confirmation outcomes 不得进入计划
hash。exact intervention 不能重选 root、hub 或 corridor；失败候选也不能由次名候选自动替换。
第 7 步是固定的 post-selection paired diagnostic，不得根据 root-cut margin 或 residual difference 回头
修改 AuditPlan。

provenance 与 root plan 是两个契约：EVIDENCE channel 从输入层起包含 world 中全部 evidence units；
selected root 仅用于 root-conditioned throughput/backbone、direct-root fraction 和第 7 步的 paired
integration。主 route-origin competition 不得退化成 selected-root 与 response 的比较。

## 3. 三类常见代理量为何不够

### 3.1 Attention

attention weight 只表示 QK gate。逐 head 真正写入 residual 的是

\[
m^{l,h}_{j\to i}
=W_O^{l,h}\left(A^{l,h}_{i,j}W_V^{l,g(h)}
\operatorname{LN}(r_{l,j})\right).
\]

相同 attention pattern 可以经 \(W_V\) 与 \(W_O\) 写入不同甚至相反的方向。因此 raw attention
只作为对照，不用于 v3 主选择；任何主分析都必须保留 (layer, head, source, destination)，不得
对 head activation 或 attention 求平均。

### 3.2 Residual difference

\[
d_v=h_v^{clean}-h_v^{counterfactual}
\]

只测 operator-specific displacement。它可能含事实、词法、语法、长度、实体类型及非线性传播
差异。高 \(\|d_v\|\) 不说明 target 读取了该差异，也不等于“证据语义向量”。

### 3.3 Gradient

\[
g_v=\nabla_{h_v}F_t
\]

只测当前基点、当前 contrast 的局部 sensitivity。它不说明实际 state 是否携带 source-specific
内容，也可能受 saturation、抵消和高阶交互影响。

两者结合的 \(\langle g_v,d_v\rangle\)，以及 discovery 使用的
\(\langle\nabla_{m_e}F_t,m_e\rangle\)，都是 AtP/EAP 风格的一阶 screen，不是 exact causal
effect。它们的用途是缩小候选集，而不是给每条 route 下因果结论。

## 4. 时间轴重锚定的注册量

对每个 `(layer, head, response predictor q)`，完整 causal source row 必须在任何 sparse edge top-k
之前聚合为四个互斥位置桶：

1. `prompt_evidence`：prompt 内、属于预注册 evidence unit；
2. `other_prompt`：prompt 内、不是 evidence unit；
3. `remote_response`：response 内且距离大于 `local_window`；
4. `recent_local`：response 内且距离不大于 `local_window`，包含对角线。

必须同时保存每桶 raw attention mass、真实 \(W_O(A V)\) message-transport magnitude、每桶最强
source position/unit。attention 仅是辅助路由图；结构 selector 只能用 message transport。四桶聚合先于
稀疏裁剪，避免不同时间的 retained-mass 变化伪造切换。实现只持久化
`[layer, head, represented row, 4]` 总量与 winner；不能把它描述成保存完整 \(T^2\) triangle。

前三桶合称 long-range anchor。候选必须在同一 head 上满足：上一位置 recent-local 占优，当前位置
long-range 占优，raw anchor transport 上升，且 raw local transport 下降。分数只由归一化 anchor rise
与 local fall 的几何均值构成，不含 target action、gradient、label 或 ablation。每个 head 独立找 peak；
时间 NMS 后按最强单 head 分数排序，head support 只作诊断。

冻结坐标后才附加 source evidence lineage、downstream action 与 selected-root integration：

- `prompt_evidence`、`other_prompt` 与 `remote_response` 必须分开；较早 response 只有 lineage 大于零
  时才可称 evidence-bearing relay candidate。该 lineage 来自 coverage-pruned sparse route
  provenance；预算外质量保持 unknown/unobserved，不重归一，不能写成 full-row exact lineage 或
  语义真值。bucket kind 只是输入 source position/unit 身份；经过 layer 更新的 node state 可以混合
  provenance，不能把 prompt-evidence/other-prompt bucket 名当成 state 纯度。
- 只有 candidate position 等于当前 artifact `query_position`，其 action 才是该事件自己的 immediate
  next-token action；更早位置的 action 只是对晚期 audited target 的 downstream action。
- selected-root integration 只有在 winner 是 `prompt_evidence` 且 winner unit 等于 selected root 时
  才适用；其余事件不得根据这个 operator 声称“已接纳”。

`reanchor` 选择最多 \(N\) 个事件中心；`reanchor-window` 把 \(N\) 作为 row 硬预算，最多先取
\(\lceil N/3\rceil\) 个中心再补 \(-1/+1\)。完全无事件才 evenly-spaced fallback。target selector 使用
当前 clean model 重算的 full rows，不接受外部 sparse attention cache 的截断值。

## 5. 固定 target route 候选的注册量

对固定 target \(F_t=z_q(a)-z_q(b)\)，每条 message 保存：

1. route transport 与 target-connected throughput；
2. signed grad-message
   \[
   a_e=\langle\nabla_{m_e}F_t,m_e\rangle;
   \]
3. head identity 与正、负 action。

候选 group \(G\) 的 functional agreement 为

\[
A_G=\frac{|\sum_h a_{G,h}|}{\sum_h|a_{G,h}|+\epsilon}.
\]

定义 route-weighted signed action \(s_G=\sum_h a_{G,h}\) 与 absolute action budget
\(B_G=\sum_h|a_{G,h}|\)。冻结的 selection magnitude 注册为

\[
S_G^{mag}=M_G\,B_G\,A_G.
\]

其中 \(M_G\) 是 route mass，\(A_G=|s_G|/(B_G+\epsilon)\)。因此分数是非负 magnitude；agreement
已经把 absolute budget 收缩为净 signed-action 幅度，不能再对 \(s_G\) 重复乘一次。另存
signed_action \(s_G\) 作为方向；支持 observed/evidence contrast 的后续命名还要求 \(s_G>0\)。所有
逐 head action 必须同时保存。

- **Root**：在预注册 source units 上计算 \(S_G^{mag}\)，保留固定数量及稳定 rank。
- **Hub**：内部 layer-position 节点必须同时有 root-side inbound 与 target-side outbound；用两侧
  absolute budget bottleneck × agreement bottleneck × route mass 排序，再做固定 NMS；方向另存。
- **Corridor**：先保留 selected root 到 target 的 connected backbone，再在硬预算内补充高
  \(S_e^{mag}\) message；单条逐-head edge 有 \(B_e=|a_e|,A_e=1\)，residual continuation 可跨越
  没有显式 message 的层。

这些规则回答“先检查哪里”，不是“哪里已经被证明重要”。

## 6. 预算与缺失质量

默认 carrier_scope=response，只展开 response-side destination rows；prompt evidence token 仍可
作为这些 rows 的 source。搜索 prompt 内部 hub 必须显式设置 carrier_scope=all，并满足同一个 rows
预算。

预注册默认预算为：

| 项目 | 默认值 | 语义 |
|---|---:|---|
| target_policy | reanchor | 用 clean full-row transport 选事件中心 |
| targets_per_sample | 1 | target-row 硬预算；事件窗口审计建议 3 |
| max_response_tokens | 128 | 时间轴 pilot 的 response horizon |
| query_chunk | 8 | full-row 临时 query chunk；不改变结构定义 |
| local_window | 10 | recent-local response 距离上限，包含对角线 |
| max_route_rows | 256 | represented destination rows 硬上限 |
| edges_per_head | 2 | capture 各取 transport top-k 与 absolute-functional top-k，最坏并集 2k；corridor 再硬限 k |
| root_candidates | 4 | root 候选上限 |
| hub_candidates | 8 | hub 候选上限 |
| corridor_edges | 64 | plan 中显式 corridor message 上限 |

full-row 时间扫描必须在 top-k 前完成四桶聚合，常驻只保留 \(O(LHP\cdot4)\) totals/winners；chunk
内仍临时计算 causal sources，因此 OOM 时应优先缩短 response horizon 或降低 query chunk，不能以
稀疏 cache 替代注册 selector。预算超限必须报错或要求新配置，不能静默扩大。capture 的两个 top-k 分支用于同时保留高 transport
与低质量但高 target action 的边，不是对 head 求平均；frozen corridor 才执行每 head-row 的 k 上限。
edge coverage 未达标时，represented full-row 中被裁剪的 mass 进入 unobserved，且不重归一到
retained edges。scope 外 prompt destinations 没有 attention row，只沿 residual identity 保留已有
all-evidence/other-prompt provenance；这让后层 represented row 仍能读取 prompt state，但没有审计
prompt 内部更新。高 unobserved fraction 或大量 unrepresented rows 都会降低 completeness，不能据
稀疏图断言“其他路径不存在”。

## 7. 从普遍模式到可验证机制

### 7.1 Preplan-and-Anchor 可迁移的发现范式与边界

[*Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy
Optimization*](https://arxiv.org/html/2510.13554v2) 不是直接遍历出一组“显著 head”就命名机制。
它先按每个 `(layer, head)` 在 response 上的 attention-weighted mean backward distance 排序，取底/顶
30% 作为 local/global head 组；然后**在组内平均 attention**发现 local sawtooth 和 global anchor，定义
WAAD/FAI，检验两者的时序 coupling、与 entropy/receiver-head 信号的共现及随机基线，再用高/低
FAI 位置的 rollout perturbation 和跨 layer/task/model 复现增强证据。

可迁移的不是 WAAD/FAI 本身，而是“**可视化发现→冻结事件定义→跨样本 coupling 与 null
比较→定向扰动→跨分布复现**”的证据阶梯。该论文也明确说 attention-derived signal 不是无偏解释或
完整因果分解，也不判断中间步的局部正确性。因此它支持“这是可复现的结构信号”，不直接支持“该
token/head 承载了正确事实”。

本项目保留该阶梯，但不沿用 head-group average：事件在逐 `(layer, head, q)` 的 true-message transport
上发现，跨 head 只统计 support、signed agreement 与 post-\(W_O\) vector coherence。任何跨样本规律都按**事件
阶段**对齐，不要求不同模型共用固定 head 编号或 layer 编号。

### 7.2 主假设：candidate competition→constraint reread→adoption→reuse

主机制是待证伪的四阶段事件，而不是单个高分节点：

| 阶段 | 预注册操作定义 | 当前代码状态 |
|---|---|---|
| C：candidate competition | 回答历史和/或 prompt 中至少两个候选 route 同时超过在 calibration/null 上冻结的 transport floor；注册四桶 transport 分布、top-2 margin 与 entropy，但不把桶 entropy 当成语义候选数 | **部分已测**：已保存四桶 totals/winners；未实现 source-unit 级候选集、统一 phase 判定与语义候选正确性 |
| R：constraint reread | 在同一 head 上 local 下降、long-range 上升且经注册 prompt-role 标注确认 winner 是当前决策所需的 constraint/evidence，而非位置上的任意 prompt token | **部分已测**：结构切换、source token/unit 已保存；未实现 constraint-role 标注和“读对限定条件”判定 |
| A：adoption/commit | reread 的 source-specific message 对紧随 \(q+1\) 的 registered factual contrast 有正 action，且 matched source patch/cut 选择性改变 margin；attention/MLP ledger 用于定位写入、抵消或接纳 | **部分已测**：query-matched `grad·message` 与 selected-root integration 已有；它们仍是 observed-token/post-selection screen，未实现 event-source-specific factual intervention |
| U：reuse | 后续 token 读取由 A 形成的 commitment/hub，evidence lineage 保留，且 restore 后的 rescue 被注册 downstream block 消除 | **未完整测量**：稀疏 provenance 与 downstream-action 只能产生 relay candidate；尚未将事件新写入的 state 与后续 reuse 因果绑定 |

完整机制要求在同一样本中冻结 C→R→A→U 的顺序和 source identity，并在样本级判定四阶段是否闭合。
多个 head 关注同一区域只是 coordination candidate；只有 post-\(W_O\) 写入方向、signed action、source-specific
intervention 和后续 reuse 一致，才能把它升级为功能链证据。

### 7.3 Grounded Evidence / Generated History 双轴

不再只用差值 \(C_t\) 压缩两类信息。对 query row 的 origin \(o\)，仍定义

\[
s_{o,l}=\sum_h a_{l,h,o},\qquad
b_{o,l}=\sum_h|a_{l,h,o}|,\qquad
Q_o=\sum_l s_{o,l}\frac{|s_{o,l}|}{b_{o,l}+\epsilon}.
\]

下一阶段预注册两个**独立 raw axes**：

\[
GE_t=\frac{Q_{evidence}}{\sum_l b_{evidence,l}+\epsilon},\qquad
GH_t=\frac{Q_{response}}{\sum_l b_{response,l}+\epsilon}.
\]

`GE` 是 Grounded-Evidence-origin adoption，`GH` 是 Generated-History-origin reliance；在实现
lineage-resolved 版之前，`GH` 不得改名为 unsupported history，`GE` 也不得称 factual correctness。高 GE/高 GH 可能是
证据已经过 hub 正常复用；低 GE/高 GH 只是 unsupported-self-reliance candidate，不自动等于 hallucination。
现有 `route_origin_competition` 仍作为冻结静态 baseline：

\[
C_t=\frac{Q_{response}-Q_{evidence}}
{\sum_l b_{response,l}+\sum_l b_{evidence,l}+\epsilon}.
\]

当前代码**已输出** \(C_t\)，但**尚未输出**独立 GE/GH、lineage-resolved GH 或二维校准边界。不得从
\(C_t\) 反推 GE/GH 的二维结构。

### 7.4 预注册失败类型

| 类型 | 操作性定义 | 当前能否判定 |
|---|---|---|
| `miss` | 任务标注表明此时需要回读限定条件，但注册窗口内没有 R | 否；缺 constraint-opportunity 标注和 non-event 对照 |
| `misread` | 发生 R，但 winner 是 distractor/错误限定条件，或对条件的 signed effect 方向错误 | 否；位置桶不含语义正确性 |
| `reject` | 正确条件的 message 到达，但 A 为零/负或 matched patch 不改变 registered margin | 仅能标记 first-order candidate；尚不能确认 |
| `overwrite` | A 曾为正，但在 target 前被 attention/MLP 抵消、翻转或丢失 lineage | 仅有 selected-root ledger 候选；缺 event-specific longitudinal patch/block |
| `false-basin` | 高 head agreement/低冲突且 GH 持续占优，对正确证据扰动不更新或扰动后回到原错误轨迹 | 否；必须用 free-running 双向证据替换/恢复轨迹 |

`false-basin` 不能由“多个 head 一致”或单次 teacher-forced 低冲突命名。同样的稳定性也可以是正确约束已被接纳的
grounded basin。

### 7.5 跨样本统计基线

机制的“普遍出现”以 sample/source 为独立单位，不把 token、head 或 layer 当成 iid 样本。预注册：

- 事件 null：在同样本、同 head/layer 内对 q 做 circular shift，或抽取匹配到 response 位置、长度与前一行
  local-dominance 的非事件位置；
- source null：使用同位置/同长度/同词法类型的 other-prompt span、distractor constraint 与 same-fact
  paraphrase；
- pair design：同 prompt/source 下匹配 grounded 与 hallucinated trajectories，或用预注册 factual swap 生成对，分别比较
  C/R/A/U 的发生率、时序、source kind、GE 和 GH；
- inference：按 source/question 聚类 bootstrap CI 或预注册 mixed-effects model；逐 head/layer 探索用
  sample-level permutation 的 max-statistic 或 FDR 控制多重比较；
- generalization：在 train/calibration 冻结 event threshold、GE/GH 方向与 detector，test 只评估；分别报告 event
  coverage 和 event-conditioned 效果，并在不同 task/model/template 上复现。

在完成上述比较、定向扰动与 held-out 复现前，C→R→A→U、GE/GH 和五类失败都是**预注册假设**，
不是当前实验结论。

## 8. --confirm 的验证矩阵

confirmation 是固定假设的检验，不是搜索：

| 检验 | intervention | 允许解释 |
|---|---|---|
| root necessity | clean 中 cut selected-root Value messages | 当前 operator 下 root 是否必要 |
| root conditional sufficiency | all-evidence-cut world 中 keep selected root | 条件性恢复，不是完整充分性 |
| corridor restore/block | 恢复或删除 frozen corridor messages | 候选 corridor 是否携带 effect |
| hub restore | source-corrupt world 中恢复 frozen hub state | hub state 是否足以部分 rescue |
| retrieval block | hub restore 后 block frozen hub→target corridor | rescue 是否由注册下游路径介导 |
| restoration | cut 后原位补回同一 message | intervention 实现是否在 tolerance 内闭合 |

root necessity 可以复用默认生成的 selected-root cut world，但只有 --confirm 补齐 conditional keep 和
注册检查后才设置 evaluated；单独存在 root_cut_margin 或 root_value_effect 不等于 root 已确认。
未指定 --confirm 时，selected_root_evaluated=false，selected_root_value_necessity、
selected_root_conditional_sufficiency 与 selected_root_causal_score 均为 NaN。

未请求 --confirm 记为 not_run/unevaluated；这与“运行后未通过”不同。single-route null 可能来自
冗余、自修复或更早的状态转移，只否定当前必要性检验。positive 结果也仅适用于注册的
target、operator 和输入分布。

## 9. “承载什么信息”的 2×2 matched factorial 审计

状态：**已设计、尚未由当前 native subset 实现或验证**。以下规则用于后续 controlled experiment，
不能从现有单样本图回填语义标签。

native 单样本图只能标记 evidence-origin、other-prompt、response-origin 与 unobserved；它不能凭一张图
给 hub 加上 fact、confidence 或 groundedness 语义。内容审计另构造 token-aligned 的
fact value \(F\in\{0,1\}\) × confidence condition \(C\in\{0,1\}\) 四个 cells，记候选 node activation
为 \(h_{fc}\)。注册：

\[
d_F=\tfrac12[(h_{10}-h_{00})+(h_{11}-h_{01})],
\]

\[
d_C=\tfrac12[(h_{01}-h_{00})+(h_{11}-h_{10})],
\qquad
d_{F\times C}=(h_{11}-h_{10})-(h_{01}-h_{00}).
\]

其中 \(d_F\) 是跨 confidence 平均的 fact contrast，\(d_C\) 是跨 fact 平均的 confidence contrast，
\(d_{F\times C}\) 是 interaction difference-of-differences。设计要求：

- 四格在 audited node 前保持 tokenizer coordinates 与序列长度对齐；teacher-forced response scaffold
  对齐，注册的 fact/confidence factor slot 可以使用不同 token，outcome 用预先固定的 candidate margin；
- fact cells 只改变注册 factual value/entity/relation，匹配 token 数、实体类型和语法角色；confidence
  cells 只改变注册 confidence cue/condition；
- same-fact paraphrase、grammar/template swap、随机同位置 span 与 other-prompt difference 作为
  nuisance controls；
- 在独立模板和独立实体上分别检验 contrast 泛化，不能只做同一词面的 probe accuracy；
- causal fact swap 应改变答案事实或 factual answer margin，而不只是改置信表达；causal confidence
  swap 应改变 confidence token/表达，同时尽量保持 answer identity 与 factual margin；
- clean→counterfactual 与 counterfactual→clean 双向 patch 都报告原始 margin，并用 downstream block
  检验注册 route；
- 无法 token-align 的 corruption 不进入原位 patching 主分析。

一个 hub 可以同时承载 fact main effect、confidence main effect 和 interaction，不能强制一个节点只有
一个标签。只有主 contrast 超过 nuisance controls、跨模板/实体泛化且 causal swap 有选择性，才允许
称操作性的 fact/confidence carrier；这些控制仍不证明存在纯净、正交的语义方向。

## 10. Verbal-confidence 研究的使用边界

[*How do LLMs Compute Verbal Confidence?*](https://arxiv.org/abs/2603.17839) 提供的是一套
cache→retrieve 的实验模板：先定位 answer-adjacent cache candidate，再用 restore、noising、
matched activation swap 和 attention blocking 检验缓存与取回。长 prompt 的 direct block null
还提醒多跳中继与冗余会隐藏单边效应。

本文只借用这个实验顺序。该论文没有操纵外部 supporting evidence，也没有证明其 cache 表示
grounded truth；PANL/CC 位置、层号和 verbal-confidence 语义均不预注册到本项目。

## 11. 预注册输出与图

discovery 图包含：

| 面板 | 内容 | 结论上限 |
|---|---|---|
| route | frozen root→hub→target backbone、逐 head message 和 unobserved 诊断 | target-conditioned candidate topology |
| switch timeline | 每个 layer/head 独立轨道上的 full-row 四桶 transport switch | local→long-range 结构候选 |
| event triangles | top 事件的 prompt 区/local band、每桶最强 source/token/unit 与 action | 桶聚合的 source 候选，不是完整 \(T^2\) |
| integration | residual/attention/MLP 与 selected-root coherence/action | post-selection 接纳 screen |
| confirmation | exact ladder，未请求时明确显示 not run | frozen plan 的验证结果 |

图不读取 hallucination labels。没有 --confirm 时空 confirmation 面板不能解释为 negative result。
一次 forward 的 layer-unrolled DAG 不画成反馈环。

命名规则：

| 证据 | 名称 |
|---|---|
| 只有 local↓、long-range↑ 与 dominance flip | structural reanchor candidate |
| remote response 且 lineage > 0 | evidence-bearing relay candidate |
| query-matched positive action | observed-token adoption candidate |
| 只有 plan score | candidate root/hub/corridor |
| native exact ladder 通过 | confirmed native causal relay |
| matched factual controls 也通过 | verified factual mediator |
| 只有高 response action | response-reliance candidate |
| 只有高 agreement/continuity | functional-alignment candidate |

## 12. 审计与 AUROC 的两阶段协议

### Phase A：label-free audit

冻结并保存 structural events、target、AuditPlan、预算、unobserved diagnostics 和三个彼此独立的
discovery-only raw axes：

\[
C_t=\operatorname{route\_origin\_competition}.
\]

\(C_t\) 越高可能对应更高 hallucination risk，这是未验证检测假设，而不是已知事实。它只由 discovery graph 的
all-evidence-origin/response-origin signed action、absolute budget 与逐层 head agreement 计算。固定的
selected-root cut、--confirm outcomes 和 labels 都不参与定义 \(C_t\)，因此 discovery-only artifact
可进入 Phase B；exact effects 只作为可缺失的 secondary diagnostics。

另外保存 `temporal_switch_score`（只在 event-center artifact 上使用冻结 layer/head/query 的
transport-only score；中性轴，预先同时报告 raw-higher 与 negated）和 `evidence_adoption`（只在
prompt-evidence event center 上使用同一 layer/head/query 的 full-row prompt-evidence bucket signed
action；注册为越高风险越低）。三者分别评价，不拟合组合权重。
下一阶段预注册的 GE/GH 是从同一 origin ledger 分开保留的二维 raw axes；当前 artifact 没有输出它们，
所以不进入现有 Phase B 指标。

### Phase B：label join 与评价

完成全部 capture 后才读取 labels。三个 raw axes 按各自预注册的方向或中性双向报告约定分别输出：

- target/sample 数、positive 数和 prevalence；
- AUROC 与 AUPRC；若只有一个类别则为 null；
- temporal axis 按预定义 source kind 分层的 raw-higher 与 negated AUROC/AUPRC；
- 按 task/source 的分组结果；
- 按 source 聚类的 bootstrap confidence interval（实现后才可报告）。

route baseline 可使用所有冻结 targets；temporal/adoption 只把 label 连接到 selector 明确记录的
event-center artifact 自己的 \(q+1\) token。窗口上下文与 no-event fallback 不进入这两个轴，
other-prompt/remote-response center 不进入 evidence-adoption；prefix 中更早 event 的 downstream
action 不进入该 label 的 axis。
评价进程不得修改 mechanism artifact、候选或方向，也不得用 test labels 选择阈值和组合权重。
若未来学习组合分数，只能在 train/calibration split 固定后评估 test。AUROC 证明的是预测关联，
不是 C→R→A→U 或五类失败的因果机制。每任务一个样本只是 smoke test，不足以判断检测效果。reanchor-selected
cohort 还是 event-conditioned sample；在加入预注册非事件对照/覆盖抽样前，不得把 AUROC/AUPRC
解释成所有自然 response tokens 的 population detector performance。

## 13. 判读清单

发布任何结果前确认：

1. 四个互斥位置桶是否在 sparse edge pruning 前由 full causal rows 聚合？
2. structural score 是否只要求同一 head 的 local↓、long-range↑ 与 dominance flip，不含 action/label/ablation？
3. 只有 query-matched event 是否被称为 immediate next-token action？
4. AuditPlan 是否在 exact rerun 和 labels 之前冻结？
5. 是否展示了真实 \(W_V\to W_O\) message 且保留 head 轴？
6. root/hub/corridor 是否按 route mass × absolute action budget × agreement 的 magnitude 选择，并另存 signed direction？
7. capture 2k 并集、corridor k cap、rows 实际用量与 unobserved fraction 是否报告？
8. 默认 selected-root diagnostic cut 是否明确位于 plan 冻结之后且不用于确认？
9. not_run、failed 和 confirmed 是否区分？
10. native observed-token support 是否被误写成 factual grounding 或 confidence semantics？
11. 2×2 内容结论是否通过 nuisance control、跨模板/实体泛化和 selective causal swap？
12. response reuse 是否被误写成 hallucination self-reinforcement？
13. 三个 raw axes 是否按预注册方向/中性双向约定，只用合格 event-center 的 query label 分别报告？
14. C/R/A/U 是否在同一 source identity 上闭合，还是只报告了共现节点？
15. GE/GH 是否仅在实际实现后分开评估，五类 failure 是否遵守各自的操作定义？
