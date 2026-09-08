# v3 基线：逐 head 时间轴重锚定与预算化接纳审计

**本文件归档旧方法。新的消息算子、来源传播、同目标路径图及运行入口位于 [message_dag/](message_dag/README.md)。**

**历史方法归档。当前研究定位统一以 [DAG 消息路由建模与幻觉机制发现](MESSAGE_LINEAGE.md) 为准：图上的来源、聚合和传递结构是主线；有符号材料分解是已实现组件，固定分数只是检测对照。完整同目标路径图的设计与未实现项见该文档 §6.1–6.3。**

本文仅记录旧 v3 实现，用于复现已有产物。其四桶 transport 是消息范数之和，不能恢复完整向量。
已实现的向量探索入口为 `discover`；它的 PCA／聚类不是经过验证的机制检测算法。更正后的研究主线见
[MECHANISM_AUDIT.md](MECHANISM_AUDIT.md)，
运行命令见 [README.md](README.md)。下文的结构选择器、来源分数和 runner 目标属于旧路径。

## 1. 方法要回答什么

本方法先回答一个时间问题，再回答一个功能问题：

1. 在 response predictor \(q\) 上，某个 head 是否从最近少数 response token 转而读取原始 prompt
   或更早的 response relay？
2. 这个冻结的结构事件读到了哪个 source；该信息是否实际写入 residual、被后续计算接纳，并对
   \(q+1\) 或一个明确的晚期 target 有帮助？

第一问是**不依赖 target gradient 的全时间轴结构发现**；第二问才是固定 target 的 route/action/
integration 审计。它们不能用同一个分数互相替代。本方法也不把 response token 对先前 response 的
正常依赖自动解释成幻觉自我强化。

对 predictor 位置 \(q\)，先冻结 target contrast：

\[
F_t=z_q(a)-z_q(b).
\]

- native audit 中，\(a\) 是 teacher-forced observed token，\(b\) 是未干预 native run 的 frozen
  runner；这测量 observed-token support，不测量 factual correctness。
- controlled pair 中，\(a,b\) 是运行前注册的 grounded/counterfactual candidates；native 与 pair
  estimand 不混报。

\(b\) 只在完整 response 的 discovery run 冻结一次。prefix capture 通过 `fixed_runner` 显式沿用它，
缓存 readout、gradient 与 cut 不重新定义 contrast；这修复了旧流程第二次 argmax 因近并列舍入换位而
报 `negative candidate is not the frozen native runner` 的问题，target identity 检查仍保留。

root/hub/corridor 与 adoption diagnostics 是 **target-conditioned** 的：改变 target、runner、source
units、destination scope 或预算就得到另一个 AuditPlan。时间轴 structural switch 本身只依赖当前
模型的 clean full-row message transport，不读取 runner、gradient、label 或 intervention outcome。

## 2. 先冻结结构事件，再冻结 AuditPlan

v3 按以下顺序把时间事件、target-conditioned 候选与因果确认隔离：

1. 在不读取 label 的 clean model 上扫描 response full rows，用 transport-only 规则冻结重锚定事件；
2. 按 `reanchor` 或 `reanchor-window` 策略冻结有限的 target rows，再冻结 observed token 与 native
   runner contrast；
3. 对每个 target 做一次 head-resolved capture，计算 route mass 与 signed target action；
4. 在固定预算内选择 roots、hubs 和一条 connected corridor，并冻结为 AuditPlan；
5. 计划冻结后，固定做恰好一次 selected-root cut，构造 native/root-cut paired state 与
   attention/MLP integration ledger；
6. 默认到此结束；只有显式传入 `--confirm` 才把冻结 root/hub/corridor 送入额外的
   necessity/sufficiency/restore/block validation ladder。

第 5 步不是搜索：它只针对已选 root，不能改变 plan score、rank、hub 或 corridor，也不因 margin
变化大就把候选标为 confirmed。--confirm 不参与第 1–3 步。确认结果不得重排 root、替换失败的 hub、
补选 corridor edge 或扩大预算。候选被 exact intervention 否定时，正确结果是“预注册候选未通过”，
不是用 intervention
effect 重新挑一个更好看的候选。结构事件的坐标也不得由 action、integration 或 exact effect 回写。

这避免了以每条 route 的 exact cut 作为搜索算法。候选发现的成本由流式 full-row 扫描、每个 target
一次 capture/backward、一次冻结后的 selected-root paired run 和固定大小的稀疏图控制；不会为每条
候选 route 各跑一次剪除。额外 exact forward rerun 只在 --confirm 下花在冻结的少量候选上。

`reanchor` 把 `targets_per_sample=N` 解释为最多 \(N\) 个经时间 NMS 的事件中心；
`reanchor-window` 把 \(N\) 解释为 target-row 硬总预算，先保留最多 \(\lceil N/3\rceil\) 个中心，再按
rank 补 \(-1,+1\) 邻居。\(N=3\) 就是最强事件的三行窗口。完全没有结构事件时才回退到
evenly-spaced rows，且 `contrast_origin` 必须记录 fallback。

`audit-all` 默认覆盖 train/test 全部可用样本和完整 response，先持久化独立 `SampleScan`，随后每样本
以 3-row event window 做功能审计。`--scan-only` 停在完整结构扫描与事后 cohort 比较，不计算 gradient/
root cut，也不能验证接纳；去掉该参数可复用同一 world/scan 补功能。完整 scan 不依赖选中 target 的并集。
冻结事件 identity/score 与预算内 prefix 重算是两次测量；评价保留前者，另报
`temporal_switch_recomputed_score` 与 `temporal_switch_score_delta`，不强求分数或 winner 完全一致。

## 3. 图的原子是实际 residual message

attention weight \(A\) 只是 QK gate。layer \(l\)、query head \(h\) 从 source \(j\) 写到
destination \(i\) 的实际 message 是

\[
m^{l,h}_{j\to i}
=W_O^{l,h}\left(
A^{l,h}_{i,j}\,W_V^{l,g(h)}\operatorname{LN}(r_{l,j})
\right),
\]

其中 \(g(h)\) 是 GQA query-head 到 KV-head 的映射，\(W_O^{l,h}\) 是该 query head 对应的
o_proj slice。实现可以在 pre-\(W_O\) code 上借助 \(W_O^TW_O\) 计算 message norm，但语义仍是
post-\(W_O\) residual write，而不是 attention × V 的未投影近似。

每条显式边必须保留 (layer, head, source, destination)。不同 heads 可以分别做 global retrieval、
local copy、抑制或补偿；不能先对 head 求平均。允许在完整逐-head值之上计算 signed sum、absolute
budget 和 agreement，但这些汇总不能替代原始 head 轴。

## 4. 全时间轴的结构重锚定发现

令 `local_window=w`。对每个 response destination \(q\)，其完整 causal source row 在 sparse top-k
之前被分为四个互斥且穷尽的集合：

| source bucket | 定义 |
|---|---|
| `prompt_evidence` | \(s<\text{response_start}\)，且 source unit 属于预注册 evidence units |
| `other_prompt` | \(s<\text{response_start}\)，但不属于 evidence units |
| `remote_response` | \(s\ge\text{response_start}\) 且 \(q-s>w\) |
| `recent_local` | \(s\ge\text{response_start}\) 且 \(0\le q-s\le w\)，包含对角线 |

对 bucket \(b\) 保存 attention mass 与真实 message-transport magnitude：

\[
T_{l,h,q,b}=\sum_{s\in b}\left\|m^{l,h}_{s\to q}\right\|_2.
\]

还保存每桶 transport 最大的 source position/unit、该 source 的 attention 和 transport。完整
`[head, query_chunk, causal_source]` row 只在 chunk 内存在；常驻结果是
\([L,H,P,4]\) 的桶总量与 winner。因此这里确实读取裁剪前 full rows，却不宣称保存了完整
\(L\times H\times T^2\) triangle。

将前三个 long-range 桶合为 \(R_q\)，`recent_local` 记为 \(L_q\)，并定义

\[
p_q=\frac{R_q}{R_q+L_q}.
\]

一个 `(layer, head, q)` 只有同时满足以下条件才是结构候选：

- \(q-1\) 为 local-dominant，\(p_{q-1}<0.5\)；
- \(q\) 为 long-range-dominant，\(p_q\ge0.5\)；
- raw long-range transport 上升，且 raw recent-local transport 下降。

相邻位置的归一化变化为

\[
r_q=\frac{\max(R_q-R_{q-1},0)}{R_q+R_{q-1}},\qquad
f_q=\frac{\max(L_{q-1}-L_q,0)}{L_q+L_{q-1}},
\]

结构分数注册为 \(S_q=\sqrt{r_qf_q}\)，非候选处为零。每个 head 独立找 temporal peak；同一时间
跨 head/layer 的平均值不参与发现。时间 NMS 后用最强单 head 分数排序，同时保存该时间的 head
support 作为诊断，而不是平均分。

这里的 anchor 只表示 long-range source location。winner 是 `prompt_evidence`、`other_prompt` 或
`remote_response` 中 transport 最大的桶；`remote_response` 只有在随后测得 evidence-lineage fraction
大于零时才能称 evidence-bearing relay candidate。该 fraction 来自 coverage-pruned sparse route
provenance；预算外质量仍是 unknown/unobserved，不重归一，所以它不是 full-row exact lineage 或语义
真值。bucket kind 是输入 source position/unit 的身份，不代表当前 layer 的 node state provenance 是
纯的；prompt-evidence、other-prompt 或 response winner 经更新后都可能混合 lineage。attention bucket
只作辅助图，不能替代上述 message transport selector。

结构候选冻结后才附加两类功能注释：

- bucket/source 的 signed `gradient · message`。只有 event 的 \(q\) 等于 artifact 的
  `query_position` 时，才是该 event 对紧随 \(q+1\) token 的 immediate action；prefix 更早事件只能
  解释为对当前晚期 target 的 downstream action。
- selected-root cut 的 integration。只有 winner 为 `prompt_evidence` 且其 unit 正好等于 selected root
  时，该 source 的 selected-root integration 才适用；其余事件必须记为 not applicable，不能泛称已
  接纳该 source。

因此 attention/transport triangle 用于**找切换位置**，action/integration 用于**检查接纳**，exact
intervention 用于**验证冻结 operator**。三者是有顺序的证据层级。

## 5. 固定 target route graph 的三个候选量

### 5.1 Route mass

message norm 与隐式 residual continuation 构造一个仅用于 topology screening 的 transition law。
从冻结 target 反向传播单位 route mass，再与给定 source root 的正向 reachability 相交，得到
root-conditioned node/edge throughput \(M\)。

这只是 norm-based routing model。它不表示语义信息的概率，也不是 causal effect。

全局 provenance ledger 与 root-conditioned throughput 必须分开：EVIDENCE channel 在输入层包含
world 注册的 **全部 evidence units**，因此主 route-origin competition 比较 all-evidence lineage 与
response lineage。selected root 只用于解释性的 root-conditioned throughput/backbone、direct-root
fraction，以及计划冻结后的 root-cut integration；它不能把其他 evidence units 降为 other_prompt。

### 5.2 Signed grad-message

一条 native message 对固定 contrast 的局部作用为

\[
a_e=\left\langle\nabla_{m_e}F_t,m_e\right\rangle.
\]

正值局部支持 \(a\) 相对 \(b\)，负值局部反对，绝对值表示候选作用强度。该量必须保留符号；只看
message norm 无法区分支持与抑制。

### 5.3 Head agreement

对候选 group \(G\)，先在每个 head 内聚合 routed signed action \(a_{G,h}\)，再计算

\[
A_G=
\frac{\left|\sum_h a_{G,h}\right|}
{\sum_h|a_{G,h}|+\epsilon}.
\]

\(A_G\) 是 cancellation penalty，不是 head 平均，也不要求 global/local heads 关注相同 token。
所有 \(a_{G,h}\) 仍写入 artifact。低 agreement 说明不同 heads 的 target action 抵消；高 agreement
只说明当前 target contrast 下的 functional alignment。

## 6. root、hub 与 corridor 如何发现

对候选 group (G)，定义 route-weighted signed action 与 absolute budget：

\[
s_G=\sum_h a_{G,h},\qquad
B_G=\sum_h|a_{G,h}|.
\]

冻结的 selection magnitude 是

\[
S_G^{mag}=M_G\,B_G\,A_G.
\]

其中 \(M_G\) 是与 target 连通的 route mass，\(A_G=|s_G|/(B_G+\epsilon)\)。忽略数值稳定项时，
它等价于按 \(M_G|s_G|\) 排序；agreement 已通过 absolute budget 到净 action 的收缩进入分数，不能再
对 \(s_G\) 重复乘一次。\(S_G^{mag}\) 非负，signed_action \(s_G\) 单独保存方向；后续“支持
evidence”的命名还要求 \(s_G>0\)。不能用 exact cut effect 替换这里任何一项。

### Root

root group 是一个预注册 source unit。每个 unit 都由其 target-connected messages 计算上述量；
在 root_candidates 预算内保存排名，首项成为 selected root。若 functional score 全为零，只允许按
预注册的 route-mass fallback 和稳定 tie-break 选择，并明确记录 fallback，不能调用 rerun 决胜。

### Hub/cache candidate

hub 是内部 (layer boundary, position)。它必须同时拥有 selected-root 的 inbound flow 与通向
target 的 outbound flow；取两侧 absolute action budget 的 bottleneck，再乘 route mass 与两侧
agreement 的 bottleneck。signed action 仍单独保留。相邻层、同一
position 的重复候选用预注册 NMS 规则合并。高分 hub 只表示“可能缓存并中继 target-relevant
source effect”，不是已验证 cache，也不是 grounded representation。

### Corridor

corridor 首先在 layer-unrolled DAG 中保留一条 selected-root 到 target 的 connected backbone；
缺少显式 message 的层可由 residual continuation 跨越。随后仅在 represented rows 内按 \(S_e^{mag}\)
补充边，受每 (layer, head, destination) cap 与全局 corridor_edges cap 约束。它是候选 mediation
subgraph，不是通过逐 route deletion 得到的“最小因果电路”。

## 7. 默认 destination 与硬预算

长上下文的完整 \(L\times H\times T^2\) 图既不必要也无法保证显存。时间轴扫描按
`query_chunk` 临时生成 full rows，在 edge pruning 前聚合四桶后立即释放；常驻开销为
\(O(LHP\cdot4)\)，不是持久化完整 triangle。以下为 `subset` 默认值；`audit-all` 改为全部样本、
完整 response、`reanchor-window` 与 3 targets：

- target_policy=reanchor、targets_per_sample=1；完整事件窗口审计建议显式使用
  `reanchor-window` 与 3 个 target rows。
- max_response_tokens=128：限制 clean 时间轴 pilot 的 response horizon。
- query_chunk=8：限制一次临时生成的 query rows；遇到 OS/cgroup OOM 可降低它。
- local_window=10：response source 距离不超过 10（含对角线）属于 `recent_local`。
- carrier_scope=response：只展开 response-side destination rows；所有 causal prompt positions
  仍可作为这些 rows 的 sources，因此这不等于忽略 prompt evidence。
- max_route_rows=256：超限时保留靠近 target 的最近连续 destination rows，保留全部 causal sources；
  截尾由 `route_row_position` 明确表示，不影响完整 sample scan。
- edges_per_head=2：capture 分别保留 transport top-k 与 \(|\text{functional}|\) top-k 的并集，
  因而每个 head-row 最坏保存 \(2k\) 条候选边；冻结 corridor 再对同一 head-row 硬限制为 \(k\)。
- corridor_edges=64：一个 AuditPlan 中显式 corridor messages 的硬上限。
- root_candidates=4、hub_candidates=8：候选数上限。

edge_coverage 只控制 transport top-k 分支在预算内尽量达到的覆盖目标；functional top-k 分支独立
保留高 absolute action 边。coverage 不得突破 capture 的 \(2k\) 并集上限或 corridor 的 \(k\) 上限。
每个 represented head-row 的完整 transport mass 在裁剪前计算；该 row 中未被保留的 edge mass 及
其他无法追踪的质量进入 unobserved sink，绝不重新归一到保留边。因此对 represented row：

\[
M_{retained}+M_{residual}+M_{unobserved}=M_{row}.
\]

若 unobserved 很大，允许的结论只能是“在当前预算下发现了候选子图”；不能把稀疏图解释成完整
信息流。response scope 外的 prompt destinations 没有 attention row；它们只沿 residual identity 保留
已有 all-evidence/other-prompt provenance，使后层 represented row 仍可读取 prompt state。该近似没有
建模 prompt 内部更新，不能用来定位 prompt hub。未展开的 response 节点从 layer 1 起进入
UNOBSERVED，不把未知 state 继续解释为纯 response-origin；缺失 evidence lineage 不代表没有证据。
carrier_scope=all 也受同一 destination 截尾预算约束。功能审计单层 autograd 仍可能使用
\(O(H T^2)\) 中间量，该预算不限制完整 causal prefix 的显存。

## 8. 为什么 residual difference 或 gradient 单独都不够

对 clean/counterfactual 或 native/source-cut 两个 worlds，节点差分

\[
d_v=h_v^{clean}-h_v^{cf}
\]

只说明该 intervention 改变了状态。它可能混合事实、词法、语法、长度、实体类型及下游非线性
效应；高 \(\|d_v\|\) 不说明 target 读取了该差异。

梯度

\[
g_v=\nabla_{h_v}F_t
\]

只说明在当前基点附近，某个方向可以改变 target。高 \(\|g_v\|\) 不说明实际 activation 中存在
source-specific 信息，并且局部线性化会受 softmax saturation、抵消和高阶交互影响。

两者的内积

\[
\widehat{IE}_v=\langle g_v,d_v\rangle
\]

把“被 intervention 改变”与“沿 target-sensitive 方向改变”结合起来，是 AtP/EAP 类的一阶
screen；它仍不是真实 intervention effect。高 displacement、低 \(|\widehat{IE}|\) 可注册为
changed_but_not_used candidate；高正值可注册为 accepted/supporting candidate；这些名字都要在
exact confirm 前保留 candidate。

attention write 与 MLP write 分开保存 displacement 和 signed action，用于观察 source effect 被
增强、抵消或覆盖。它们不能从 residual 中“纯化”出事实语义；纯语义主张还需要 token-aligned
factual swaps、same-fact paraphrase 和 grammar controls。

默认的一次 selected-root cut 只为上述差分与 integration ledger 提供 paired state。因为 root 是先由
同一 native graph 选出的，这个差分是 post-selection diagnostic，不是独立的 root confirmation；它也
不能用来重排候选。

## 9. Hub 到底承载什么：2×2 factorial 审计

这是下一阶段的 controlled experiment 预注册；当前 native subset CLI 不生成四格 worlds，也不会输出
fact/confidence content label。

native 单样本图只能区分 evidence-origin、other-prompt、response-origin 与 unobserved lineage，不能
单独判断某个 hub 编码的是事实值、置信表达、二者交互，还是共享的词法/模板特征。要回答“承载什么
信息”，需要另建 token-aligned 的 \(2\times2\) matched factorial worlds：

| 因子 | 水平 0 | 水平 1 |
|---|---|---|
| fact value \(F\) | factual value/entity A | matched factual value/entity B |
| confidence condition \(C\) | low/uncertain cue | high/certain cue |

令同一 node 的 activation 为 \(h_{fc}\)。分别注册跨 confidence 平均的 fact contrast、跨 fact 平均的
confidence contrast，以及 interaction difference-of-differences：

\[
d_F=\tfrac12[(h_{10}-h_{00})+(h_{11}-h_{01})],
\qquad
d_C=\tfrac12[(h_{01}-h_{00})+(h_{11}-h_{10})],
\]

\[
d_{F\times C}=(h_{11}-h_{10})-(h_{01}-h_{00}).
\]

same-fact paraphrase、grammar/template swap 与 matched random-span difference 是 nuisance controls；
候选方向只有在主 contrast 超过这些 control，并能跨模板、跨实体泛化时才获得内容标签。一个 hub
可以多路复用，同时有 fact、confidence 与 interaction 成分，不能强制分配单一标签。

最终需要 causal swap，而不只看 probe 或 cosine：把 fact component 从 B world patch 到 A world 应
改变答案事实/对应 answer margin，同时尽量不只改变 verbal-confidence 表达；patch confidence
component 应改变 confidence token/表达，同时尽量保持答案身份与 factual margin。双向 swap、matched
cell controls 与 downstream block 都通过，才支持“该 hub 因果承载相应信息”。线性 contrast 只是
操作性分解，不证明模型内部存在彼此正交、可完全分离的语义变量。

## 10. --confirm：少量因果验证

只有显式 --confirm 才对冻结计划执行额外验证：

1. **root cut/keep**：测试 selected source 的 necessity 与 conditional sufficiency；
2. **corridor restore/block**：恢复或删除计划中的真实 pre-\(W_O\) message endpoints；
3. **hub restore**：在 source-corrupt world 中恢复 frozen hub state；
4. **hub→target block**：恢复 hub 后阻断预注册 downstream corridor，检验 mediation；
5. **restoration check**：cut 后原位恢复必须在 dtype tolerance 内重建相应 world。

确认对象始终来自同一 AuditPlan。single-node null 可能来自冗余、自修复或信息已在更早层转移；
它否定当前 intervention 下的必要性，不证明模型从未使用该信息。positive result 也只支持该
operator、target 与数据分布下的因果作用。

默认 selected-root cut 已保存 root_value_effect = native_margin − root_cut_margin，但它只是
post-selection diagnostic。未传 --confirm 时 selected_root_evaluated=false，
selected_root_value_necessity、selected_root_conditional_sufficiency 与 selected_root_causal_score 都是
NaN；不能把 root_value_effect 改名为 necessity。

## 11. 两类可迁移的发现/验证模板

[*Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy
Optimization*](https://arxiv.org/html/2510.13554v2) 的发现流程是：按 attention-weighted mean
backward distance 将 heads 分成 local/global 组，**组内平均 attention**后发现 sawtooth/anchor 图样，再定义
WAAD/FAI，用时序 coupling、随机基线、高/低 FAI 位置扰动及跨 layer/task/model 复现逐级增强证据。
可迁移的是“图样发现→冻结指标→null 比较→定向扰动→复现”的阶梯，不是其 head 分组或平均法。
论文本身也将 WAAD/FAI 限定为结构位置信号，而非完整因果分解或局部正确性判定。

[*How do LLMs Compute Verbal Confidence?*](https://arxiv.org/abs/2603.17839) 提供 cache→retrieve 的验证顺序：
steering、corrupt-restore patching、noising、matched swap 和 attention blocking。长模板中 direct block 的 null
result 还提醒多跳中继会隐藏单条直连路径。

本项目结合两个模板：先逐 head 找可复现事件，再对冻结的 source/hub/retrieval 做特异性验证。不能迁移论文的固定
head/PANL/CC/层号或语义结论；两篇论文都没有证明本项目的 RAG hub 表示 grounded truth。

## 12. 四阶段主机制与当前实现边界

本文将待验证机制定义为 **candidate competition→constraint reread→adoption/commit→reuse**，
并按事件阶段而非绝对 token/layer/head 对齐样本：

| 阶段 | 当前已有测量 | 尚缺的关键验证 |
|---|---|---|
| C：候选竞争 | 裁剪前逐 head 四桶 transport totals/winners | source-unit 级多候选、top-2 margin/entropy 的冻结 phase 判定；桶 entropy 不等于事实候选数 |
| R：限定条件回读 | 同 head local↓/long-range↑/dominance flip，source token/unit | prompt-role 标注、distractor 对照与“读对条件”判定 |
| A：接纳/承诺 | query-matched `grad·message`、selected-root attention/MLP/residual ledger | event-source-specific matched factual patch/cut；当前仅是 observed-token/post-selection screen |
| U：后续复用 | sparse evidence lineage、hub/corridor 与 downstream-action candidate | 事件新写入 state→后续 target 的 restore/block 闭合与 free-running 轨迹 |

多个 head 聚焦同一 source 只是 coordination candidate；高 agreement 只说明当前 contrast 下作用低抵消。只有
post-\(W_O\) 写入、signed action、source-specific intervention 和 downstream reuse 都一致，才能声称完整功能链。

因此五种失败模式也要按阶段定义：`miss`=应回读时无 R；`misread`=R 指向 distractor/错误条件；
`reject`=正确 message 到达但 A 为零/负或 patch 无选择性效果；`overwrite`=A 曾成立但在 target 前被
attention/MLP 抵消或丢失 lineage；`false-basin`=低冲突的 response-history 轨迹对正确证据扰动不更新或回到原错误轨迹。
现有代码不能确认这五类：前四类至多有部分 candidate measurement，`false-basin` 必须加入 free-running 双向扰动；
单次 teacher-forced 的高 agreement 同样可能是 grounded basin。

native RAGTruth audit 不能把 observed-token support 命名成 correctness 或 grounding。root/hub/corridor 在默认模式下仍是冻结
候选；只有 `--confirm` 能给出注册 operator 下的 exact effect，matched factual pair 加选择性 restore/block 才能称
verified factual mediator。

## 13. 审计与检测严格分阶段

当前 `cohort_summary.json` 先比较完整 sample scan：四类来源 transport 占比与 local→long-range
切换率均保留 layer/head。每样本内按标注组平均 tokens，再等权平均样本；差值只取包含两类 token
的 mixed 样本内“幻觉−非幻觉”。图中贡献样本数 <3 的格子置灰。这是描述性比较，未控制位置、
claim 类型或共享 source 依赖，也未实现 bootstrap CI/多重检验，不能称显著机制发现。
结构扫描覆盖、selected-target action/integration 与后述 AUROC 是独立统计口径。

阶段 A 是 label-free mechanism audit：冻结 structural events、targets、AuditPlan、预算和 raw
mechanism axes，生成图；固定的一次 selected-root cut 只增加 integration diagnostic，可选
`--confirm` 再增加验证字段。阶段 B 才读取 hallucination labels，并对三个互不混合的、未训练 raw
axes 分别计算 AUROC/AUPRC：

| axis | 定义 | hallucination-risk 方向 |
|---|---|---|
| `route_origin_competition` | response-origin action 相对 all-evidence-origin action | 越高风险越高 |
| `temporal_switch_score` | 只在 event-center artifact 上，冻结 layer/head/query 的 structural score | 中性；预先同时报告 raw-higher 与 negated |
| `evidence_adoption` | 只在 prompt-evidence event center 上，同一 layer/head/query 的 full-row prompt-evidence bucket signed action | 越低风险越高 |

下一阶段还将不做差值地分别注册 `GE` (Grounded-Evidence-origin adoption) 与 `GH`
(Generated-History-origin reliance)。两轴必须分开报告：高 GE/高 GH 可能是证据经 hub 正常复用，低 GE/高
GH 才是 unsupported-self-reliance candidate。当前 artifact **尚未输出**这两轴或 lineage-resolved GH，所以它们不进入现有
AUROC/AUPRC，也不能从 `route_origin_competition` 的差值反推。

route baseline 可使用所有冻结 target rows；temporal/adoption 只把 artifact 自己的 \(q+1\) label 接到
结构 selector 明确记录的 event center。窗口上下文与 no-event fallback 的 temporal/adoption axis 为
缺失；`other_prompt` 或 `remote_response` center 也不进入 `evidence_adoption`。任何 position 小于
query 的 event，其 action 都是通往晚期 target 的
downstream action，不能继承 query 的 label。评价阶段不得回写 artifact、重选
target/event/root/hub/corridor，或根据 test AUROC 选择 temporal 方向、阈值或组合权重。

每个 axis 的 AUROC/AUPRC 只能说明预注册机制量与 hallucination label 有预测关联；它不能反向证明
cache/retrieve、grounding 或 self-reinforcement 的因果解释。每任务一个样本只适合 smoke test，若
label 只有一个类别，AUROC/AUPRC 必须为 null。由于 reanchor target policy 按结构事件抽样，这些
结果还是 event-conditioned association，不是所有自然 response tokens 的 population performance；
population detector 需要预注册非事件对照/覆盖抽样，报告必须保留
`selection_is_not_population_evaluation=true`。中性 temporal axis 还要按预定义的
`prompt_evidence/other_prompt/remote_response` 分层报告同一双向指标，不能看完 labels 后挑 source
kind。

跨样本机制检验以 source/question 而非 token/head 为独立单位：用同 head/layer 内的时间 circular shift 和
匹配非事件位置作 event null，用 distractor/同长度 other-prompt span 作 source null，在同 prompt/source 内匹配 grounded
与 hallucinated trajectory。报告聚类 bootstrap CI 或预注册 mixed-effects model；逐 head/layer 探索用
sample-level permutation 的 max-statistic 或 FDR；所有 event threshold、GE/GH 方向和 detector 都在
train/calibration 冻结，test 只评估并分别报告 event coverage 与 event-conditioned 效果。

## 14. 主要方法来源

- [AtP*: scalable component localization](https://arxiv.org/abs/2403.00745)：一阶筛选的效率、
  saturation/cancellation false negatives 与 top-candidate verification。
- [Edge Attribution Patching](https://arxiv.org/abs/2310.10348)：两次 forward、一次 backward 的边筛选，
  以及一阶分数不能替代 activation patching。
- [Activation patching 的解释边界](https://arxiv.org/abs/2404.15255)：exploratory 与 confirmatory
  模式、necessity/sufficiency 和 corruption dependence。
- [Path patching](https://arxiv.org/abs/2304.05969)：用冻结路径假设检验 mediation，而不是穷举路径
  生成假设。
- [Causal tracing](https://arxiv.org/abs/2202.05262)：corrupt-restore 的节点定位逻辑。

## 扫描后的逐 head 时序路由检测

这一步先检验已经捕获的结构信息能否支持检测，不把干预结果当作选点分数。
一个统一的概率模型代替手工拼接“幻觉风险指标”：分别建模每个 `(layer, head)` 的当前
路由状态和前后状态依赖，再冻结模型评估所有 test token。

**状态。** 四桶实际 message transport 是各 source edge 的 post-WO 向量范数之和，记为
`m[t,l,h,b]`。它包含绝对量，但不是抵消后的净 residual write。状态 `z[t,l,h]` 有四维：
归一化四桶比例的三个正交 log-ratio 坐标，加 `log(sum_b m + ε)`。比例用 ε=1e-5 平滑。
这样“来源占比重新分配”和“绝对 transport 变化”具有各自坐标，不把四个相加为 1 的比例
当作独立证据。每个 head 的状态、参数和协方差都独立保留。

**时间条件。** `c[t]` 只包含常数、生成 token 索引的 `log1p/5` 及平方、已知 prompt 长度
的 `log1p/8` 及平方、两者乘积、local/remote response 是否已经可读。索引由
`response_index=q+1-response_start` 得到；没有最终回答长度、回答完成比例、右邻居峰值或标签。

**密度。** 对每个 head，拟合

\[
u_t=[z_{t-1};z_t],\qquad
p_h(u_t\mid c_t)=\mathcal N(W_hc_t,\Sigma_h).
\]

均值用固定 ridge=1e-3；8×8 协方差向自身对角收缩 5%，加小数值 floor。两遍流式统计
完成拟合。每个 source 权重相同，同 source 内样本均分权重，各样本最多 128 个均匀位置
参与估计；评分用全 token。核心接口是 `RoutingTransitionModel.fit(factory)`，它只接收
`RoutingSequence` 的状态、上下文、位置和样本权重，接口没有标签。

主分数为独立 head 因子乘积的负 log likelihood，即逐 head NLL 求和。求和是模型的独立性
假设，不是先平均 head 状态；它也没有刻画跨 head 协同。每个 head 的联合 NLL **精确**分解为
`previous-state NLL + conditional innovation NLL`。因此能同时检查进入异常状态，以及从前一
状态无法解释的变化。另在所有训练行拟合独立 4D 当前状态密度，作为 `routing_static` 对照。
`routing_transition` 只用条件 innovation。首 token 无前态，回退到当前密度，单独校准并记录数量。

train source 中固定约 20% 留作无标签校准，剩余 source 拟合。校准按任务、首 token/续写分别
估计各分数的 source 平衡均值和标准差，保留未裁剪的连续尾部分数。先按 ID 排除与 test source
重叠的 train 样本，记录排除名单。所有密度、校准和 test 预测完成后，才允许标签读取。
分数越高表示越异常，方向固定，不依据 test 结果翻转。`top_head_excess` 去掉各 head 的
协方差体积常量和模型期望能量，排序的是异常能量贡献；这不是该 head 对事实正确性的因果贡献。

**评估与独立诊断。** 逐 token AUROC、AP、各任务幻觉比例、source-cluster bootstrap 95% CI，
及同一次重采样下的主方法−对照差值。`position` 是已生成 token 索引基线。
可选 `SupervisedRoutingProbe` 用 train-fit 标签训练固定 L2 logistic readout：输入为所有 head
当前状态、因果相邻差分和上下文；另训练仅上下文的位置对照，二者使用同一行预算与权重。
固定 3 epochs、alpha=1e-3、每样本最多 128 行，不在 test 调参。其结果单独命名并单独报告，
不能作为无监督有效性的证据。

**事件审计。** `RoutingEventAudit` 只在 train 按 source 平衡、同样本同 log2 位置区间的 H−N
切换分数差选择每任务最多三个 head。各 head 在 train 的正切换分数 90% 分位数冻结阈值；
test 事件采用阈值穿越和左侧 4 行 refractory。对照为同一样本、相同中心标签、16 token 内且
避开事件附近的未使用零切换位置。分别展示四桶比例、log 总量、切换分数和最强 evidence unit
连续性；先做事件−对照，再在同 source 内做 H−N。未来 `[-8,+12]` 窗口仅用于描述，不能进入
当前 token 检测。图中区间是按 source 的逐点近似区间，未做多重比较校正；不能据此宣称机制
具有统计显著性。每个事件的 q、q+1 和 winner source IDs 另存 JSONL，方便继续检查具体信息。

尚未验证的假设：

- 幻觉在路由状态或条件转换密度中足够罕见。若错误状态常见，或训练混合分布吸收了幻觉，
  无监督密度可能检测不到；“稳定 basin”不自动等于正确，也不自动等于幻觉。
- 线性位置条件、单高斯和跨 head 独立假设足以表达正常变化。协同模式、多模态路由可能需要
  更合适的模型；不能凭合成测试或聚合均值确认。
- 四桶和每桶 winner 保留了关键判别信息。它们无法恢复完整 source 分布、hub 的事实 lineage、
  MLP 接纳、词法/语法与事实信息的解耦，也不能区分事实证据与 verbal confidence 表征。
- 时序路由是否增加检测价值，必须看固定 test 中相对位置/静态对照的指标和差值区间。
  本轮设计已看过 test cohort 汇总，指标属于探索性；正式机制确认需要未参与设计的新数据。

实现检验覆盖未来前缀不变性、联合密度链式分解、保留状态分布而打乱时序的对照、source 权重、
首 token、标签隔离、并列分数指标和完整离线工作流。合成检验不是真实 RAGTruth 检测效果。
