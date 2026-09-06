# 机制审计注册：source 中继、target 作用与生成依赖

## 1. 状态与研究问题

本文注册机制假设、测量量和结论门槛。它区分当前 native source-cut 能回答的问题与仍需
matched factual counterfactual 的问题；下文写出的未来分析不得被报告成已经实现。

对每个 teacher-forced target token，研究四个问题：

1. selected-source effect 是直接到达 target，还是先经过 prompt/response hub？
2. selected-source intervention 是否改变 residual/attention/MLP，这些改变是否推动 target readout？
3. 该影响是被后层保留、抵消或覆盖，还是逐渐让位于 response-history 依赖？
4. native source-mediated observed margin 与 response-origin supporting action 能否区分 hallucinated 与 clean
   target？

“目标没有直接回看 prompt”不是失败条件。只有 selected-source/factual intervention 对 hub state
有可复现效应，且 hub→target patch/block 闭合 mediation，才构成对应 operator 下的 causal relay；
只有 matched factual pair 才可进一步命名 grounded route。强 attention、强 residual change 或
稳定的 response reuse 本身都不能证明 grounding。

## 2. 为什么三个常用代理量单独不够

### 2.1 Direct attention

attention weight 只给出 QK 路由门；真正写入 residual 的是逐 head 消息

\[
m^{l,h}_{j\to i}
=W_O^{l,h}\!\left(A^{l,h}_{i,j}V^{l,g(h)}_j\right).
\]

同样的 attention destination 可以经 OV 写入相反方向。IOI 电路中的正、负 Name Mover
heads 是直接反例；path patching 也表明应检验计算路径而非只看 attention map
（[Wang et al., 2023](https://arxiv.org/html/2211.00593v1)）。raw attention 与模型输出常不
构成忠实解释（[Jain and Wallace, 2019](https://aclanthology.org/N19-1357/)）。因此候选筛选
必须保留 `(layer, head, source, target)`，不得先对 head 平均。

### 2.2 Residual difference

\[
d_v=h_v^{clean}-h_v^{counterfactual}
\]

只能说明干预改变了该位置的状态，即 operator-specific state displacement；它还可能包含词法、语法、长度、
实体类型和下游非线性交互的差异。它不说明其中哪部分被 target 使用，也不等于独立的“证据
语义向量”。

### 2.3 Gradient

\(\nabla_vF_t\) 只测给定基点和给定目标下的局部 sensitivity。它依赖 contrast/loss，可能受
饱和和高阶交互影响；即使基点激活不含与当前 contrast 可分离的表征，也可以有大梯度。可扩展的
筛选量是 activation patching 的一阶近似

\[
\widehat{IE}_v=
\left\langle \nabla_vF_t\vert_{cf},
v^{clean}-v^{cf}\right\rangle,
\]

但 top candidates 仍须 exact patching。方法选择会显著改变 localization 结果
（[Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042)）；AtP 也存在重要 false-negative
模式（[Kramár et al., 2024](https://arxiv.org/abs/2403.00745)）。

## 3. 固定目标与 hub-aware 四账模型

对 predictor 位置 \(q\) 固定两个 token，并在所有 rerun 中保持不变：

\[
F_t=z_q(a)-z_q(b).
\]

在 native audit 中，\(a\) 是 observed token，\(b\) 是未干预 native run 中排除 \(a\) 后的
frozen top runner。未来 factual-pair audit 中，主分析应把 \(a,b\) 注册成 grounded 与 matched
counterfactual candidate；两种 estimand 不得混报。

### 3.1 路由账：selected-source lineage 从哪里来、经哪里中继

对每个 destination row，以逐 head message norm 与 residual continuation 构造显式但仅用于
screening 的 transport law：

\[
p_e=\frac{\lVert m_e\rVert_2}
{\lVert r_{l,i}\rVert_2+\sum_{h,j}\lVert m^{l,h}_{j\to i}\rVert_2},
\qquad
p_r=\frac{\lVert r_{l,i}\rVert_2}
{\lVert r_{l,i}\rVert_2+\sum_{h,j}\lVert m^{l,h}_{j\to i}\rVert_2}.
\]

每个节点维护 `evidence / other_prompt / response / unobserved` provenance：

\[
Z_{l+1,i}=p_rZ_{l,i}+\sum_{h,j}p^{l,h}_{j\to i}Z_{l,j}
+p_{sink}e_{unobserved}.
\]

这使 `evidence → hub → target` 与 direct evidence read 使用同一本 lineage 账。它是 norm-based
拓扑模型，不是隐藏向量的语义/概率分解，也不是因果效应。事实回忆研究确实发现了“早层 MLP 丰富
subject、关系传播、上层 attention 从 subject state 抽取属性”的多阶段过程，说明最终位置无需
直接复制原始事实 token（[Geva et al., 2023](https://aclanthology.org/2023.emnlp-main.751/)）。
`carrier_scope` 默认是 `all`；若缩减 scope，未表示的 prompt destination 在下一层进入
`unobserved`，不能沿用其初始 source 身份。

### 3.2 状态位移账：source intervention 是否改变 hub

\[
P_v=\lVert h_v^{clean}-h_v^{cf}\rVert_2,
\qquad
P_v^E=\lVert \Pi_E(h_v^{clean}-h_v^{cf})\rVert_2.
\]

当前 native audit 只实现未投影的 (P_v)，它注册为 source-cut state displacement。投影
\(\Pi_E\) 只能由训练集或预先固定的 factual contrasts 学得，并在 held-out 样本上使用；即使如此，
高 displacement 也不自动证明存在可分离的事实表征或该改变被 target 使用。

### 3.3 Target-action 账：source-conditioned change 是否推动 target

逐 head/MLP 保留 signed action：

\[
c_k=\langle \nabla_{u_k}F_t,\Delta u_k\rangle,
\]

其中 \(u_k\) 是 head 或 MLP 对 residual 的真实写入。它用于筛选支持、反对和相互抵消的模块。
最终结论来自 corrupt-restore：

\[
TE=F_t(clean)-F_t(cf),
\quad
IE_v=F_t(cf;v\leftarrow v^{clean})-F_t(cf),
\quad
U_v=\frac{IE_v}{TE},\qquad |TE|>\tau_{dtype}.
\]

并对 `source → hub → target` 做 path patch/block。高 displacement、低 exact mediation 表示
source intervention 在 hub 留下状态差异，但该差异未被证明对 target 有因果作用；低 direct prompt
attention、但高 hub-mediated rescue 可以是合法中继。clean/corrupt/restore 的因果
追踪范式来自 [Meng et al., 2022](https://proceedings.neurips.cc/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)。

### 3.4 下游账：target-supporting effect 是否被增强、抵消或覆盖

沿后续层记录 source-cut-conditioned residual、attention 与 MLP action。模块抵消率注册为

\[
C_{l,i}=1-\frac{\left|\sum_k c_{l,i,k}\right|}
{\sum_k|c_{l,i,k}|+\epsilon}.
\]

还必须保存 \(\sum_kc_k\) 的符号；`C` 高表示冲突，低表示当前 margin 下的功能对齐。早层
displacement/rescue 为正而后层净 action 转负，注册为 `integrated_then_overwritten`。该账不以
“heads 关注相同 token”为一致性，也不跨 global/local heads 求平均。

## 4. Cache → retrieve 的启发与边界

[Kumaran et al., *How do LLMs Compute Verbal Confidence?*](https://arxiv.org/html/2603.17839v3)
在 Gemma 3 27B 和 Qwen 2.5 7B 上用 steering、corrupt-restore patching、mean noising、
interchange swap、probing 和 attention blocking，支持以下序列：answer tokens 把 confidence-relevant state 汇入第一个
post-answer-newline token（PANL），PANL 缓存该状态，后续 confidence-colon（CC）再取回并
verbalize。Gemma 3 27B 中，answer→PANL 的阻断效应集中在约 L22–28，PANL 因果干预峰约
L25–26，PANL→CC 的阻断效应晚至约 L30–36；这些层段针对论文的 Gemma 3 minimal numeric
prompt。长 categorical prompt 中的多跳模板路由是论文对 direct block null result 的可能解释，
而非已定位的固定路径；因此 direct attention 缺失不能排除 hub relay。

可迁移到本审计的是实验逻辑：

1. 先用位置/层序和 per-head route 提出 cache node；
2. 用 clean/corrupt patch 检验 cache 是否足以 rescue；
3. 用 noising/block 检验必要性；
4. 用 matched interchange swap 排除一般内容扰动；
5. 区分“可 probe”与“对行为有因果作用”。原论文也发现某些位置可解码 correctness/confidence，
   但 steering/patching 无效。

不能迁移的是语义结论：该论文 corruption 的是 answer tokens，目标是 verbal confidence，未
操纵外部 supporting evidence，也未证明 PANL 缓存的是 grounded truth。其后续研究发现
verbal confidence 和 post-answer state 对后续 commit/abstain 的预测强于 correctness，且 decision
与 correctness probe directions 近正交；因此 cache 可能主要包含 commit-readiness
（[Kumaran et al., 2026](https://arxiv.org/html/2606.29490v1)）。本项目必须分别测
`evidence/correctness`、`verbal confidence` 与 `commit/response-history`，不得把强 cache 当作
grounding。

## 5. 稳定性不等于 attractor

跨层 head action 同号、低抵消或 residual update 变小只能命名为 `functional alignment` 或
`low incremental change`。只有显式扰动传播增益 \(\rho<1\) 才可称局部 perturbation damping；
Transformer depth 由不同参数的层组成，不是同一动力系统的反复迭代。teacher-forced DAG 可以
包含 prior-response→later-token 路径，但不识别 free-running 闭环自强化。

固定输入下可探索局部扰动增益

\[
\rho_{l\to l+k,t}=
\frac{\lVert\Pi_E(h'_{l+k,t}-h_{l+k,t})\rVert_2}
{\lVert\Pi_E(h'_{l,t}-h_{l,t})\rVert_2+\epsilon},
\]

但 \(\rho<1\) 仍不足以称为 attractor。该术语至少需要 future free-running 实验：同 prompt
多次采样、在分叉前后做 evidence/hub/history 扰动、比较 correct→hallucinated 与
hallucinated→correct 的双向恢复，并测试持续窗口干预。未经这些实验，不报告“推理谷底”或
“幻觉吸引子”。

## 6. 当前 native source-cut 的可识别范围

当前 native 管线已经能够、且只能精确声称：

- 在标签封闭的 teacher-forced 单世界中，固定 observed-token vs frozen-runner margin；
- 将某个 source unit 的 token positions 在所有层作为 Value-message source 删除；Q/K 不被
  直接 mask，attention 不重新归一，但 source state 与后续 Q/K 会在 cut world 中自然变化；
- 测量该特定 Value-source-cut operator 对 margin 的 necessity，并对筛出的 root、corridor 和
  carrier 做 cut/restore/block rerun；
- 计算逐 head edge/action 与 source-cut stage difference，并在分析模型内建立 provenance
  screening；schema-2 artifact 保存完整的 hub/state ledger 和逐 head 坐标。现有 provenance
  传播可表示 hub relay，但这不自动使一个候选成为经 exact factual intervention 验证的 evidence hub。

因此当前结果可以称为 `source-unit dependence`、`candidate corridor` 或通过既定 rerun 门槛的
`native causal chain`；不能声称 target 因而得到正确事实 grounding。RAGTruth 没有为每个 target
提供 token-aligned corrected fact world，observed token 也可能是 hallucinated。native source cut 同时改变 source
内容及其后续非线性交互，不能排除纯词法/语法贡献。

## 7. Future：matched factual counterfactual

升级到 evidence-specific 结论前，必须预先构造并冻结 paired worlds：

- clean/counterfactual 使用相同 tokenizer 坐标、序列长度和 response；
- 只替换明确 factual value、entity 或 relation span，尽量匹配 token 数、实体类型和语法角色；
- \(a,b\)、target、candidate units、layer/head 筛选阈值均在查看 hallucination labels 前冻结；
- 加入 grammar/template-preserving、随机同位置 span、other-prompt 和 response-history controls；
- 无法 token-align 的自然语言 corruption 不进入原位 patching 主分析；
- 在 clean 与 counterfactual 两个方向都检查 patch/block，并报告原始 margin，而非只报归一化
  recovery。

matched pair 的作用不是假装获得“纯语义向量”，而是让共同词法/语法最大程度抵消，并使
state displacement、target action 与 exact causal effect 具有可比较坐标。

## 8. 预注册可视化与判据

当前 native subset 为每个 target 输出以下四联图；第四栏验证的是 native source-cut operator，
不是 factual correctness。matched factual pair 尚未完成时，图中不得把 native carrier 改称为
verified evidence hub。

| 面板 | 固定编码 | 允许结论 |
|---|---|---|
| A. Layer-unrolled route | connected root→target widest backbone 含 residual steps，再补 top-throughput marginals；颜色为 root-lineage allocated action | 候选路由，不是因果证明 |
| B. Head-resolved target action | target row 的 layer×head selected-root/response-origin action；opacity 由 transport×coherence 给出 | 分辨逐 head 支持、反对与抵消，不平均 heads |
| C. Source-cut integration | target row 的 residual/attention/MLP displacement 与 action，并列 module/continuity diagnostics | 区分“被 cut 改变”与“局部推动 target” |
| D. Intervention ladder | clean、cf/source-cut、hub restored、hub/path blocked 的固定 margin | 因果链是否闭合 |

后续汇总多个 target artifact 时可另画 token-time 图，分别显示 selected-source-mediated action 与
response-origin supporting action；不得把一次 forward 的 DAG 画成反馈环。

### 8.1 Verified re-anchor / evidence hub

一个节点只有同时通过以下门槛，才注册为 `verified evidence hub`：

1. **Route**：存在来自 factual evidence 或已有 evidence provenance hub 的逐 head 连通路径；
2. **Displacement**：matched factual contrast 在该节点的 state displacement 显著高于全部预注册 controls；
3. **Target effect**：clean hub patch 能 rescue 固定 grounded margin，hub noising/block 能破坏它；
4. **Mediation/retention**：hub→target path patch/block 方向一致，且该影响未在下游被反向覆盖。

只通过 1–2 的节点称 `candidate cache/hub`；只通过 native source-cut 的节点称 `native carrier`。
任一单个 attention peak、probe、gradient peak、residual norm peak 或跨 head 平均都不能升级命名。

### 8.2 Hallucination 检测假设

在机制捕获全部冻结后才解封 labels。预注册两个轴：

\[
G_t=\text{native source-mediated observed margin},
\qquad
R_t=\text{response-origin supporting-action candidate}.
\]

`R_t` 高本身可属于正常生成；`R_t` 高且 `G_t` 低目前只能称 response-reliance candidate。
真正的生成自我强化主张需要独立的 response-history cut/patch。当前 post-hoc report
分别评价这两个预注册 raw axes，不拟合分类器，也不把二者合成为风险分数。后续探索才可预先
固定 \(R_t-G_t\)，并与 target position、native
margin、entropy、attention-only、message-norm-only 和 stability-only baselines 比较。
主评价使用 hallucination onset 与 position/entropy-matched clean targets、按 source 聚类的
bootstrap；在 free-running 扰动完成前，不把相关性结果报告成 hallucination attractor。
