# Re-Anchor Transport：从“回看峰值”到“证据被采纳”的研究规划

## 0. 结论与研究对象

当前结果没有验证一条完整的 re-anchor 因果链。它支持的最窄结论是：

> 正常生成中的内部路由转折会重新访问一般 prompt，但不稳定地重新访问 RAG evidence；
> QA 的幻觉起点较少出现 evidence re-entry，而幻觉 token 在三个任务中都更容易被后续位置读取。

这更接近一个待验证的 **missed grounded re-entry + ungrounded self-anchor** 模式，而不是已经成立的
“history takeover”“evidence integration failure”或“readout silence”。后续不再把 prompt revisit、远距离
attention、future attention 任一单项称为 re-anchor。核心研究对象改为：

> 在模型内部出现信息需求时，外部证据是否被选择，是否以真实 Value/Output message 进入残差流，是否在
> 多 head、多来源的聚合中保留，是否改变候选 token 分布，是否被实际生成 token 采纳，以及这份证据条件化
> 状态是否被后续预测继续使用。

图的意义不是把若干统计量相加，而是明确表示这条有方向、有层序、有 head 身份的计算路径。主图应是
Transformer 的 **layered operator graph**，而不是 layer/head 平均后的 token attention 图，也不是需要标签
训练的 GNN。

### 当前代码落地状态（schema v8）

| 项目 | 状态 | 当前可检验的结论边界 |
|---|---|---|
| H0 conditional prompt-vs-history slope、CI 决策 | 已实现 | 可用旧 v6 artifact 重评估 |
| onset/transition 的 position、entropy、log-prob、boundary 匹配 | 已实现 | 取代 H1 的 circular-shift 主 null；循环平移只保留为敏感性分析 |
| `q=p-1` predictor reuse 与 source `p` emitted anchor 分离 | 已实现 | 必须重新 capture；旧 artifact 只有 emitted-token 量 |
| 全样本 `[layer,head,event]` attention selection 与核心 transport trace | 已实现 | 分开保存 raw attention role mass 与 `A||W_OV||` 归一化预算；后者仍不是 signed semantic contribution |
| 全样本 context-cut vocabulary candidate、JS、target gain/rank、adoption margin | schema v8 已实现，待运行 | 可回答整个 external context 改变了哪些候选；不能替代 exact support span |
| 无标签 train→test transport/adoption detector | 已实现，待完整 held-out 运行 | 主分数只用 causal-prefix 信号；future reuse 只进入 offline score；详见 `DETECTOR.md` |
| signed edge message、coalition coherence、top-k operator event | 待实现 | 完成前不能区分 transport cancellation |
| event-targeted suffix cut 与逐层 attention/MLP state accounting | 待实现 | 完成前 predictor reuse 仍是观察量，不是未来因果效应 |
| claim-specific support/distractor mask | 待数据/人工标注 | 完成前统一使用 `context`，禁止声称“准确事实已输送” |

当前代码升级的作用是先堵住会直接改变结论的坐标、聚合和统计漏洞，并为下一次 capture 生成足够的
head-resolved 与 vocabulary-functional artifact；它不是把后续 Phase 3–4 宣称为已经完成。

本文件第 1 节的数值来自 schema v7 的 test pilot：全量 broad capture，但 functional/grouped cut 仍仅为
每任务 30 条。schema v8 必须先完成独立 train/test recapture，才能据此冻结检测表示与阈值。

---

## 1. 当前结果如何解释

### 1.1 H0：注册的 direct drift 基线被否定

注册预测是：在完全正确的回答中，随 response 位置增长，capacity-adjusted prompt lift 下降、history lift
上升。实际三个任务中两条 slope 都显著为负：

| Task | prompt lift slope | history lift slope | 结论 |
|---|---:|---:|---|
| QA | -0.2005 | -1.3941 | history 相对 capacity null 下降得更快 |
| Summary | -0.3003 | -1.6062 | 同上 |
| Data2txt | -0.4247 | -1.5381 | 同上 |

因此，不能再用 H0 支持“正常生成自然从 prompt 转向 response history”。这里的 lift 不是 raw share，而是
相对于当前所有可见 source 的 `||W_O V||` availability null 的对数比。随着 response history 增长，null 中
可用 history capacity 增长可能快于模型真正选择它的速度，所以 raw history share 即使上升，history lift
也可能下降。

下一版必须同时报告：

\[
L^P_t=\log(R^P_t/B^P_t),\qquad
L^H_t=\log(R^H_t/B^H_t),
\]

以及成分闭合的条件对数优势：

\[
D_t=\log\frac{R^P_t/R^H_t}{B^P_t/B^H_t}=L^P_t-L^H_t.
\]

`D_t` 的 source-bootstrap CI 必须从每个 sample 的配对 slope 直接计算，不能用两条独立 CI 相减。

### 1.2 H1：找到的是 generic prompt transition，不是 grounded re-anchor

内部 route-change peak 与一般 prompt delta 在三个任务中稳定正相关（约 `+0.04`），但 evidence delta 在
QA、Data2txt 为负，在 Summary 为零；future influence 也接近零或略负。这说明：

1. 内部转折点确实会回到 prompt；
2. 回看的对象常是 instruction、question、格式 token 或其他 prompt 内容，而不是 RAG evidence；
3. 当前实验没有复现“回看后建立一个有持续后效的 grounded anchor”。

### 1.3 H2：最值得继续验证的是“漏掉证据入场后形成自锚点”

QA 的 hallucination onset 相对同回答 clean token 有较低 evidence entry（`-0.0277`）；Summary 和 Data2txt
为零。另一方面，三个任务的 hallucination token 都有略高的 future attention influence。这可以提出但尚未
证明以下模式：

\[
\text{demand/transition}
\;\not\Rightarrow\;
\text{grounded entry}
\;\Rightarrow\;
\text{generated token becomes a self-anchor}.
\]

这里的最后一个箭头目前不是因果箭头，只是配对相关。尤其要先修正第 2 节所述的坐标错位。

### 1.4 H3/H4：统计功效和 estimand 都不足

每个任务虽然选择了 30 个 mechanism samples，实际 onset-clean pair 只有 QA 15、Summary 9、Data2txt 38。
`evidence_effect`、`integration`、`late_control_loss` 的 CI 都跨 0 且很宽，不能据点估计命名故障类型。

Data2txt 的 `history_effect=-1.5299 [-2.7796,-0.1524]` 表示 history 对 hallucinated target 的支持相对 clean
更弱或更负，不支持统一的 history takeover。Summary 的 readout gain 差为负，但没有同时证明 final
evidence-state presence 仍高，故不能单独称作 readout silence。

---

## 2. 当前实现必须先修正的定义问题

### 2.1 prediction event 与 token-state event 不是同一节点

生成 response token `y_p` 的预测状态位于：

\[
q=p-1,\qquad y_p\sim p_\theta(\cdot\mid x_{\le q}).
\]

而未来位置读取 token `y_p` 时，读取的是输入位置 `p` 在各层形成的状态。当前代码把 `q` 上的回看变化与
未来对 source `p` 的注意配成一对。在 teacher forcing 下，`p` 的 token embedding 是外部给定的；`q` 上刚
获得的证据状态不会自动写入 `p`。因此必须分开保存：

- `predictor_reuse[p]`：未来 prediction rows 对 source `q=p-1` 的读取；
- `token_anchor[p]`：未来 prediction rows 对 source `p` 的读取（当前 FAI）；
- `emission_link[p]`：只表示 `q` 的 logits 选择了 `y_p`，不能伪装成 hidden-state edge。

teacher-forced主实验可以验证 predictor-state 的后续复用和 token-state 的后续锚定，但不能仅凭二者相关
声称证据经由采样 token 被写回上下文。若不做自由生成干预，这条 claim 必须明确停止在这里。

### 2.2 当前 evidence mask 只是整段外部上下文

`build_evidence_mask` 将 QA 的所有 passages，或 Summary/Data2txt 的整个 source，标为 evidence。它没有标出
“当前 claim 的准确支持 token”。因此当前实验最多可以说 **external-context entry**，不能说“正确事实被
输送”。后续使用两级语义范围：

1. `context_mask`：当前可直接构造的完整外部证据区，用于全量审计；
2. `support_mask[c]`：针对 claim `c` 的精确支持 span，只用于有可靠标注或人工核验的小规模机制集。

若第二级缺失，所有 `fact-grounded` 结论停止，不用 attention 自己选 support 再用同一 attention 验证。

### 2.3 当前 scalar capacity 丢失了消息内容和抵消

当前权重

\[
c_{l,h,q,s}=a_{l,h,q,s}\|W^O_{l,h}V_{l,g(h),s}\|_2
\]

只保存每条消息的范数。真实进入 residual stream 的 edge message 是：

\[
m_{l,h,s\to q}
=a_{l,h,q,s}\,W^O_{l,h}W^V_{l,g(h)}\,\widetilde r_{l-1,s}.
\]

必须保留 head 身份和有符号向量。`sum ||m||` 只能叫 transport budget；不同 source/head 可能相互增强、
正交或抵消，不能把它当作被聚合后的贡献。

### 2.4 当前主 trace 过早平均 head 和 layer

`RouteAccumulator` 先对 head 取均值；`build_rhythm` 再对 layer 取均值。一个平均后的 peak 可能不属于任何
真实 head/layer circuit，也会把方向相反的 heads 抵消。下一版事件检测在 `(layer, head, prediction_event)`
空间完成；只有在事件组件已经形成后，才汇总 component coverage、coherence 和 effect。

### 2.5 circular shift 不是充分的时间序列 null

response 越往后，可见 source 数、history role、标点密度、claim 类型和熵都在变化。循环平移 peak 会把
序列尾部事件放到序列头部，破坏 risk set。主 null 改为同一回答内匹配：相近相对位置、相近 entropy、
相近 target log-prob、相同 boundary/非-boundary 状态的非事件 token。block shift 只作敏感性分析。

### 2.6 当前 grouped cut 的真正 estimand

evidence cut 在所有 layer、所有 response-query rows 删除 evidence Value messages。因此它不只是“当前 token
的直接 evidence entry”；它同时移除了 evidence 经较早 response positions 再传到当前 position 的路径。
它测量的是 **所有 evidence-to-response path family 的总效应**，但仍保留 evidence 在 prompt 内部先转运到
其他 prompt token 的旁路。文档和变量必须据此命名：

- `direct_entry`：只观察或干预当前 predictor row；
- `response_path_family_effect`：当前全局 source cut；
- `prompt_relay_residual`：全局 cut 后仍无法移除的 prompt 内旁路。

---

## 3. 与两篇交大工作的关系，以及真正可超越的 gap

### 3.1 Preplan-and-Anchor

*Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables Fine-Grained Policy
Optimization*（ICML 2026）用 WAAD 描述局部 heads 的长距离回看，用 FAI 描述 token 被未来读取的程度，并
发现 WAAD peak 往往先于或重合于 FAI peak。论文把 heads 按平均 attention distance 分成 local/global 组，
然后在组内平均；FAI 的因果验证是替换高 FAI token 后重新 rollout。

我们的继承点是“backward consultation 与 forward influence 必须配对”。需要超越的点是：

- 它找的是 reasoning rhythm，不判断回看的来源是否为 RAG support；
- 它使用 attention weight 和 head-group average，不追踪 `W^O W^V` message 内容；
- 它证明 anchor 影响后续生成，不证明 anchor 携带的是正确外部证据；
- 它没有拆开 selection、transport、aggregation、persistence 与 readout adoption。

### 3.2 FlowTracer

*How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs*（ICML
2026）把 layer/head 平均 attention 建成 token DAG，以

\[
h(i)=\sum_{k>i}W_{ik}h(k),\qquad
W'_{ik}=W_{ik}h(k)/h(i)
\]

做 answer-conditioned Doob-h-like 重加权，再从 question 注入单位流，得到 flow throughput 和 reasoning
backbone。

它的重要贡献是从点指标进入全局路径，并显式过滤不能到达 answer 的死支路。它自己也承认 outcome-only
flow 无法区分“支持正确答案的推理”和“支持错误答案但内部连贯的推理”。我们的突破口正是：

- 不将 layer/head 折叠；
- 不把 attention 当作非负、守恒的信息实体；
- 允许 residual/MLP 对信息进行放大、抵消和改写；
- 流必须以外部 support 为根，并以对输出分布的功能作用为终点；
- 分开 grounded backbone 与能稳定支持错误 token 的 self-closed backbone。

### 3.3 Operator 依据

*Attention as a Hypernetwork* 给出：

\[
\mathrm{MHA}_q(X)
=\sum_s\left(\sum_h a_{h,q,s}W^O_hW^V_h\right)x_s.
\]

因此对同一 `(q,s)`，跨 head attention 向量不是应先平均的冗余坐标，而是配置 source-specific linear
operator 的 latent code。我们的图边应保存 `a_h` 与对应 `W^O_hW^V_{g(h)}` 的配对；只保留 `W^O W^V`
或只保留 `a_h` 都不完整。

### 3.4 相邻工作的边界

- *Thought Anchors* 同时使用 sentence perturbation、receiver-head attention 和 attention suppression，说明
  anchor 需要独立方法交叉验证；但它仍不区分 grounded 与 ungrounded anchor。
- *Information Flow Routes* 用 attribution 而非逐边 patching 高效抽取 prediction-specific circuit，启发
  “梯度筛选、精确干预验证”的两阶段设计。
- *Information Flow Reveals When to Trust Language Models* 使用 value-aware、layer-ordered contribution
  layout，但最终 trust calibration 有外部 reranker、SHAP 和 correctness-supervised XGBoost；其 attribution
  核可以作对照，监督 detector 不能成为本项目的无监督主方法。

---

## 4. Layered operator graph

### 4.1 节点与边

对每层 `l`、token position `s` 建三类 state node：

1. `r_in[l,s]`：attention 前 residual；
2. `r_mid[l,s]`：attention residual addition 后；
3. `r_out[l,s]`：MLP residual addition 后。

对每条 `(l,h,s->q)` 建 message node，边为：

\[
r_{in}^{l,s}
\xrightarrow{W^V_{l,g(h)},\;a_{l,h,q,s},\;W^O_{l,h}}
m_{l,h,s\to q}
\rightarrow r_{mid}^{l,q}.
\]

另有 residual identity edge 与 MLP write edge：

\[
r_{mid}^{l,q}=r_{in}^{l,q}+\sum_{h,s\le q}m_{l,h,s\to q},
\]

\[
r_{out}^{l,q}=r_{mid}^{l,q}+\mathrm{MLP}_l(\mathrm{LN}(r_{mid}^{l,q})).
\]

下一层 `r_in[l+1,q]=r_out[l,q]`。这个图严格有向、逐层、因果，不需要训练一个 GNN 才能进行消息传递；
Transformer 本身已经执行了消息传递。

### 4.2 为什么不对真实流强加 conservation

FlowTracer 对非负 attention flow 强制局部守恒，用于消除路径长度造成的数值衰减。但 residual stream 中
真正的信息可以被 head cancellation、MLP 放大、归一化和重写。强行守恒会删除我们正要研究的机制。
新方法只要求计算恒等式可核验，不要求 scalar mass 守恒：

\[
\delta r_{out}^{l,q}
=\delta r_{in}^{l,q}
+\delta a^{l,q}
+\delta m^{l,q},
\]

其中 `delta` 是 baseline 与 evidence-path cut 的状态差，`a`、`m` 分别是 attention write 与 MLP write。
这给出逐层精确的 evidence-conditioned state accounting。

---

## 5. 从候选到采纳：六阶段 operational definition

### S0. Demand：当前预测是否出现重新检索需求

对每个 `(l,h,p)`，`p` 对应 predictor `q=p-1`，从完整 source distribution 检测多尺度变化：

\[
C_{l,h,p}^{(w)}
=\mathrm{JS}\left(P_{l,h,p},\operatorname{barycenter}
\{P_{l,h,p-w:p-1}\}\right),\quad w\in\{2,4,8,16\}.
\]

这里只找内部 route transition / nonlocal consultation，不使用 evidence share、hallucination label 或标点。
标点、entropy、log-prob 只用于匹配 control。

### S1. Selection：模型选择了哪些候选 source

保存每个 head 的 attention gate 和 `(q,s)` 跨 head latent code：

\[
z_{l,q,s}=[a_{l,1,q,s},\ldots,a_{l,H,q,s}].
\]

分别统计 `support/context/question/history/self` source set，但不在这一步把不同 heads 合并。

### S2. Transport：候选 source 实际送入了什么

保存真实 message 或可验证的 top-k message：

\[
m_{l,h,s\to q}
=a_{l,h,q,s}W^O_{l,h}W^V_{l,g(h)}\widetilde r_{l-1,s}.
\]

对 source coalition `C` 定义：

\[
B_C=\sum_{e\in C}\|m_e\|_2
\quad\text{(transport budget)},
\]

\[
M_C=\sum_{e\in C}m_e
\quad\text{(net residual write)},
\]

\[
\rho_C=\frac{\|M_C\|_2}{B_C+\epsilon}
\quad\text{(aggregation coherence)}.
\]

`B_C` 大但 `rho_C` 小表示消息虽然大量到达，却在 head/source 聚合中抵消。二者不能互换。

### S3. Integration：证据条件化状态是否被保留和加工

运行 baseline 与 evidence-path cut，逐层保存 input、attention write、MLP write。用上一节的 exact delta
identity 判断：

- attention 写入是否建立 evidence-conditioned difference；
- residual identity 是否保留它；
- MLP write 与该 difference 同向、正交还是反向；
- 中层建立的 readout control 是否在晚层被削弱或反转。

这里不再用单一 `late_control_loss` 代替整条曲线，至少保存 signed control、presence、attention-write
alignment、MLP-write alignment 四条 layer curve。

### S4. Candidate：证据在 vocabulary 中支持哪个 token

令 `p_t(v)` 为 baseline next-token distribution，`p_t^{-E}(v)` 为切断 evidence-to-response path family 后的
分布：

\[
d_t^E(v)=\log p_t(v)-\log p_t^{-E}(v).
\]

`d_t^E(v)` 大表示外部证据使候选 `v` 更可能。保存：

- `evidence_candidate_id = argmax_v d_t^E(v)`；
- top-k candidate IDs 和 gains；
- `KL(p_t || p_t^{-E})` 或 JS，表示 evidence 是否真正改变候选分布；
- observed target `y_t` 的 `d_t^E(y_t)`。

这一步第一次把“输送了信息”与“信息支持哪个输出候选”连接起来。完整 vocab delta 只流式计算，artifact
保存 top-k 和标量，不保存 `[tokens, vocab]` 大矩阵。

### S5. Adoption：实际 token 是否采纳 evidence-supported candidate

定义：

\[
A_t^E=d_t^E(y_t)-\max_{v\ne y_t}d_t^E(v),
\]

并保存 `y_t` 在所有 `d_t^E(v)` 中的 rank/percentile。`A_t^E>0` 表示实际 token 是 evidence intervention
最支持的候选；`A_t^E<0` 表示 evidence 更支持另一个 token。该量不需要 hallucination label，方向在打开
标签前已经冻结。

为排除“删掉大量 message 只改变 activation scale”，必须同时有：

1. other-prompt path cut；
2. role/lag/transport-budget matched history cut；
3. norm-matched value-direction permutation；
4. target row only 与 all-response-row path-family cut。

### S6. Retention/Reuse：采纳后的 evidence state 是否继续发挥作用

保存两个不同的未来量：

- predictor reuse：未来 rows 对 `q=p-1` 的 message transport；
- emitted-token anchor：未来 rows 对 source `p` 的 message transport。

真正的 grounded global anchor 还需一个 event-targeted cut：只移除 re-entry event 在 `q` 的 evidence
messages，重新运行 suffix，测量未来 prediction distributions 的变化。如果未来只读取 `p`，但移除 `q`
的 grounded entry 不改变后续输出，则它是 token anchor，不是 grounded-state relay。

---

## 6. Local 与 global re-anchor 的发现算法

### 6.1 Local event

local event 是单个 `(layer, head)` 上的多尺度 route change，不是先平均后的 token peak。每个 channel 的
threshold 用其本回答内过去时刻的 robust distribution 校准，只允许使用 `<=p` 的值，避免未来泄漏。

每个 local event node 保存：

```text
(prediction p, query q, layer l, head h,
 route-change scale, nonlocal shift,
 selected top-k sources, head latent code,
 message budget, message direction sketch)
```

### 6.2 Event graph 与 global event

在 event nodes 之间连边，而不是在已经平均的 token scalar 之间连边。两个 nodes 只有在以下至少两项一致
时相连：

1. prediction time 相同或相邻；
2. source top-k 有显著 overlap；
3. message vectors cosine 同向；
4. 位于相邻层且下层 source state 可达上层 query state。

global candidate 是跨多个 depth bands 的 connected component。它的描述量是 layer coverage、head diversity、
source overlap、net message coherence 和 causal adoption，而不是 component node 数的手工加权和。

### 6.3 独立发现与验证

避免循环论证：

```text
discovery view   = attention/QK route change + nonlocal consultation
transport view   = W_O W_V message vectors
adoption view    = evidence-cut vocabulary distribution
causal view      = event-targeted suffix rerun
evaluation view  = hallucination labels（最后才打开）
```

一个 route-change component 只有通过后面至少两个独立 view，才升级为 re-anchor。否则只叫 transition。

---

## 7. “共贡献”应该如何定义

### 7.1 不采用的定义

以下量都不能单独称为 contribution：

- attention weight；
- `||W_O V||`；
- `attention * ||W_O V||`；
- message norm 的 head/source 求和；
- 能到达 answer 的非负 attention path sum；
- 删除一组边后固定 runner margin 的变化。

它们分别只是 selection、capacity、transport budget、reachability 或特定 intervention effect。

### 7.2 三层共贡献

第一层是 **vector coalition**：同一 source group 的消息是否在 residual space 中协同，使用 `M_C` 与
`rho_C`。

第二层是 **functional coalition effect**：对冻结 outcome `F_t`，比较 baseline、`-E`、`-Q`、`-(E,Q)`，
报告两个 main effects 与 factorial interaction，不把 interaction 单独当“整合成功”。`F_t` 主版本使用
target log-prob、evidence-candidate margin 和 distribution JS，而不是任意单一 runner。

第三层是 **edge attribution within a validated coalition**。先用一次 backward 得到局部 gradient 对所有边
筛选，再只对候选 coalition 用 integrated-gradient message injection：

\[
c_e^{IG}
=\int_0^1
\nabla_{a_{l,q}}F_t(a^{-C}_{l,q}+\alpha M_C)^\top m_e\,d\alpha.
\]

同一 coalition 内的 `sum_e c_e^{IG}` 具有 completeness 目标，可与 exact group intervention effect 核验。
若误差超过注册阈值，该 attribution 只作近似排序，不进入机制结论。

对正、负 contribution 分开建图；不能取绝对值后再声称“支持”。正图表示支持 evidence candidate，负图表示
抑制或竞争。

---

## 8. 幻觉机制的可证伪分类

后续报告不先合成一个 hallucination score，而是对每个 onset 给出以下状态之一或“不确定”：

| 模式 | Demand | Selection/transport | Integration | Adoption | Future |
|---|---|---|---|---|---|
| M0 无需重锚 | 低 | 可低 | — | 局部 continuation 正常 | — |
| M1 missed selection | 高 | support/context entry 低 | 无状态可整合 | evidence candidate 未形成 | self-anchor 可高 |
| M2 transport cancellation | 高 | attention 高、`B_E` 高 | `rho_E` 低 | evidence effect 低 | 弱 |
| M3 integration failure | 高 | net message 已到达 | MLP/residual 抵消 | candidate effect 低 | 弱 |
| M4 late override | 高 | 中层 evidence control 高 | 晚层被 history write 反转 | actual adoption 低 | history relay 高 |
| M5 readout silence | 高 | evidence state 到达 | final presence 高 | readout/adoption 低 | 可高 |
| M6 ungrounded self-anchor | 任意 | grounded ancestry 低 | self/history state 稳定 | actual token 不受 evidence 支持 | token FAI 高 |
| M7 wrong-evidence anchor | 高 | context 高、exact support 低 | 状态与 readout 均强 | 错误候选被采用 | grounded-to-wrong passage 高 |

M7 只有具备 claim-specific `support_mask` 与 distractor mask 时才能识别。其余模式也必须按注册阈值判定；
阈值从 label-free calibration split 或 effect-size confidence interval 得到，不能用 test hallucination label 选。

---

## 9. 注册假设与反证条件

### R0：正常生成不存在统一 prompt→history lift 方向

主分析改为报告 raw share 与 conditional role log-odds，不再预注册固定符号。若跨任务异质性大，则 position
drift 只作为 task-specific background，不进入通用机制。

### R1：内部 transition 不等于 grounded re-anchor

route-change event 对一般 prompt delta 的 lift 应显著强于对 exact/coarse evidence delta 的 lift。若 evidence
同样稳定上升，则当前“generic transition”解释被反证。

### R2：QA hallucination onset 存在 missed grounded entry

onset 的 context/support message net write、candidate distribution effect、actual-token adoption 三者至少两项
低于严格匹配 clean token。若只 attention share 低而 message/adoption 不低，则 H2 被否定。

### R3：transport 与 adoption 可以脱节

存在 `B_E` 或 `||M_E||` 高但 evidence distribution JS / adoption 低的事件；exact state accounting 应能将其
定位为 cancellation、integration loss 或 readout loss。若 transport 与 adoption 几乎单调等价，则复杂
分阶段模型没有必要。

### R4：hallucination 更可能形成 ungrounded self-anchor

hallucination onset 的 emitted-token anchor 或 self/history throughput 较高，但 event-targeted evidence cut 对
未来分布影响较低。若高 FAI token 的未来作用同样依赖 evidence entry，则 self-anchor 假设被否定。

### R5：layer/head identity 有不可替代增量

head-preserving operator graph 必须优于以下 matched controls 才能成为主方法：head permutation、layer order
shuffle/reverse、`a` 与 `W_OV` 错配、message direction permutation、layer/head mean。若真实配对没有增量，
则不再以 operator graph 作为创新 claim。

### R6：机制能跨 task 解释，而非强行同方向

QA、Summary、Data2txt 允许具有不同 mode mixture，但每个已命名 mode 的操作定义固定。若只能针对每个任务
重新选择 head、layer、符号或权重，则不构成统一机制。

---

## 10. 实验阶段与停止规则

### Phase 0：只用已有 artifact，先修正统计解释

不重新 capture 即可完成：

1. 增加 `prompt-vs-history conditional log-odds slope`；
2. 输出每个 H 的 `supported / contradicted / inconclusive`，规则写死；
3. 报告 mechanism pair 数、source 数与 CI half-width，不再只报 sample 数；
4. 为现有 onset matching 增加 position、entropy、log-prob、boundary balance diagnostics；
5. 将当前 `future_influence` 正名为 `emitted_token_future_influence`。

停止规则：如果 corrected H1/H2 只由 position/boundary mismatch 解释，不进入下一阶段的 detector 设计。

### Phase 1（已实现）：全样本 head-preserving recapture

train/test 的每个 selected sample 都保存核心 `[layer, head, event]` trace，不再只给 plot sample 保存 head 数据：

```text
attention prompt/context/history mass
message transport budget by source role
route-change score
predictor reuse
emitted-token anchor
top-k source endpoints with (layer, head, source, attention, message norm)
```

source 维不保存 dense tensor；top-k 之外保存 exact remainder mass。artifact 使用 float16 存大数组、float32
做在线计算，并记录 conservation/reconstruction error。

停止规则：若 head-preserving transition 不优于 head-mean control，停止 event graph 扩展。

### Phase 2（schema v8 已实现，待全量运行）：全样本最低成本 functional pass

每个 sample 只做 baseline + context-path-family cut，流式计算 evidence candidate/adoption，不保存 vocab 矩阵。
其余三种 grouped cuts 只在 source-diverse 子集运行，因为这一步更直接回答“context 支持什么、实际 token
是否采用”。

停止规则：若 onset 的 candidate/adoption 与 matched controls 无差异，不能把 missed re-anchor 写成主机制。

### Phase 3：分层、按 source hash 抽样的 event-targeted deep audit

不再取数据顺序最前 30 个 sample。捕获前按 `(task, source_id hash)` 固定抽样；pilot 后按目标 CI half-width
确定样本数。每个 selected sample 比较：

```text
top transition events
position/entropy/log-prob/boundary matched non-events
hallucination labels仍在capture后才打开
```

运行 target-row cut、all-response path cut、other-prompt cut、matched-history cut、direction permutation，并保存
exact residual write decomposition。

停止规则：任何 mode 必须有独立 discovery signal + causal effect + matched control；缺一项只称 candidate。

### Phase 4：claim-specific support 小规模审计

只在能取得可靠支持 span 的样本上区分 support、distractor、question/instruction。人工小集需在查看模型内部
结果前完成标注。该阶段验证“准确事实输送”，不是用来调 test detector。

### Phase 5：冻结表示后才研究无监督 detector

机制成立前不再堆 one-class/GNN detector。冻结的 token representation 至少包含各阶段的原始可解释量和
mode posterior；无监督 scorer 在 calibration sources 上拟合，test label 最后打开。必须与 position-only、
confidence、raw attention、WAAD/FAI、FlowTracer attention-only、ALTI/value-aware baseline 比较。

---

## 11. 统计协议

1. capture、event selection、threshold 和 artifact hash 全部在 label firewall 内完成；
2. bootstrap unit 是 `source_id`，同 source 的多个 response/token 不能被当作独立样本；
3. onset 与 clean matching 至少平衡 response position、token log-prob、entropy、boundary 和 token identity；
4. 每个 task 单独报告，再报告随机效应 meta-analysis；不把 task 混合后得到的显著性当跨任务机制；
5. 除均值 CI 外，报告 source-level sign fraction 和 mode prevalence；
6. 多个层/head 只允许预注册的 hierarchical test 或 max-stat permutation correction；
7. discovery 与 causal validation 使用不同信号；不允许以 evidence pulse 定义事件，再用同一 pulse 宣布成功；
8. 所有 point estimate 必须伴随有效 source 数、pair 数和 CI；少于注册最小 pair 数只输出 exploratory。

---

## 12. 最小代码结构

当前目录已有过多相邻模块。新实现不继续平行堆文件，核心收敛为：

```text
capture.py      orchestration、坐标与 artifact schema
routes.py       head-preserving selection/transport trace + local/global events
mechanism.py    grouped/event cuts、state accounting、candidate/adoption
report.py       label-after-capture matching、CI、hypothesis decisions
visualize.py    operator-event graph、layer curves、candidate/adoption panel
run.py          单一 CLI
```

迁移完成后，`rhythm.py`、`message_norm.py`、`events.py` 中仍被使用的函数并回上述核心文件；旧模块只在 schema
迁移期保留，不长期维护两套定义。

### Artifact schema v8（当前实现）

每个 sample 的 NPZ 至少包含：

```text
coordinates:
  prediction_position[T], predictor_position[T], emitted_position[T]

head trace:
  head_attention_{prompt,evidence,history}_mass[L,H,T]
  head_{prompt,evidence,history}_transport_share[L,H,T]
  head_route_change[L,H,T]
  head_predictor_reuse[L,H,T]
  head_emitted_token_anchor[L,H,T]

functional:
  evidence_effect[T]
  context_distribution_js[T]
  context_target_logprob_gain[T]
  context_candidate_id[T,K]
  context_candidate_logprob_gain[T,K]
  context_target_rank[T]
  context_target_log_rank[T]
  context_adoption_margin[T]

grouped subset:
  evidence_state_presence[L+1,T]
  evidence_state_control[L+1,T]
  evidence_readout_gain[T]
  {other_prompt,prompt,history}_effect[T]
  evidence_prompt_interaction[T]
  evidence_late_control_loss[T]
```

top-source operator event、signed residual write 与 attention/MLP accounting 仍属于后续 schema；v8 不宣称
已经实现这些字段。v8 manifest 固定 model、dtype、数据根目录、token alignment 和 capture 参数，并使用新
output 目录隔离不同配置。

---

## 13. 必须生成的图，而不是只输出数字

### 单样本 operator-event graph

- 横轴为 token/prediction time，纵轴为 layer；
- node 是 local route-change event，颜色表示 source role，大小表示 net message，不用 attention mass 冒充；
- event component 用 source overlap / message alignment 连边；
- 在同一图上标出 predictor reuse 与 emitted-token anchor，但使用不同图形；
- hallucination label 只在评价版本叠加，不进入图构造。

### 单 token 六阶段面板

对任一 onset/clean token 展示：source selection → top message vectors → coalition coherence → layerwise state
accounting → vocabulary evidence candidates → future causal reuse。这个图是机制审计的主可视化。

### 群体图

1. transition-centered head/layer heatmap；
2. onset vs matched-clean 的六阶段 effect forest plot；
3. mode mixture by task；
4. real operator graph 与 head/layer/direction controls 的 effect delta；
5. predictor reuse 与 emitted-token anchor 的分离散点图。

---

## 14. 当前允许与禁止的论文表述

当前允许：

> Generic internal route transitions revisit the prompt but do not consistently re-enter the retrieved evidence.
> QA hallucination onsets show reduced direct evidence entry, while hallucinated tokens receive slightly greater
> downstream attention; whether this reflects an ungrounded self-anchor requires coordinate-corrected causal tests.

当前禁止：

- 正常生成必然从 prompt 漂移到 history；
- 已经发现通用 missed re-anchor；
- evidence 已经到达但未整合；
- history 覆盖了 evidence；
- residual 中仍有 evidence 但 readout 沉默；
- FAI/attention throughput 是“准确事实贡献”；
- 当前 30-sample deep audit 已经识别四类机制。

只有完成 Phase 2–4 后，才能把“信息选择—输送—聚合—采纳—复用”的哪一段失败写成主要机制发现。

---

## 15. 参考实现边界

- Li et al., [Attention Illuminates LLM Reasoning](https://arxiv.org/abs/2510.13554), ICML 2026.
- Dong et al., [How Does Reasoning Flow?](https://arxiv.org/abs/2606.10646), ICML 2026.
- Schug et al., [Attention as a Hypernetwork](https://arxiv.org/abs/2406.05816), NeurIPS 2024.
- Ferrando and Voita, [Information Flow Routes](https://arxiv.org/abs/2403.00824), EMNLP 2024.
- Bogdan et al., [Thought Anchors](https://arxiv.org/abs/2506.19143), ICLR 2026.
- Xu et al., [Information Flow Reveals When to Trust Language Models](https://openreview.net/forum?id=vd8HzoFZ7v), ICML 2026.

这些工作提供对象和对照，不提供本项目的结论。新方法的核心增量必须由真实 operator pairing、
evidence-conditioned state accounting、vocabulary-level adoption 与 matched causal controls 共同证明。
