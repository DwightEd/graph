# ETCC：Evidence-to-Target Causal Corridor

## 1. 研究对象与结论门槛

ETCC 不再把“回看峰值”直接称为 re-anchor。它审计一条更严格的链：

```text
source-unit candidate
→ target-specific route
→ multi-layer carrier
→ fixed candidate margin
→ exact cut / patch / block
```

只有 source root 的双向 patch、连通 corridor 的 message patch、以及 carrier 或
terminal block 三者方向一致时，才能说某个 evidence unit 对 target 具有机制作用。
attention、message norm、gradient score 和 graph throughput 都只用于提出候选。

旧 schema-v8 的 frozen detector 是停止扩展的诊断基线。其 held-out 结果可以说明旧特征有
弱相关性，但不能回答准确证据是否进入、经何处中继、以及是否决定 target。

### 1.1 旧流程偏差审计

低 AUPRC 不应简单归因于分类器不够强。旧实现的若干 estimand 本来就没有对应到上述
机制命题：

| 旧步骤 | 与原假设的偏差 | 对结果的影响 | ETCC 修正 |
|---|---|---|---|
| `A * ||W_O V||` route share | 非负容量替代了真正进入 residual 的向量及其 target sign | 高 norm、相互抵消或与 readout 正交的边都可得高分 | 保存 `AV` code、post-`W_O` norm、合成 coherence 与 signed path-gradient |
| prompt/evidence/history role 汇总 | passage、sentence、field、token 身份及连通路径消失 | 只能说“回看 prompt”，不能定位 source root | 每个 token 绑定 source unit，并保存逐层逐 head 精确边 |
| route-change peak | 通用路由突变被当作 re-anchor attempt | transition 可与事实选择无关，且样本内峰值规则仍会制造候选 | 不先找峰；对预先固定 target 直接计算 source-to-target flow |
| future influence | 后续 token 回看当前 response state | 测到的是 response token 被复用，不是 evidence lineage 被中继 | 用 root-conditioned `T(v|u,t)` 找多层 carrier，再做 state patch/block |
| 全 context/group cut | 多个 source 与所有通路同时删除 | margin 变化无法归因到具体 root、carrier 或 corridor | joint world 只筛 root，随后隔离单 root 并替换精确 message corridor |
| emitted-token/runner readout | runner 随基线产生，且没有受控 clean/corrupt fact pair | 能测分布敏感性，不能证明正确事实候选被采用 | 运行前固定 `a,b,origin`，使用同坐标 paired worlds |
| onset 检测评价 | 极稀疏 onset label 检验上述代理特征 | onset AUPRC 接近 prevalence；即使 AUROC 尚可也不能升级为机制证据 | labels 退出机制 schema，只在完整审计后做外部评价 |

因此旧结果包含两类信息：它可靠地否定了“代理特征足以检测重锚定”这一版本；但它既不
否定模型存在 evidence corridor，也不能用于选择 ETCC 的 edge coverage、root limit 或因果
阈值。后续效果首先取决于能否构造合法 paired worlds，而不是继续拟合旧 onset score。

## 2. 输入：配对计算世界

每个审计输入是一个 `PairedWorld`：

- clean 与 corrupt 使用同一 tokenizer、相同 token 长度和相同位置坐标；
- 两个世界的 teacher-forced response 完全相同；
- prompt 只允许在显式列出的 `candidate_unit_id` 内不同；
- target 在 predictor `q` 上定义，实际预测位置严格为 `p=q+1`；
- 固定功能量为

  \[
  F_t=z_q(a)-z_q(b),
  \]

  其中 `a`、`b` 和 `contrast_origin` 在运行前固定。

这既防止未来 token 泄漏，也使 clean carrier 可以在相同 layer/token 坐标原位 patch。
若自然语言 corruption 无法保持 token 对齐，就不能直接进入本管线；必须先构造长度匹配的
受控替换，或换成位置可对齐的 activation baseline。

## 3. Source units

`units.py` 提供四种可审计的语义单元：

| 任务 | 默认 unit | 代码 |
|---|---|---|
| QA | passage | `passage_spans` |
| Summary | sentence | `sentence_spans` |
| Data2txt | nested leaf field / list item | `field_spans` |
| 已知精确事实 | evidence span | `UnitSpan` / `SourceUnits` |

分隔符被确定性分配到相邻 unit；prompt 的非证据 token 属于 `other_prompt`；response
source position 各自拥有独立 carrier unit。每个 causal source token 恰好对应一个 unit。

`candidate_unit_id` 只是待筛选的受控 source 候选，不预先宣称它们是真正 root。

## 4. 同一图的两种 edge backend

图节点是 `(layer, token position)`，attention edge 保留完整
`(layer, head, source, target)` 坐标，vertical edge 表示 residual continuity。代码绝不在
保存前对 layer/head 求均值。

### 4.1 Attention backend

```bash
--flow-signal attention
```

候选边分数严格为 clean world 的原生 softmax attention：

\[
s^{A}_{l,h,s\to q}=A^{+}_{l,h,q,s}.
\]

它只回答“路由门选择了哪里”，不声称信息内容、方向或功能作用。该模式不运行 gradient
attribution；`selector_score`、`content_score` 和 stage target score 因而为空或 NaN。

### 4.2 Message backend

```bash
--flow-signal message
```

真实进入 attention head 输出的 pre-\(W_O\) code 与 residual message 分别为

\[
c^{\pm}_{l,h,s\to q}=A^{\pm}_{l,h,q,s}V^{\pm}_{l,g(h),s},
\qquad
m^{\pm}_{l,h,s\to q}=W^O_{l,h}c^{\pm}_{l,h,s\to q}.
\]

沿 corrupt-to-clean embedding path 对固定 margin 取平均梯度。边的 signed screen 是

\[
\phi_e=
\left\langle
\overline{\nabla_{c_{l,h,q}}F_t},
c^+_e-c^-_e
\right\rangle.
\]

这等价于在 residual 坐标中使用 \(W_O^T\nabla F_t\)，因而过滤高 norm 但与 target
readout 正交的边。它是 path-gradient screen；精确结论仍来自后面的 rerun。

对 \(A\) 和 \(V\) 的 clean/corrupt 二因素使用对称分解：

\[
\phi_e=\phi_e^{selector}+\phi_e^{content}.
\]

代码以四个组合 `A+V+`、`A+V-`、`A-V+`、`A-V-` 计算二者，并测试该恒等式。

在每个 `(layer,target)` 内，保存的 edge messages 还形成三项聚合量：

\[
B=\sum_e\lVert m_e^+-m_e^-\rVert_2,
\qquad
M=\left\lVert\sum_e(m_e^+-m_e^-)\right\rVert_2,
\qquad
\rho=M/(B+\epsilon).
\]

`B` 是 transport budget，`M` 是真正合成后的 residual write，`rho` 是 coherence。
同时分开累加正、负 target score；因此高 attention 或高 budget 但低 coherence、负 score 的
route 不会被解释为“准确信息已整合”。

## 5. 从 source contribution 到多路由 throughput

每个 destination row 先取 backend magnitude

\[
w_e=\begin{cases}
A_e,&\text{attention}\\
|\phi_e|,&\text{message}.
\end{cases}
\]

若该 row 有非零质量，则 residual continuation 获得 `1/2`，其余 `1/2` 按所有
head/source 的 \(w_e\) 分配：

\[
P(e)=\frac{1}{2}\frac{w_e}{\sum_{h,s}w_{h,s}},
\qquad P(residual)=\frac12.
\]

零 message row 全部走 residual。每个 head row 只保留达到 `--edge-coverage` 的最小边集；
被裁掉的质量进入 unobserved sink，绝不重新归一化到已保存边。

从 target 反向传播单位 route mass，得到 source-unit contribution

\[
C(u\to t)=\Pr(\text{retained reverse path ends in }u).
\]

再计算每个 node 到指定 root 的 reach probability。条件 throughput 为

\[
T(v\mid u,t)=
\frac{
\Pr(t\rightsquigarrow v)\Pr(v\rightsquigarrow u)
}{C(u\to t)}.
\]

这是统一的候选路径模型，不是“真实向量消息守恒”假设。真实 residual message 可以被
MLP 放大、反转或抵消；这些现象由 stage ledger 和因果 rerun 检验。

默认 `--carrier-scope all`，因为 `response` 便宜模式会主动忽略 prompt-to-prompt carrier，
只能作为消融。

## 6. Root discovery

候选 unit 先按 `C(u→t)` 排序。前 `--root-screen-limit` 个 unit 分别接受两个 layer-0
embedding intervention：

unit 内每个 evidence token 的 root mass 是 `reverse_node_visit[0,s]`；unit score 是这些
token mass 的和。因此既可在 passage/sentence/field 层选择 source，也保留真正进入多条
target path 的 token roots，而不是把整段 context 当成一个不可解释 mask。

\[
N_u=F(x^+)-F(x^+_{u\leftarrow -}),
\qquad
S_u=F(x^-_{u\leftarrow +})-F(x^-).
\]

以 paired effect 的方向对齐后，root causal score 定义为

\[
R_u=\min(\operatorname{sgn}(F^+-F^-)N_u,
         \operatorname{sgn}(F^+-F^-)S_u).
\]

优先选择 `R_u>0` 且 route mass 非零的 unit；若没有，只返回 routing fallback，并在
artifact 中保留每个候选的 `evaluated` 与负结果，不能把 fallback 写成 confirmed root。

若输入同时改变多个 candidate units，这个 joint world 只用于 root screening。选中 unit 后，
代码自动把其他候选恢复为 clean token，构造只改变该 unit 的 isolated world，并重新运行
gradient、message capture、throughput、carrier 与 corridor。最终路径因此条件于一个 source
root，不会把其他 passage/sentence 的 Value 变化归到它名下。artifact 同时保存 screening
corruption 与 isolated corruption。

message backend 还保存 input-state path-gradient score，供比较 gradient screen 与真实
root patch 是否一致；attention backend 对该字段写 NaN。

## 7. Integration 与 carrier

clean/corrupt 两个世界保存每层三个明确的 computation node：

```text
layer_input
attention_write
mlp_write
```

对同一 target gradient 保存 endpoint delta 的 norm 与 signed score。它们定位 source
conditioned difference 在哪一层建立、经 MLP 增强或反向、以及何时对 readout 沉默。

排除 selected source root 后，prompt/response 中位于 target 之前的 layer-unrolled node
先按 `T(v|u,t)` 提名。message backend 进一步要求 `layer_input` signed score 与 paired
effect 同向，并按 throughput 与 functional score 的乘积排序；attention backend 严格只按
attention-derived throughput 排序，不偷用 message gradient。随后两种 backend 都执行相同的
真实 state 干预：

1. 在 clean world 把 carrier 换成 corrupt state，测 source-conditioned carrier necessity；
2. 在 corrupt world 原位 patch clean carrier state，测 rescue；
3. 删除该 carrier 在当前及后续层发往所有 causal receivers 的消息；
4. 后续层把 carrier 自身重置为 corrupt state，阻止 residual 旁路。

block 条件使用 difference-in-differences：同时运行 `corrupt + block` 和
`corrupt + clean carrier + block`，`blocked_rescue` 是二者之差，不把删除原生 corrupt
carrier messages 的主效应误算成 mediation。necessity、rescue 与
`rescue - blocked_rescue` 同向、且 `abs(blocked_rescue)` 不超过 dtype tolerance 时，
才构成完整 carrier mediation。这里切断的是该 position 作为 Value source 的所有
attention message，并在后续层重置 residual state；若 key/selector 交互仍产生功能作用，
`blocked_rescue` 会保留该作用，carrier 不会被误报为完整 message carrier。单独的高
throughput、gradient score 或 norm 均不足以命名 carrier。

## 8. Corridor 的精确确认

对选中 root，`T(e|u,t)>0` 的边与隐式 residual continuation 共同在 augmented graph 中
形成 connected corridor；稀疏 message 边集合本身不必连通。所有 intervention 都在
post-softmax、pre-Value-sum 位置删除精确 `(layer,head,query,source)` 边，并在同一 head
的 pre-\(W_O\) output 原位加入 paired code：

- **positive control**：clean/corrupt 边分别删除后补回同世界 code，两个 margin 都必须在
  dtype tolerance 内恢复；
- **necessity**：clean corridor 改成 corrupt codes；
- **sufficiency**：corrupt corridor 改成 clean codes；
- **block**：保持非 terminal corridor 为 clean，但把进入 target 的 corridor edges 冻结为
  corrupt codes；
- **mediated sufficiency**：`sufficiency - blocked_sufficiency`。

若 `corridor_restoration_valid=false`，该样本的 corridor 因果结果无效，只能用于调试。
coverage 小于 1 时，未选路径导致的残余效应必须报告，不能把 corridor 当作完整机制。

## 9. 允许的结论

| 证据 | 最多允许的表述 |
|---|---|
| raw attention | route candidate |
| nonzero true message / norm | transported message candidate |
| signed path gradient | target-aligned screen |
| `C` / `T` | retained source-to-target corridor candidate |
| root necessity 与 sufficiency 同向 | functional source root |
| carrier rescue 且 block 消失 | functional carrier |
| corridor patch、block、restoration 均通过 | target-specific causal corridor |

“准确事实被采用”还要求 clean/corrupt evidence、candidate token 语义和 supporting span 来自
受控数据或独立标注。RAGTruth 的粗 context 与 hallucination span 本身不足以提供这些对象；
不得用 attention 自己选择 support，再用相同 attention 宣布其正确。

## 10. 实现映射

| 步骤 | 唯一正式实现 |
|---|---|
| 配对世界、target contract | `worlds.py` |
| passage/sentence/field units | `units.py` |
| path-gradient computation nodes | `attribution.py` |
| attention/message edge capture | `flow.py` |
| `C(u→t)`、`T(v|u,t)` | `throughput.py` |
| root/carrier/corridor rerun | `corridor.py` |
| orchestration 与 schema | `audit.py` |
| CLI | `run.py corridor` |

完整字段定义见 [`SCHEMA.md`](SCHEMA.md)。

## 11. Native RAGTruth subset：单世界机制验证

受控 pair 仍是“正确事实候选被采用”的最高证据，但不应成为真实数据 pilot 的手工瓶颈。
`run.py subset` 因此实现了一个独立的 native audit；它不伪造 clean/corrupt 文本，也不把
hallucination label 当作 pair。

### 11.1 冻结 cohort 与 target

程序在 `retain_embedded_labels=False` 下按 task 和 `source_id` 做确定性抽样。对截取后的每个
teacher-forced response，先运行未干预模型，并冻结

\[
F_t=z_q(a)-z_q(b),
\]

其中 `a` 是位置 `q+1` 的 observed token，`b` 是同一 native logits 中排除 `a` 后的最高
runner。默认 `uncertain` 只按 native `|F_t|` 选择 target；也可使用 `evenly-spaced`、
`low-margin` 或 `all`。任何 source cut 和 label 都发生在这之后，runner 在所有 rerun 中固定。

这一定义测量 observed-token adoption，不把 observed token 预设为事实正确。

### 11.2 Transport 与 functionality 是两个轴

同一个 layer-unrolled DAG 提供两种互斥 transport backend：

\[
\tau^A_{l,h,s\to v}=A_{l,h,v,s},
\]

\[
\tau^M_{l,h,s\to v}
=\left\|W^O_{l,h}\left(A_{l,h,v,s}V_{l,g(h),s}\right)\right\|_2.
\]

`--flow-signal` 只在 `tau^A` 与 `tau^M` 之间切换。无论选哪一种 transport，每个保留边都
另外计算相同的 target-specific 功能量：

\[
\phi_{l,h,s\to v}
=\left\langle
\nabla_{c_{l,h,v}}F_t,
A_{l,h,v,s}V_{l,g(h),s}
\right\rangle.
\]

因此 raw attention 不能绕过功能性检查，message norm 也不会被误写成 readout contribution。
`phi>0` 表示局部一阶作用支持 observed token；`phi<0` 单独保留，表示支持 runner 或压低
observed margin。所有最终结论仍以非线性 rerun 为准。

native gradient 使用逐层 reverse VJP：每次只保留一个 decoder layer 的 autograd graph，
避免同时保留整条深度计算图。这只是峰值显存的结构性削减，不承诺 24GB 可运行；长上下文下，
单层 eager attention 仍可能需要 \(O(S^2)\) 显存，必须以实际序列长度和运行配置测量。

### 11.3 Residual-aware path law

attention backend 把每层所有 head 的单位 attention mass 与同权 identity continuation 比较，
所以满覆盖时 residual share 为 `1/2`。message backend 不再使用固定 `1/2`，而是用 native
layer-input norm 与所有真实 message norm budget 共同归一化：

\[
P(e)=\frac{\tau^M_e}
{\lVert r_{l,v}\rVert_2+\sum_{h,s}\tau^M_{l,h,s\to v}},
\qquad
P(residual)=\frac{\lVert r_{l,v}\rVert_2}
{\lVert r_{l,v}\rVert_2+\sum_{h,s}\tau^M_{l,h,s\to v}}.
\]

coverage 剪枝的质量仍流入 unobserved sink。`C(u→t)` 和 `T(v|u,t)` 是这个显式 screening
law 下的路径量，不宣称 residual vector 本身是概率守恒流。

### 11.4 Native source root

source unit 仍为 QA passage、Summary sentence 或 Data2txt field。对候选 unit `u`，干预算子
在所有层、所有 receiver 上删除从 `u` 的 token positions 发出的 post-softmax Value messages；
Q/K 不被直接 mask，attention 也不重归一。但 source-to-self Value message 同样被删除，
所以该 position 的后续 state 与 Q/K 会在 cut world 中自然演化，并不是冻结的 native
selector。因此结果只精确命名为这一 Value-source cut 算子对 margin 的效应。

\[
N_u=F(native)-F(cut\ u),
\]

\[
S_u=F(keep\ only\ u)-F(cut\ all\ evidence).
\]

先保存 raw transport `C_transport(u→t)` 作对照，再把 `phi<=0` 的边送入 sink，
以正功能 support graph 上的 `C_support(u→t)` 排序候选。前
`root_screen_limit` 个接受上述两个真实 rerun；`min(N_u,S_u)` 必须超过
model dtype tolerance 才能成为 confirmed root。方向永远是提高
observed-vs-frozen-runner margin，不乘 baseline margin 或任意 pair-effect 的符号。

### 11.5 Corridor 与 carrier

选中 root 后，程序构造 `root-cut` computation world，保存每层 layer input、attention write、
MLP write，并在同一正功能 support graph 上重新计算 single-root-conditioned throughput。
`T(e|u,t)>0` 的 attention/message 边在包含隐式 residual continuation 边的 augmented
layer-unrolled graph 中各自位于一条 root-to-target path 上；单独列出 sparse message
edges 不保证连通。随后执行：

1. native corridor 换成 root-cut codes，测 necessity；
2. root-cut world 原位补入 native codes，测 conditional rescue；
3. 把进入 target 的 corridor edges 冻结为 root-cut codes，测 mediated rescue；
4. 在 native 和 root-cut 两个 base gate 下分别 delete-and-restore，验证精确干预算子；
5. 对高 throughput carrier 做 native/root-cut state patch，再删除其下游 Value messages并重置
   residual carrier，使用 difference-in-differences 判断 mediation。

root、corridor 与 carrier 的同向 necessity/rescue/mediated rescue 都必须超过 model
dtype tolerance。完整 terminal mediation 还要求 corridor 或 carrier 被 block 后的残余
rescue 绝对值不超过该 tolerance；大幅反向残差也不算完整 mediation。

单个 carrier 的通过结果只保留为 `carrier_any_confirmed` 诊断。用于 evaluation 的
`carrier_value_mediated` 与 `full_chain_confirmed` 还必须同时满足完整 corridor confirmation，
不能把局部 carrier patch/block 结果报告成完整 root-to-target 链。

root-cut 的 broad source mask 在每次 suffix rerun 中持续存在；不能只从它的 layer-0 hidden
重新运行而忘记后续层的 source cut。测试要求两个 base world 都在 dtype tolerance 内恢复。

### 11.6 保存与标签解封

每个 target 的全量 pre-`W_O` codes 只在内存中用于精确干预。默认 artifact 保存完整
root/carrier/stage 台账、row-level transport/aggregation、node throughput，以及 top-2048
positive-functional corridor scalar edges；这个 top-K 只是截断导出，本身不宣称连通。不保存巨大的
`[E,D_h]` payload。原先 q=60 约
155MB 的单 target 文件因此不再线性复制到 cohort。

manifest 冻结数据哈希、sample/source、target、backend 和全部超参数，并支持逐 target 恢复。
capture artifact 永远写 `labels_used_for_capture=false`。只有 `subset-evaluate` 在
`analysis_complete=true` 后打开 labels，比较 hallucinated 与 clean target 的机制通过率。
这种富集子集不是总体 detector 评价，不能报告成全 token AUROC/AUPRC。
