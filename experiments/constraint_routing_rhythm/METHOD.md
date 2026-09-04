# Constraint Routing Rhythm：单一因果分数与功能路由节律

## 0. 文档状态

本文定义 `constraint_routing_rhythm` 的方法与实验边界。baseline、主干预、U/D
诊断、逐样本 artifact、可视化和后验评价已经实现；真实数据结果、完整对照和跨模型
复现尚未运行。合成测试通过只说明代码符合这里的 estimand，不等于机制成立。

方法只保留一条主线：

1. 对冻结模型做一次 baseline teacher-forced 前向，以精确
   \(A\lVert W_OV\rVert\) 构图，并把观察收敛为两个描述量：窗口化
   `FunctionalReach` 与严格层序的 evidence-conditioned `RelayCapacity`；
2. 再做一次 **全 evidence-source Value-message cut**，以固定
   target-versus-runner margin 的有符号变化定义唯一检测分数
   `ConstraintDeficit`；
3. 只在读取标签前固定的小子集上做 layer-ordered U/D/UD 四干预，检验候选
   response carrier，U/D/UD 不进入检测分数。

本方法不包含 GNN、learned surrogate、mask frontier、minimum circuit、ICG、
多特征分类器或手工加权 detector。后续若研究这些对象，必须建立新的实验身份，
不能反向写成本方法的一部分。

---

## 1. 研究问题与唯一主张

本文检验一个窄且可证伪的问题：

> 当模型预测已经生成的 response token 时，删除所有由声明 evidence tokens
> 发出的 Value messages，会怎样改变该 token 相对固定竞争 token 的 logit
> margin；幻觉是否表现为这种外部约束作用的缺失或反向。

若结果通过全部对照，可以声称：在指定模型、数据、teacher-forced prefix 与
post-softmax no-renormalization 干预下，hallucinated token 对 evidence-source
Value transport 的有符号因果敏感性较弱或方向不同；baseline 功能图显示这种
差异与特定的 uptake—carrier—delivery 节律共同出现。

最值得检验的机制结论不是“高路由强度就是高输出贡献”，而是两者是否发生系统性
解耦：图上的 route availability 可以很强，但 signed output control 仍然很弱、
正交或反向。两个 rhythm 量与主分数必须分别报告，不能拼成复合 detector。

不能声称：

- attention 或 \(A\lVert W_OV\rVert\) 是完整计算图或逐边因果效应；
- evidence cut 删除了模型中的事实知识、Q/K selection 或所有 evidence 表征；
- `ConstraintDeficit` 是 hallucination 的必要或充分条件；
- 一个 response carrier 在语义上“保存了证据”，除非对应 U/D 诊断通过；
- teacher-forced 结果等于自由生成轨迹中的因果效应；
- 当前工作发现了 minimum sufficient circuit 或全局信息流图。

---

## 2. Teacher-forcing 与因果坐标

设拼接序列为

\[
x_{0:n-1}=(x^{prompt},x^{response}),
\]

第一个 response token 的物理位置为 \(P\)。位置 \(p_t=P+t\) 的 response token

\[
y_t=x_{p_t}
\]

由 predictor position

\[
\boxed{q_t=p_t-1}
\]

的 logits 预测。每个事件必须同时保存：

```text
query_position      = q_t
prediction_position = p_t
target_token_id     = x[p_t]
```

所有进入该 predictor 的 source 满足 \(s\le q_t\)。\(y_t\) 自身的 embedding
不得进入预测它的图。

需要特别区分“生成 carrier token 的状态”和“carrier 作为以后 source 的状态”。
response token \(x_c\) 由 \(c-1\) 预测；只有在它已经进入序列并形成位置 \(c\)
的 residual state 后，它才能在未来 \(j>c\) 的计算中充当 carrier。因此：

- `evidence -> carrier c` 的 uptake 发生在 query position `c`；
- `carrier c -> future query j` 的 delivery 满足 `P <= c < j`；
- uptake 不能被解释为生成 \(x_c\) 本身的原因。

主实验固定已经观察到的 response prefix，只估计

\[
p_M(\cdot\mid x_{\le q_t};\operatorname{do}(g)).
\]

这带来一个不可省略的反例：早先正确 token 可能确实由 evidence 生成，但在后续
事件的 teacher forcing 中，其离散 embedding 已被固定成新的 response source；
后续 `ConstraintDeficit≈0` 可能只是模型合法地读取了这个既成中介，而不是原先没用
evidence。因此本方法估计的是“给定已观察 prefix 的边际 Value-channel 依赖”，
不能恢复整段回答最初的自由生成因果历史。

如果 observer model 与生成这段文本的 generator model 不同，必须单独记录模型
身份；此时分数是 observer-conditioned 诊断，而不是 generator 的原生生成机制。

---

## 3. Baseline 的精确功能消息幅值

### 3.1 GQA 与逐 query-head \(W_OV\)

对第 \(l\) 层，记原生归一化 residual 为 \(\widetilde x_{l,s}\)。模型有
\(H_q\) 个 query heads 和 \(H_{kv}\) 个 KV heads。query head \(h\) 必须读取
模型实际使用的 KV head \(g(h)\)。标准连续 GQA repetition 下，

\[
g(h)=\left\lfloor\frac{h}{H_q/H_{kv}}\right\rfloor,
\]

但实现应从模型配置/模块取得真实映射，不能假定 head 已按此顺序排列，也不能在
计算前平均 query heads。

先定义 head-space value 与其真实 residual write：

\[
v_{l,h,s}
=W^{V,[g(h)]}_l\widetilde x_{l,s},
\]

\[
u_{l,h,s}
=W^{O,[h]}_l v_{l,h,s},
\]

其中 \(W^{O,[h]}\) 是 `o_proj` 与 query head `h` 对应的输入块。baseline 中从
source \(s\) 到 query \(j\) 的实际 Value message 为

\[
m^0_{l,h,s\to j}
=A^0_{l,h,j,s}u^0_{l,h,s}.
\]

因为 post-softmax attention 非负，其消息 norm 精确等于

\[
\boxed{
\kappa^0_{l,h,j,s}
=\lVert m^0_{l,h,s\to j}\rVert_2
=A^0_{l,h,j,s}\lVert W^{O,[h]}_l
W^{V,[g(h)]}_l\widetilde x^0_{l,s}\rVert_2
}.
\]

这就是本文所称的精确 \(A\lVert W_OV\rVert\)。它不是裸 attention，也不是只到
head space 的 \(A\lVert V\rVert\)，更不是忽略 GQA 的静态整层 \(W_OW_V\)。

若所支持架构含 projection bias，source-independent bias 必须作为单独 local
write 保存，不能复制到每个 source。首个正式实现可以把范围明确限制在对应投影
为 bias-free 的 Llama/Qwen 类模型。

### 3.2 闭合边界

逐消息实现必须在每个 layer/query position 验证

\[
\sum_h\sum_{s\le j}m^0_{l,h,s\to j}
=o^{0,attn}_{l,j}
\]

在预注册 dtype 容差内成立。\(\kappa\) 是消息幅值，不能闭合 signed vector；
vector closure 与 norm statistics 必须分开测试。

这张图只描述 baseline 中实现的 Value writes，并条件于原生 RMSNorm、Q/K、
softmax、residual 与 MLP scaffold。Q/K 如何选择 source、MLP 如何变换写入状态，
没有成为独立图边。因此它是 **functional value-message map**，不是完整 Transformer
计算图。

---

## 4. Functional routing rhythm

所有 rhythm map 均由一次 baseline 前向得到，只负责解释和提出 U/D 候选，不
进入 `ConstraintDeficit`。完整 \([layer,head,query,source]\) 矩阵只在当前样本
内流式存在；全数据只保存固定曲线、head summaries 与少量 top motifs。

### 4.1 行归一化与绝对质量

对每个 \((l,h,j)\)，定义

\[
Z_{l,h,j}=\sum_{s\le j}\kappa^0_{l,h,j,s},
\qquad
\pi^F_{l,h,j,s}
=\frac{\kappa^0_{l,h,j,s}}{Z_{l,h,j}}.
\]

\(\pi^F\) 只在 \(Z\) 越过数值 resolution 时定义。实现于当前层计算它，然后只
保留二维流式均值。设预注册 split 为 \(l_*\)，early/late 层带分别为

\[
\mathcal L_U=\{l:l<l_*\},\qquad
\mathcal L_D=\{l:l\ge l_*\}.
\]

对 \(B\in\{\mathcal L_U,\mathcal L_D\}\)，保存

\[
K^B_{j,s}=\frac1{|B|H_q}\sum_{l\in B,h}\kappa^0_{l,h,j,s},
\qquad
\Pi^B_{j,s}=\frac1{|B|H_q}\sum_{l\in B,h}\pi^F_{l,h,j,s}.
\]

`early_absolute_map` / `late_absolute_map` 是 \(K^U/K^D\)，`early_map` /
`late_map` 是 \(\Pi^U/\Pi^D\)。`absolute_map` 与 `all_map` 是对应的全层均值。
因此“absolute”指未做 source-row normalization 的精确消息幅值均值，不是跨层/head
总和。逐层/head 的 \(Z\) 不跨 layer 保存；`functional_mass` 保存全层平均的
\(\sum_s Z_{l,h,j}\)。只看 normalized pattern 会隐藏功能质量很小但形状尖锐的行。

### 4.2 Functional local heads 与局部回看

受 *Attention Illuminates LLM Reasoning* 的 local/global 分解启发，对当前样本的
每一层分别计算各 query head 的平均 functional backward span：

\[
d^F_{l,h}
=\frac1{|\mathcal R|}
\sum_{j\in\mathcal R}
\sum_{s\le j}\pi^F_{l,h,j,s}(j-s),
\]

其中 \(\mathcal R\) 是 response query positions。每层内部按预注册分位数将较小
\(d^F\) 的 heads 定义为 `functional-local`，较大者定义为 `functional-global`。
逐层分组避免 local/global 图被少数深度垄断，也允许单次流式前向；所有层仍进入
最终 map。它是 sample-conditioned 的描述性分组，不声称发现跨样本固定的 head
身份。head distance 只在当前层用于分组，随后立即释放。

对 local heads，固定窗口 \(W\) 后定义 token-level local reach：

\[
L^F_j=
\frac1{|\mathcal H_{loc}|}
\sum_{(l,h)\in\mathcal H_{loc}}
\sum_{s\le j}\pi^F_{l,h,j,s}
\frac{\min(j-s,W)}{W}.
\]

峰值表示原本局部工作的 heads 在该位置向较远上下文回看，是“chunk boundary /
preplan”候选，不是正确推理的证明。

时间轴上 `FunctionalReach[t]` 描述预测当前 token 的 query，`RelayCapacity[t]`
描述随后形成的 token 能否充当 carrier。因此同下标对应“先回看、再成为锚点”的
一步相位关系。正式结论必须在 source-disjoint held-out 数据上比较峰值重合/时滞
与 circular-shift null；单个样本的好看曲线不是节律证据。

### 4.3 Functional global map

对每层 functional-global heads 的 \(\pi^F\) 分别求均值，再对所有层求均值，得到
`global_map`。它显示一个已经存在的 token \(i\) 是否持续被较长回看 heads 读取：

\[
G^F_{j,i}=
\frac1{|\mathcal H_{glob}|}
\sum_{(l,h)\in\mathcal H_{glob}}
\pi^F_{l,h,j,i}.
\]

高 \(G^F_{j,i}\) 是 functional anchor 的描述性图，不直接用于 U/D 候选；U/D
必须使用与实际 split 对齐的 late-band map。它仍是 baseline association，不等于
删除该 token 后的效果。

### 4.4 Evidence uptake

数据提供 evidence mask \(E\subset\{0,\ldots,P-1\}\)。对 response carrier
position \(c\ge P\)，只使用 early layers 定义

\[
U_c
=\sum_{s\in E}K^U_{c,s},
\qquad
\bar U_c=\sum_{s\in E}\Pi^U_{c,s}.
\]

`evidence_uptake` 是 \(U_c\) 的绝对质量，`evidence_binding` 是 \(\bar U_c\) 的
相对份额。两者都来自将被 U gate 删除的 early layer band；高份额但极低绝对质量
不构成可靠 uptake。

### 4.5 Response-carrier delivery

对 prior response token \(c\) 和未来 query 集
\(J_c=\{j:c+H_{low}\le j\le c+H_{high}\}\)，只使用 late layers 定义

\[
D_c=\frac1{|J_c|}\sum_{j\in J_c}K^D_{j,c},
\qquad
\bar D_c=\frac1{|J_c|}\sum_{j\in J_c}\Pi^D_{j,c}.
\]

实现分别保存 `future_delivery` = \(D_c\) 和 `future_influence` = \(\bar D_c\)。
候选是显式的 evidence → carrier → future 两跳 bottleneck：

\[
R_c=\min(\bar U_c,\bar D_c),
\qquad
M_c=\min(U_c,D_c).
\]

其中 `relay_capacity` = \(R_c\) 是 normalized 两跳容量，`relay_mass` = \(M_c\)
是 absolute 数值质量门。实现用 `mass_floor` 排除仅由浮点噪声形成的 bottleneck；
它必须在读取标签前固定。一个候选 relay motif 必须同时满足：

- \(R_c\) 越过预注册 normalized 分位数；
- \(M_c\) 越过预注册、与标签无关的 absolute numerical floor；
- U endpoint 来自 \(K^U_{c,E}>0\)，D endpoint 来自 \(K^D_{J_c,c}>0\)。

因为所有 U 层都满足 \(l_U<l_*\)，所有 D 层都满足 \(l_D\ge l_*\)，endpoint
天然满足 \(l_U<l_D\)。carrier 只由 `RelayCapacity` 与 absolute floor 提出，
`FunctionalReach` 不参与筛选；因此两条曲线的相位对应必须作为独立观察，而不是
在候选规则中预先制造。两种 bottleneck 只用于 label-free route proposal，不进入
检测分数，也不升级为第二个 detector。

严格说，`RelayCapacity` 是把 evidence sources 与 future queries 分别收缩成集合
节点后的三节点 quotient graph 上的 path bottleneck；它不是原始 Transformer DAG
的 max-flow，也没有证明 U/D 两段承载相同语义内容。没有满足未来窗口的末尾 token
记为 `NaN/unobserved`，不能当成零容量。

---

## 5. 主干预：全 evidence-source Value cut

### 5.1 干预位置

对 baseline 相同 token 序列重新前向。在每一层、每个 query head 和每个合法
query position，先从当前干预状态原生计算 RMSNorm、Q/K、softmax 与 V，然后在
softmax 之后、Value aggregation 之前执行：

\[
A^{-E}_{l,h,j,s}V^{-E}_{l,g(h),s}
\longmapsto 0,
\qquad s\in E.
\]

非 evidence sources 保持 gate 为一。删除后 **不重新归一化 attention row**；
evidence keys 仍参与当前层的 softmax denominator，但其 Value transport 被删除。
当前层写入改变后，所有后续层的 residual、RMSNorm、Q/K、softmax、V 与 MLP 从
改变后的状态真实重算。

因此该干预估计的是：

> 在保留 selection competition 的条件下，删除全部 evidence-source Value
> transport 对固定输出 margin 的总效应。

它不是 input-token deletion，也不是把 baseline messages 离线相减。实现必须
验证全一 gate 精确复现原生 logits；任何缓存了 baseline A/V 而没有重算下游的
分支只能作为错误对照。

### 5.2 固定 target–runner margin

对预测事件 `t`，baseline logits 为 \(z_t^0\)。target 是已经观察到的 \(y_t\)，
runner 在 baseline 上固定为

\[
\hat y_t
=\arg\max_{v\ne y_t}z_t^0(v).
\]

定义 baseline margin

\[
\mu_t^0=z_t^0(y_t)-z_t^0(\hat y_t).
\]

同一次 baseline logits chunk 还保存两个只用于分层控制的量：

\[
\log p_t^0(y_t)=z_t^0(y_t)-\log\sum_v\exp z_t^0(v),
\qquad
H_t^0=-\sum_v p_t^0(v)\log p_t^0(v).
\]

实现以 FP32 计算 `baseline_target_logprob` 和 `baseline_entropy`，不增加模型前向；
它们不参与 runner、干预或主检测分数。

在 evidence-source cut 重前向后仍使用同一个 \((y_t,\hat y_t)\)：

\[
\mu_t^{-E}=z_t^{-E}(y_t)-z_t^{-E}(\hat y_t).
\]

runner 不能随干预重新选择，否则每个样本比较的不是同一 estimand。

### 5.3 唯一检测分数

evidence 对 target margin 的 signed support effect 为

\[
\Delta_t^E=\mu_t^0-\mu_t^{-E}.
\]

唯一主检测分数定义为

\[
\boxed{
\operatorname{ConstraintDeficit}_t
=-\Delta_t^E
=\mu_t^{-E}-\mu_t^0
}.
\]

方向在读取标签前固定为“越高越符合 constraint deficit”：

- `ConstraintDeficit < 0`：删除 evidence transport 降低 target margin，evidence
  在该干预下支持已生成 token；
- `ConstraintDeficit ≈ 0`：输出对 cut 不敏感，可能是未使用、冗余、饱和或抵消；
- `ConstraintDeficit > 0`：删除 evidence transport 提高 target margin，evidence
  在该干预下对已生成 token 构成净 veto。

不得取绝对值、按 test labels 翻方向、除以不稳定的小 margin、与 confidence 或
rhythm maps 相乘，或再训练分类器。token-level 数据直接评价该量；只有 sequence
label 时，唯一允许的主聚合是在预注册有效 response positions 上做等权平均：

\[
\operatorname{ConstraintDeficit}_{seq}
=\frac1{|\mathcal T|}
\sum_{t\in\mathcal T}\operatorname{ConstraintDeficit}_t.
\]

这只是同一分数的固定聚合，不是第二个 detector。

### 5.4 Validity 与解释限制

以下事件标为 invalid，不填零：

- evidence mask 为空或与 tokenization 无法对齐；
- `q_t != p_t - 1` 或出现 future-source leakage；
- 自定义 attention backend 的全一 gate 集成测试不能在容差内复现原生 logits；
- 任一分支出现非有限 logits 或 hook 未覆盖声明的层/head；
- observer 输入并不包含数据声明的 evidence span。

`ConstraintDeficit≈0` 不能区分“不使用 evidence”和“多个 evidence/parametric
routes 相互冗余”。正确答案也可能主要依靠参数记忆而得到高 deficit；错误答案也
可能强烈依赖错误或冲突 evidence。因而主量测量 constraint sensitivity，而不是
无条件 truthfulness。

---

## 6. Label-blind、layer-ordered U/D/UD 因果诊断

U/D 实验只验证 baseline rhythm 提出的 relay motif。子集在 evaluator 打开标签
前固定：按 task/model 均匀采样，并可按 label-free `ConstraintDeficit` 与 uptake
分位数覆盖不同机制区间；禁止按 correctness/hallucination 类别挑样本。

对一组 carrier \(c\)、未来 query \(j\) 和严格有序层带
\(l_U<l_D\)，定义：

- `U`：split 以前所有层和 query heads 中 `evidence source -> carrier c` 的
  Value messages；
- `D`：从 split 开始所有层和 query heads 中
  `carrier c -> future query j` 的 Value messages。

carrier 与 endpoint 必须由同一个 split 下的 \(K^U,\Pi^U,K^D,\Pi^D\) 提出；不得
先用全层聚合图选 carrier，再用任意 midpoint 人为制造层顺序。

除 U/D 外的计算全部保留。四个分支均使用 post-softmax zero/no-renorm，并从
最早受干预层开始真实重算：

| 分支 | U | D | 含义 |
|---|---:|---:|---|
| `11` | on | on | 原生 U/D 路段 |
| `01` | off | on | 删除 evidence uptake |
| `10` | on | off | 删除 carrier delivery |
| `00` | off | off | 两段同时删除 |

用 baseline 固定的 target 与 runner 得到 \(\mu_{11},\mu_{01},\mu_{10},\mu_{00}\)，
报告：

\[
U\mid D=1=\mu_{11}-\mu_{01},
\]

\[
D\mid U=1=\mu_{11}-\mu_{10},
\]

\[
\boxed{
UD=(\mu_{11}-\mu_{01})-(\mu_{10}-\mu_{00})
}.
\]

`UD` 是两个消息组在该 margin 上的 difference-in-differences。它只说明删除效应
非加性；非零 UD 不是 semantic mediation、信息守恒或“证据存进 token”的证明。
至少还需满足 U/D 条件效应越过数值 resolution、`l_U<l_D`，并优于
position/layer/mass-matched random carrier 对照，才可称为 causal relay evidence。

U、D、UD 原值只出现在机制诊断表和可视化中，不进入
`ConstraintDeficit`、AUROC、AP 或 sequence aggregation。

---

## 7. 标签防火墙与评价

数据流程分成两阶段：

1. `capture/intervene` 阶段只读取 token ids、response boundary、evidence mask、
   model identity 和 source id；生成 baseline rhythm、固定 runner、
   `ConstraintDeficit`、validity 以及预注册 U/D 子集；
2. `evaluate` 阶段确认上述 artifacts 已冻结后，才读取 hallucination/correctness
   labels。

QA、summarization 与 data-to-text 分开报告，不能用一个任务上的层/head 选择服务
另一个任务。主指标为 token-micro AUROC、sklearn AP 和 source-cluster bootstrap
置信区间；sequence labels 只使用固定等权平均。必须另报：

- 每个任务和模型的有效覆盖率；
- response absolute/relative position、response length、evidence span length；
- baseline target margin、target log-probability 与 entropy；
- observer/generator 是否一致；
- correct-but-evidence-insensitive 与 hallucinated-but-evidence-sensitive 反例。

这些变量只用于匹配或分层控制，不进入主分数。若只有加入它们的监督模型才有
效果，应将结论写成混淆变量解释，而不是 routing mechanism。

另行报告 `route_control_dissociation`：`RelayCapacity` 与 evidence support
（`-ConstraintDeficit`）的相关性、高 RelayCapacity 区间内弱控制事件比例，以及
该区间内原主分数的评价。这里不把两者相加、相乘或拟合成新 detector；它只检验
“路由存在但未转化为输出控制”这一中心假设。

---

## 8. 必要对照与停止条件

### 8.1 构图与干预对照

1. `attention_only`：用 A 重算 local/global、uptake、delivery，检验精确
   \(A\lVert W_OV\rVert\) 是否真正超越 A-only。
2. `A||V||`：去掉匹配的 \(W_O^{[h]}\)，直接对照 HAVE 式 value calibration。
3. `head_mean / GQA_shuffle`：破坏 query-head 身份或 KV 映射，检验逐头/GQA
   保真是否必要。
4. `offline_subtraction`：从 baseline message 离线相减而不重算下游，量化真正
   反事实重算的增量。
5. `renormalized_cut`：pre-softmax mask 或删后重归一化，界定主干预语义。
6. `matched_non_evidence_cut`：切除长度、位置和 baseline 功能质量匹配的
   non-evidence sources，排除“删任意 prompt source 都会变化”。
7. `direct_response_cut`：只在 response query rows（包括预测首个 response token
   的 `q=P-1`）切 evidence sources，区分直接读取与 prompt 内预先整合；它只作
   诊断，不能成为第二检测分数。若比较 prompt-only/response-only，必须做完整四格，
   不能把两个非线性干预之差直接叫作间接效应。
8. `carrier_rewire`：在 position/layer/mass 匹配下替换 carrier，检验 U/D 是否
   只是局部位置效应。
9. `confidence/position/length`：独立非机制基线，不与主分数融合。
10. `phase_shift_null`：在 source-disjoint held-out 数据上检验两条 rhythm 曲线的
    峰值时滞，并以 response 内 circular shift 形成零分布；否则只展示曲线，不声称
    存在稳定节律。
11. `fixed_head_calibration`：用 label-free calibration sources 固定逐层 local/global
    head 集并报告稳定性；当前 sample-conditioned 分组只用于可视化。
12. `carrier_state_rescue`：U cut 后把候选 carrier 的 baseline residual state patch
    回去，检验输出效应是否恢复；它与 matched carrier 一起，才能支持更强的 relay
    解释。
13. `evidence_polarity_swap`：构造支持/冲突的实体、数字或关系替换，要求 signed
    effect 随 evidence polarity 翻转；未通过前只能称 evidence-source Value
    sensitivity，不能称“验证约束已完成整合”。

### 8.2 停止条件

出现任一情况时缩小或停止对应 claim：

1. 全一 gate 无法复现原生 logits，或 evidence cut 没有覆盖所有声明的
   layer/head/query/source；停止全部因果结论。
2. Functional maps 相对 attention-only/\(A\lVert V\rVert\) 不能更好地提出经 U/D
   验证的 carrier，或 rhythm 在 head/layer 置乱后不变；停止“功能节律超越
   A-only”的主张，但仍可保留主 intervention score。
3. Evidence-source cut 与 matched non-evidence cut 无可区分的效应，或结果完全由
   evidence length、position、baseline margin 解释；停止 constraint-specific
   mechanism claim。
4. U/D/UD 在 label-blind 独立样本、任务或模型上不能复现；停止 response carrier
   的因果解释，不得把 baseline delivery map 当成替代证据。
5. `ConstraintDeficit` 的预注册方向不能跨任务/模型复现；报告负结果，不增加第二
   detector、重新选层/head 或按 test labels 翻方向。
6. FunctionalReach–RelayCapacity 的 held-out 相位关系不优于 circular shift，停止
   “preplan-and-anchor rhythm”命名，只保留两条独立描述曲线。
7. evidence polarity swap 不产生可复现的 signed reversal，停止“constraint
   integration”语义，只报告 evidence Value-channel sensitivity。

---

## 9. 单卡执行与 artifact 边界

一次 baseline 中，\(\lVert W_O^{[h]}V_s\rVert\) 对 query `j` 不变，可先按
`[layer,head,source]` 计算，再与 query block 的 A 相乘。实现应：

- 一次只处理一个样本；
- attention backend 一次保留当前 layer 的原生 dense attention；随后只按 query
  block 计算 functional mass；
- 在线累积 local/global、uptake、delivery 和少量 top motifs；
- 每层完成后立即释放 attention/message dense matrices；二维 route sums 在当前
  样本结束时移到 CPU；
- baseline 默认只缓存 layer 0；启用 U/D 时再缓存预注册 split 的 CPU checkpoint；
- baseline 与 evidence-cut 分支顺序运行，不沿 branch 维度扩 batch；
- U/D 四分支只在固定小子集顺序运行；
- 不为全数据持久化 dense `[layer,head,query,source]` attention 或 message tensor。

`--limit` 只能减少样本数量，不能降低单样本峰值显存。若原生单样本前向本身不
适配目标设备，应明确缩短序列或更换设备；不能静默裁层、量化或丢 head 后沿用
同一实验名。

当前实现每个样本只保存：

```text
sample_id, source_id, task_type, model_id
query_position, prediction_position, target_token_id, runner_token_id
baseline_margin, baseline_target_logprob, baseline_entropy
cut_margin, constraint_deficit, valid
functional_mass, functional_reach, future_influence, future_delivery
evidence_uptake, evidence_binding, relay_capacity, relay_mass, carrier_mask
direct-response and matched-non-evidence controls, when sampled
U-cut, D-cut, joint-cut and interaction diagnostics, when a carrier exists
```

不保存文件身份摘要链、重复 schema wrapper 或几十个衍生特征。

---

## 10. 与直接近邻工作的边界

### Attention Illuminates LLM Reasoning

[*Attention Illuminates LLM Reasoning: The Preplan-and-Anchor Rhythm Enables
Fine-Grained Policy Optimization*](https://arxiv.org/html/2510.13554v2) 区分 local/
global attention heads，以 WAAD 描述局部回看，以 FAI 描述 token 获得的未来
attention，并据此提出 preplan-and-anchor rhythm 和 RL credit assignment。

本文明确受其 local/global 与 rhythm 视角启发，但不把 A 当作功能贡献：四张
解释图使用逐 query-head、GQA-correct 的
\(A\lVert W_O^{[h]}V^{g(h)}\rVert\)。更关键的区别是，本文唯一检测量来自全
evidence-source Value cut 的真实下游重算，而不是 local/global attention 指标。
因此可声称“从 A-only rhythm 推进到 output-projected functional maps，并以总
source cut 验证约束敏感性”，不能声称首次发现 local/global 或 anchor rhythm。

### FlowTracer

[*How Does Reasoning Flow? Tracing Attention-Induced Information Flow for
Targeted RL in LLMs*](https://arxiv.org/html/2606.10646v1) 将 token 构成 attention-
induced DAG，以聚合 A 作为非负容量，经 answer-targeted/flow-conserving 变换得到
多跳 backbone，用于 RL token credit。

FlowTracer 已覆盖“attention 图、多跳 global flow、anchor/hub 和 targeted RL”叙事。
本文不重新宣称这些贡献，也不做 rollout/flow conservation；functional rhythm
保留真实 Value/\(W_O\) 幅值，主因果对象是 evidence source group 的总 cut effect。

### HAVE

[*HAVE: Head-Adaptive Gating and ValuE Calibration for Hallucination Mitigation
in Large Language Models*](https://arxiv.org/html/2509.06596v1) 已指出 raw attention
不能代表 residual contribution，并使用 head-adaptive gating 与
`Attention × ||V||` value calibration 构造单步 decoding evidence。

因此“给 A 乘 value norm”不是本文的新颖点。本文必须使用匹配的 query-head
\(W_O\) block 后的精确 norm、真实 GQA 映射，并以 `A||V||` 作直接 ablation。
HAVE 的目标是单前向 mitigation/fusion；本文不修改 decoding distribution，
只以一次 baseline 解释节律，以第二次真实 source cut 定义 signed causal score。

### SinkProbe

[*Attention Sinks as Internal Signals for Hallucination Detection in Large
Language Models*](https://arxiv.org/html/2604.10697v2) 从 attention maps 计算未来
queries 指向 token 的 sink order statistics，再训练监督 logistic-regression
hallucination probe；论文也观察到 probe 偏好关联大 value norms 的 sinks，并明确
承认 sink score 本身是相关性信号。

本文不训练监督 feature probe，不把高 future influence 自动解释为 hallucination，
也不把 sink/anchor 作为第二 detector。`functional global` 只负责提出 carrier；
标签面对的唯一量是预先定向的 `ConstraintDeficit`。

### AVW_O 信息流与因果边界

[*Information Flow Routes: Automatically Interpreting Language Models at
Scale*](https://aclanthology.org/2024.emnlp-main.965/) 已使用 source/head 级
attention Value write 构造信息流路线。因此逐 source AVW_O 本身不能作为首创点。
本文的增量必须由以下实验事实支撑：精确 functional rhythm 相对 A-only 产生更好
的可验证 carrier 候选，且全 evidence-source post-softmax cut 的 signed effect
在独立任务/模型上与 constraint failure 稳定相关。

---

## 11. 最低可接受证据链

只有依次满足以下条件，方法故事才完整：

1. `q=p-1`、GQA、\(W_O\) head block、逐消息 closure 与全一 gate 测试通过；
2. baseline 一次前向能精确重算 functional local/global、uptake、delivery；
3. evidence cut 确实位于 post-softmax、pre-Value-sum，且下游真实重算；
4. `ConstraintDeficit` 在标签打开前按固定 target/runner 和方向冻结；
5. functional rhythm 超越 A-only 与 `A||V||`，并能提出优于 matched random 的
   U/D carriers；
6. U/D/UD 在 label-blind 子集提供可重复、严格 layer-ordered 的非加性证据；
7. 最后才在 source-disjoint 的 QA、summarization、data-to-text 和独立模型上评价
   唯一主分数，并如实报告无效覆盖与反例。

若只完成第 1–2 项，结论只是 functional attribution；若只完成一次 evidence
cut，结论只是指定干预下的 group sensitivity；若 U/D 不成立，就不能把 rhythm
解释为真实 relay。任何阶段都不以增加 GNN、surrogate、minimum-circuit search
或第二检测分数来补救负结果。
