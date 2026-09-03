# Shortcut-route 机制审计：逐头真实消息与来源—载体分离

## 1. 研究问题

本实验不再把“回答 token 关注了多少 prompt”直接当作幻觉信号。它研究的是一个更窄、也更可检验的问题：

> 当模型预测当前回答 token 时，真正写入残差流的消息来自什么信息根源，又由什么物理 token 承载；幻觉是否表现为外部依据仍然可见，但功能性支持逐渐从外部依据及其有根据的中继，转向回答自身产生的历史。

这里的 **shortcut route** 不是“response attention 很高”的同义词，而是以下三个现象的联合候选机制：

1. 外部根源的信息从原始 prompt 载体迁移到少数 response 载体，路由的层—头轨迹发生改变；
2. prompt 中能够提供消息的 source 集合异常收缩到少数锚点；
3. 在同样是 response carrier 的消息中，response-born 支持开始压过 evidence-grounded relay。

三个轴分别保存和检验，不用学习到的分类器、手工加权和或看过标签后的方向翻转把它们拼成一个分数。单个轴显著不等于幻觉；只有真实数据上的跨任务、位置匹配和构图对照能够决定该候选机制是否成立。

## 2. 为什么 response-heavy 不等于 hallucination

正确回答本来就需要大量 response-to-response 计算：语法延续、实体指代、已生成数字的复用、列表格式和长实体的连续拼写都会产生很强的 response route。更重要的是，response token 可以携带早先从 evidence 得到的信息。此时物理 source 位于 response，但其信息 root 仍是 evidence。

因此必须同时记录两件不同的事：

- **root**：消息中的状态最初来自 evidence、question/instruction、已有 response embedding，还是逐算子边界的数值闭合余项；
- **carrier**：这条消息当前从哪个物理 token 发出，它位于 evidence prompt、其他 prompt、严格 response history，还是当前位置的局部计算。

以下两条边的 carrier 都是 response history，但含义相反：

- `E root × response carrier`：回答 token 正在中继外部证据，是正常的 grounded relay；
- `R root × response carrier`：回答 token 在复用由回答自身注入的状态，是 response-born route。

只有把 root 和 carrier 分开，才能避免把正常的有根据续写误判为“自我依赖”。

## 3. Teacher-forcing 对齐

令 `P` 为缓存序列中第一个 response token 的位置。第 `t` 个回答 token为

\[
y_t=x_{P+t},
\]

它由 predictor 位置

\[
q_t=P-1+t
\]

的状态预测。该事件只允许使用

\[
s\le q_t
\]

的 source；`y_t` 的 embedding 不能进入它自己的图。artifact 必须同时保存 `query_position=q_t`、`prediction_position=q_t+1` 和 `target_token_id=x_{q_t+1}`，不能用含糊的 `target` 字段混合注意力 query 与预测标签。

严格 response history 使用 `P <= s < q_t`，predictor self `s=q_t` 不属于 history。前两个回答事件没有可与 predictor self 分开的严格 response-history carrier；依赖该 carrier 的量应标为 invalid，而不是填零后进入评价。

## 4. 逐头真实 AVWO 消息

模型冻结并以 eager attention、evaluation mode 做一次原生 teacher-forcing 前向。对第 `l` 层、第 `h` 个 query head、query `q` 和 source `s`，局部消息定义为

\[
m^{r}_{l,h,s\to q}
=
A_{l,h,q,s}
W^{O}_{l,h}
W^{V}_{l,g(h)}
\widetilde X^{r}_{l,s},
\]

其中：

- `A` 是该次原生前向的 post-softmax attention；
- `g(h)` 是 GQA 中 query head 到 KV head 的真实映射；
- `W^O_{l,h}` 是 `o_proj` 对应 query-head 输入块，而不是整层静态 `W_O W_V` 代理；
- `X^r` 是下面定义的 root ledger；
- `\widetilde X^r` 使用原生完整状态产生的 observed RMSNorm scale。

消息必须保留 `source/query/prediction/layer/head` 身份；四个 root 是同一物理
边的四列。构图前不得平均 head 或 layer。实现应在 FP32 中验证

\[
\sum_r\sum_h\left(
\sum_{s<q}m^{r}_{l,h,s\to q}+m^{r}_{l,h,q\to q}
\right)
=u^{attn}_{l,q},
\]

其中右侧是该层实际写入残差流的 attention output；BF16 fused kernel 的舍入
差异单独记录为 closure error。正式跨 token route table 只保存 `s<q` 的
strict first-arrival 项；`q->q` 的 predictor-self 项由同位置 suffix recurrence
承接。因此 `route table + self suffix` 闭合原生 attention write，而不是要求
strict route table 单独闭合整个 write。

每条物理边只保存一次目标无关的容量：

\[
\kappa_{l,h,s\to q}
=
\left\lVert\sum_r m^{r}_{l,h,s\to q}\right\rVert_2,
\]

而四个 root 的 signed functional contribution 是同一物理边的四列。这样
coverage 分母不会因 root 数量被重复计算；逐 root norm 只可作为审计量。裸
attention `A` 只作为控制，不替代真实消息。

## 5. Observed-gate 的 E/Q/R/N root ledger

root 顺序固定为

\[
\mathcal R=(E,Q,R,N).
\]

- `E`：passage、document 或 structured record 的外部 evidence；
- `Q`：question、instruction、system text、模板和其余 prompt；
- `R`：teacher-forced prefix 中已经可见的 response-token embedding；
- `N`：逐算子边界的浮点、融合 kernel 与数值闭合余项。

原生 RMS、attention 和 SwiGLU 的 observed-gate 分配已经进入 `E/Q/R`；
不得再把一般的 nonlinear state 或 MLP write 整体塞进 `N`。`N` 不是“参数
知识”，也不是幻觉标签。

### 5.1 初始化

输入 embedding `e_s` 按 token 位置互斥初始化：

\[
X^E_{0,s}=e_s\mathbf 1[s<P\land s\in evidence],
\]

\[
X^Q_{0,s}=e_s\mathbf 1[s<P\land s\notin evidence],
\]

\[
X^R_{0,s}=e_s\mathbf 1[s\ge P],
\qquad X^N_{0,s}=0.
\]

在每个层边界都要求

\[
x_{l,s}=\sum_{r\in\mathcal R}X^r_{l,s}.
\]

### 5.2 RMSNorm 与 attention

RMSNorm 的 scale 只由完整原生状态计算一次，并对四个 root 使用同一个对角算子：

\[
\widetilde X^r_{l,s}
=
\gamma_l\odot
\frac{X^r_{l,s}}
{\sqrt{\operatorname{mean}(x_{l,s}^2)+\epsilon}}.
\]

随后使用原生 `A`、动态 `V`、GQA 映射和匹配的 `W_O` head block 形成上一节的消息。Q/K 和 softmax 产生 `A` 的过程不再被伪装成一个可加 root；`A` 是明确声明的 observed gate。

### 5.3 SwiGLU MLP

MLP 也只使用原生前向观察到的 gate。令归一化完整输入为 `u=\sum_r u_r`，

\[
a=W_{gate}u,\qquad b=W_{up}u,
\]

并逐坐标定义稳定的 observed gate

\[
g=\operatorname{SiLU}(a)/a,
\]

在 `a=0` 处取连续极限。令 `a_r=W_{gate}u_r`、`b_r=W_{up}u_r`，使用对称分配

\[
F^r
=\frac12W_{down}
\left[
(g\odot a_r)\odot b
+
(g\odot a)\odot b_r
\right].
\]

于是对 bias-free Llama 有

\[
\sum_r F^r=F^{native}.
\]

这个等式指同一实数算术下 `a=\sum_r a_r`、`b=\sum_r b_r` 时的代数恒等式。
实际实现使用由 `--dtype` 指定的原生前向（默认 BF16）捕获 `a,b`，root 投影与
读出使用 FP32；两条数值路径之间的 kernel 舍入差异在算子边界先记录，再进入
`N` 闭合，超过预注册容差时 validity 失败。辅助 root 算子可以沿 token 和
intermediate 坐标分块求同一组和；这只改变 FP32 reduction order，不删除
token/root/head，也不分块或替代原生 full-sequence 前向。分块式与密集式必须
在声明的闭合容差内一致。

这只是固定原生 gate 后的精确加性记账；删除某个 root 会改变 RMS、Q/K、softmax 和 SwiGLU gate，因此它不是 counterfactual causal effect。每层最后用原生 layer output 对 `N` 做一次数值闭合，并保存闭合误差。闭合误差超过预先固定容差时，该行无效，不能把错误吸收到 `N` 后继续打分。

## 6. Root × carrier cells

每条严格非 self 的 attention 消息都同时落入一个 root 和一个物理 carrier。完整 root×carrier 表保留为 `[token, layer, head, root, carrier]`，carrier 至少区分：

- `evidence_prompt`；
- `other_prompt`；
- `response_history`。

稀疏 `tail` 只是没有显式保存 endpoint 的压缩统计，不是第四种 carrier，
更不能改写成 `N` root。

predictor 的原始状态、residual carry、self-attention 后缀和同位置 MLP 不重复创建跨 token 边，而作为 `injection` 项进入最终读出。为便于解释，完整表再投影到下列命名 cell；它们不是替代 root×carrier 表的穷尽类别：

| Cell | 定义 | 解释 |
|---|---|---|
| `D` | `E root × original evidence carrier` | direct evidence |
| `P_E` | `E root × other-prompt carrier` | evidence 在 prompt 内的中继 |
| `G` | `E root × response-history carrier` | grounded response relay |
| `B` | `R root × response-history carrier` | response-born history |
| `Q` | `Q root × any non-self carrier` | question/instruction constraint；保留 direct/relay 子载体 |
| `I` | predictor 初始状态经 residual/self/MLP suffix 到读出的项 | local injection，不等于参数知识 |
| `N` | `N root` 与数值 closure | numeric remainder；tail 另存 |

完整 root×carrier 表是数据本体，`D/P_E/G/B/Q/I/N` 只是固定的可解释汇总。不能只保存这些标量后丢弃 layer、head、source 或 carrier subtype。

每个事件的固定 resolution 至少包含逐项数值 total variation

\[
\begin{aligned}
\delta_t={}&|\iota^{N,initial}_t|
+\sum_{l,h,s<q_t}|\phi^N_{t,l,h,s}|\\
&+\sum_l\left(
\sum_h|\phi^{N,selfV}_{t,l,h}|+|\phi^{N,postA}_{t,l}|
+|\phi^{N,layer}_{t,l}|
\right)
+|\phi^{N,finalRMS}_t|.
\end{aligned}
\]

其中 `selfV` 是逐 query head 保存的 predictor-self 原生 closed-V 与 suffix
FP32 V 算子之差，
`postA` 是 post-attention 边界闭合，`layer` 是对称 MLP 后的 layer-output
边界闭合，`finalRMS` 是最终归一化输出闭合。它们分别在对应 suffix adjoint
上投影后再取绝对值。resolution 还包含独立的算子闭合误差上界。禁止先把
这些 signed 项并入 `N` 净值再取一次绝对值；那会用跨边界抵消低估数值
不确定性。后文所有比值只有在分母严格大于 `δ_t` 且 operator-valid 时才定义。

## 7. Signed support 与 veto

令原生预测的正确 teacher-forced token 为 `y_t`，原生 logits 中最强的其他 token 为 `\hat y_t`。最终 readout covector 是

\[
d_t=W_U[y_t]-W_U[\hat y_t].
\]

在 observed-gate 后缀上将该 covector 反向拉回到每层 query residual write，得到 `\beta_{t,l}`。self-attention、residual、final RMSNorm 和 MLP 均使用同一原生 observed gate；跨 token 的首次到达消息 `s<q_t` 保持为显式 atom。每个 atom 的 signed contribution 为

\[
\phi^{r}_{t,l,h,s}
=
\langle\beta_{t,l},m^{r}_{l,h,s\to q_t}\rangle.
\]

对任何 cell `c`，先逐 atom 分开正负项，再求和：

\[
S^c_{t,l,h}=\sum_{a\in c}\max(\phi_a,0),
\]

\[
V^c_{t,l,h}=\sum_{a\in c}\max(-\phi_a,0).
\]

`S` 是对当前 target-versus-runner margin 的 observed-gate support，`V` 是 veto。禁止先把消息相加再取绝对值或 ReLU，否则 head/source 间的抵消会被隐藏。每个 layer/head map 都保存 `capacity/support/veto`，且满足声明的 margin closure；三个机制轴不得只读取已经跨 head 平均的量。

这里的 support/veto 仍是 observed-gate attribution。它说明原生路由下一个消息与最终 margin 的关系，不说明删除这条消息后模型会发生相同大小的行为变化。

`y_t` 是待审计生成记录中已经出现的 token，不是 hallucination/correctness
标签。因此该量是 response-conditioned 的事后机制诊断，而不是生成 `y_t`
之前可用的风险预测器。若要声称 prefix-causal 预警，必须另行固定一个不读取
`y_t` 或未来 token 的 readout；不能把本实验的 target margin 伪装成预生成
特征。

## 8. 三个核心轴

### 8.1 Carrier drift

该轴只回答物理消息由 prompt 还是 response-history token 承载。先在每条
物理边内部合并三个科学 root，不能跨 source 或 head 合并：

\[
\psi_{t,l,h,s}=\phi^E_{t,l,h,s}+\phi^Q_{t,l,h,s}+\phi^R_{t,l,h,s}.
\]

对 `σ∈{support,veto}` 分别令
`J_support(z)=max(z,0)`、`J_veto(z)=max(-z,0)`，再统计：

\[
M^{\sigma,prompt}_{t,l,h}
=\sum_{s<P}J_\sigma(\psi_{t,l,h,s}),
\]

\[
M^{\sigma,response}_{t,l,h}
=\sum_{P\le s<q_t}J_\sigma(\psi_{t,l,h,s}).
\]

逐层逐头的 carrier map 为

\[
C^\sigma_{t,l,h}
=
\frac{M^{\sigma,response}_{t,l,h}}
{M^{\sigma,prompt}_{t,l,h}+M^{\sigma,response}_{t,l,h}}.
\]

分母未越过数值 resolution 时返回 `NaN + mask=False`。保存的正式对象是
support/veto 两张 `[token,layer,head]` map，而不是一个 early/late 手工特征。
该轴包含 response-born 物理消息；随后由 takeover 轴判定 response carrier
究竟是正常的 `E/Q` 中继还是 `R` root 接管。

若需要一个 token 级表格摘要，层坐标记为 `\tau_l=2l/(L-1)-1`，分别计算
prompt-carried 与 response-carried 消息的质量加权层质心：

\[
\bar\tau^{\sigma,k}_t
=
\frac{\sum_{l,h}\tau_l M^{\sigma,k}_{t,l,h}}
{\sum_{l,h}M^{\sigma,k}_{t,l,h}},
\]

\[
Drift^\sigma_t
=\frac{\bar\tau^{\sigma,response}_t-\bar\tau^{\sigma,prompt}_t}{2}.
\]

`Drift_t` 仅是展示和 AUROC 用的预声明摘要；论文中的机制证据必须同时
展示 layer/head map。正值表示 response-carried 消息比 prompt-carried 消息
更晚到达，并不自动表示幻觉。

### 8.2 Prompt source dispersion

该轴回答“功能性 prompt 消息是由广泛 source 共同提供，还是缩成少数
shortcut anchors”。它使用上一节逐物理边得到的 `J_support(ψ)` 与
`J_veto(ψ)`，而不是裸 attention，也不混入 response source。对每个
`(t,l,h,σ)`：

\[
p^\sigma_s
=\frac{J_\sigma(\psi_{t,l,h,s})}
{\sum_{u<P}J_\sigma(\psi_{t,l,h,u})}.
\]

归一化熵为

\[
H^\sigma_{t,l,h}
=
-\frac{\sum_{s<P}p^\sigma_s\log p^\sigma_s}{\log n_{eligible}}.
\]

无 prompt 功能质量或少于两个 eligible endpoint 时标为 invalid。该 map 始终
保留逐头值；不得先平均 head 再计算熵。token 摘要只在逐头熵已经计算后，
按对应 row 的 support/veto 质量加权。

稀疏 artifact 对每个 tail 保存逐 source 的
`sum(x log x)`，而不只保存 tail 总量。因此 retained atoms 加 tail moments
能够精确重算这个熵；tail 不需要也不允许被均匀摊回虚构 endpoint。shortcut
假设预期 hallucinated token 的 prompt dispersion 更低，即更依赖少数 prompt
anchors；如果真实数据稳定显示相反方向，应记录为机制修正，不能在 test
label 上翻转分数继续声称原假设成立。

### 8.3 Response-born takeover

该轴只在相同的 `response_history` carrier 内比较 root，直接排除“response-heavy 就是错误”的混淆。对每个 `(t,l,h)`：

\[
T^S_{t,l,h}
=
\frac{S^B_{t,l,h}}
{S^B_{t,l,h}+S^{G}_{t,l,h}+S^{Q_{response}}_{t,l,h}},
\]

\[
T^V_{t,l,h}
=
\frac{V^B_{t,l,h}}
{V^B_{t,l,h}+V^{G}_{t,l,h}+V^{Q_{response}}_{t,l,h}}.
\]

无严格 response-history message 或分母未越过数值 resolution 时该 cell 无效。
`T^S` 高表示 response carrier 对当前目标的正向支持主要来自 response
embedding root，而不是它中继的 evidence 或 question root；这才是
response-born takeover。`G` 高而 `B` 低是正常 grounded relay，不能判作
shortcut。

token 级摘要只做有效 layer/head 的分母质量加权平均。`I` 和 `N` 始终单独报告，不塞入 `B` 分子；否则局部 predictor 状态、MLP dynamics 或数值 remainder 会被误称为“回答自生知识”。

## 9. 稀疏选择与 tail

完整 dense 聚合在线计算，稀疏边只用于保存端点、可视化和 endpoint 对照。选择规则在读取标签前固定：

1. 对每个 `(token,layer,head)` 的物理边独立处理，绝不跨 head 做全局 top-k，也不为四个 root 重复同一条边；
2. 非 self 物理边按真实消息容量 `\kappa` 降序，平局按 source index；
3. 保留达到固定累计容量 `rho=0.95` 的最小前缀，且最多 `K_max=64`；
4. `I`、cell 总量和所有 closure 量不受 top-k 影响；
5. 对遗漏部分按 carrier×root 保存 `tail_count`、`tail_capacity`、`tail_support`、`tail_veto`，并保存物理 support/veto 的 `sum(x log x)`。

tail 是“端点未显式保存”，不是观测到的零。不得把 tail 平均分配给虚构
source，不得复制一个显式 source 来承载 tail，也不得在 endpoint rewire 中把
tail 当作真实边。`D/P_E/G/B/Q/I/N` 的 dense 总量在压缩前计算，因此不因
top-k 改变；prompt entropy 由 retained atoms 与 tail `sum(x log x)` 精确恢复。
这里的“精确”指相同的实数定义；不同 FP32 reduction order 的一致性使用按
root total variation 和 reduction depth 声明的累计舍入预算，不要求 bitwise 相等。

## 10. 输出与最小评分边界

每个样本至少保存：

```text
source_token_id
evidence_mask, top_k, cover_mass
query_position, prediction_position, target_token_id
competitor_token_id, target_logprob
selected physical row: source, layer, head, carrier
selected physical row: attention, value_energy, physical_message_norm
selected physical row: root_phi[E,Q,R,N]
root_carrier_support/veto [token, carrier, root, support/veto]
carrier_drift_map       [token, layer, head, support/veto]
prompt_dispersion_map   [token, layer, head, support/veto]
takeover_map            [token, layer, head, support/veto]
tail_count/attention/value_energy/message_norm sum and max by row and carrier
tail root support/veto and physical sum(x log x) by row and carrier
numeric_self_v_phi, numeric_post_attention_phi
numeric_layer_phi, numeric_final_phi, numeric_total_variation
operator_error, root_closure_error, native_margin
```

一条物理边只占一行，`root_phi` 不展开成四条重复边，也不存在 root-specific
capacity。`root_carrier_support/veto` 是 token 级 dense 汇总；完整的
`[token,layer,head,root,carrier]` 信息由 retained physical rows 与逐 row tail
moments 共同构成，不虚构 tail endpoint。

主实验只评价三个预声明 token 摘要及其完整 map，不从这些 tensor 再派生几十个人工特征，不训练 GNN、autoencoder、probe 或监督 combiner。位置、相对位置、回答长度和 token surprisal 是独立控制，不能进入核心轴后再把增益解释为路由机制。

标签打开前冻结的三个主关联量只取 support 通道：

| 主关联量 | 固定评价方向 |
|---|---|
| `carrier_drift_support` | 越高越符合 shortcut 假设 |
| `prompt_source_dispersion_support` | 越低越符合 shortcut 假设；AUROC 输入固定为负熵 |
| `response_born_takeover_support` | 越高越符合 shortcut 假设 |

veto 通道仍逐边、逐层、逐头保存，用于守恒、抵消和机制组差异审计，但没有
预注册的单调 hallucination 方向，不得在 test labels 上选择正负号后升级为第四
至第六个 detector。若后续研究要把 veto 设为主终点，必须在新的独立数据评价
前另行预注册。

## 11. 决定性合成测试

除第 11 项是开放分块模式前的门禁外，代码实现前必须先固定以下测试；全部通过
只说明实现符合定义，不代表真实数据有效。

1. **逐边 AVWO oracle**：微型 GQA Llama 上显式计算每个 source/head 的 `A W_O^h W_V^{g(h)} RMS(x_s)`，逐项匹配 capture。
2. **attention write closure**：所有 root/source/head 消息向量之和重构每层原生 attention write；交换两层 `W_O` 后测试必须失败。
3. **GQA identity**：每个 query head 只读取声明的 KV head，且不通过 head mean 获得相同结果。
4. **q→q+1 与无泄漏**：首个 response token 只读取到 `P-1`；改变 `y_t` 的 embedding 不得改变其自身已冻结 predictor trace。
5. **observed-gate closure**：embedding、RMSNorm、attention、对称 SwiGLU、layer output、final RMS 和 target margin 分别闭合；故意清零一类 write 时 validity 必须失败。
6. **root×carrier relay**：构造一个 response source，其中同时含 `E` 与 `R` root；相同物理 endpoint 必须分别进入 `G` 与 `B`。
7. **support/veto cancellation**：两个 head 写入相反方向时，signed sum 可以接近零，但 support 和 veto 都必须保持非零并精确守恒。
8. **三轴单调性**：只把 E/Q root 从 prompt carrier 移到 response carrier时只改变 drift；只增加 B 时提高 takeover；把 prompt 容量集中到一个 source 时降低 dispersion。
9. **逐头稀疏与 tail**：安静 head 不能被强 head 的全局 top-k 吞掉；显式量加 tail 等于 dense 量，tail `sum(x log x)` 精确恢复 dense entropy。
10. **因果端点**：所有显式边满足 `s<q`，self 只进入 `I`；未来 token 和当前预测 token不得出现为 source。
11. **chunk invariance**：仅在公开 KV-cache/chunk 模式前启用；完整前向与分块前向必须在声明的数值容差内产生相同端点、消息、root cells 和三个轴。当前 v1 只有 full-sequence observer，不声称或伪装支持 chunk。
12. **标签隔离**：capture、root ledger、稀疏选择、轴方向和 validity mask 均在
    数据接口不暴露标签时完成。canonical cache 的 attention 与 label 物理分离；
    对历史单体 formal `.pt`，共享 loader 会反序列化整份 payload，但
    `retain_embedded_labels=False` 必须丢弃且不向构图代码暴露其中标签。

## 12. 真实数据判据

QA、Summary、Data2txt 分别报告，不能以 QA 的结果替代任务泛化。三个固定
support 主关联量使用 token-micro AUROC、sklearn AP、source-cluster bootstrap
CI，并报告 hallucinated/correct 的位置匹配差异；veto 只报告带方向的原始组
差异和 CI。所有方向、阈值、有效 token mask 和摘要公式在打开标签前冻结。

该候选机制只有同时满足以下条件才保留：

1. 在 response-heavy token 内，`B` 相对 `G` 的 takeover 能区分 unsupported 与 grounded token；单纯 response mass 不能取得同等解释力；
2. hallucinated token 的外部-root carrier map 出现可重复的层—头迁移，而不是只在跨头平均或一个被挑选的 head 上出现；
3. prompt dispersion 的变化在独立 task/model 上方向一致，并在位置、长度、置信度匹配后仍存在；
4. 三个轴至少各有一个结构对照破坏其信号，且 native graph 相对对照的 paired source-bootstrap CI 排除零；
5. `N` 比例和数值 closure error 不解释主要差异；高 tail 行不主导结果；
6. 结果不能通过看过 test labels 后选层、选头、翻方向或改变 `rho/K_max` 才成立；
7. 正确的窄路由、格式续写和 evidence relay 不被系统性赋予高 shortcut 风险。

若这些判据失败，应将结论写成“该机制在当前数据/模型上未得到支持”，保留可复现结果，而不是增加新的特征或监督模块挽救分数。

## 13. 构图对照

本节是机制保留前的下一阶段验证计划，尚未进入当前 v1 一键入口。实现前必须为
每个对照明确它能定义的轴和 estimand，不能为了得到成对分数而给未定义量填零。
可比较的对照使用相同 token、共同有效集、相同层/头预算和相同 seed；破坏量定义
本身的 negative control 只报告它实际检验的契约：

- `attention_only`：用 `A` 替代 `AVW_O`，检验 value 与 `W_O` 是否真正提供信息；
  它没有 E/Q/R root，不能伪造 takeover 对照；
- `physical_response_ratio`：只看 prompt/response carrier，不看 root，检验 root×carrier 分离是否必要；
- `endpoint_rewire`：在 `(layer,head,carrier_role,lag_bin)` 内做保持因果性和度数的双交换；必须改变真实 source 后重新读取该 source 的 root state、`V` 和 `AVW_O`，不能只交换已算好的标量；
- `weight_shuffle`：保留 endpoint/value 集合，在 row 内打乱 attention weight，检验准确配对；
- `head_or_layer_shuffle`：保留边的边际量但破坏 head identity 或层序；若 token
  摘要在代数上对某种置换不变，则只比较受影响的 layer/head map，不报告虚假的
  标量差异；
- `no_message`：保留位置和图规模但移除邻居消息；三个比值轴因此未定义，它只作
  closure 与 fail-closed negative control，不计算伪造的 AUROC；
- `confidence/position/length`：独立的非图基线。

rewire 不得虚假声称同时保持真实 message norm；source 改变后 `V` 改变，message capacity 理应重新计算。对照只声明它实际保持的 row 数、head、role、lag、attention multiset 和因果约束，并报告重算后的质量漂移。

## 14. 旧方法的地位

以下内容只作为锁定的历史控制，绝不成为核心图的输入、额外特征或三个轴的一部分：

- `F/noE/noH/noEH` 四分支 attention-write deletion；
- evidence-adoption/autonomous-history 两个 finite-difference register；
- cross-layer register Gram、leading eigenvalue 与 `provenance_takeover`；
- `evidence_bypass`、`symmetric_route_capture`、`unsupported_history_takeover`；
- 旧 prompt collapse 的 effective sources/rank/anchor/log-volume；
- SFAC 的 signed first-arrival conflict 标量。

历史控制可以在独立报告中用原来冻结的公式重算，用于说明新方法是否超越已有结果；不能与 `D/G/B/Q/I/N` 拼接、加权、校准或送入第二个 detector。新方法可以共享“observed-gate 加性闭合”和 signed atom 这一基础数学，但不使用 SFAC 的 conflict 公式，也不以旧分支定义 root。

这些历史控制的执行文件保留在独立的兄弟目录
`experiments/attention_mechanism_audit/` 中。新方法不得导入或复用旧入口；复核
旧结果时应独立运行旧包，不能把旧分数接回当前主流程。

## 15. 可声称与不可声称的边界

若实现闭合且真实数据判据通过，可以声称：模型原生前向中存在一个逐头、逐层、root×carrier 分离的 shortcut-route 统计模式，并且该模式对 token-level hallucination 有可重复关联。

不能声称：

- attention 或 observed-gate attribution 是完整因果效应；
- `R`、`B`、`I` 或 `N` 等于参数知识、内部事实或模型“相信”的内容；
- response-heavy、低/高 dispersion 或任一单轴是幻觉的必要或充分条件；
- teacher-forced observer trace 等于生成模型原始采样时的隐藏状态，尤其在 observer 与 generator 不同时；
- 稀疏显式边构成完整图，或 unresolved tail 有已知 endpoint；
- token centroid 可以替代 layer/head map；
- 一个任务、一个模型或 smoke subset 的结果能够证明通用机制。

若需要因果必要性或充分性，必须另做真正重新前向的 remove/keep-only intervention，并明确其 gate 会重算；该实验属于后续验证，不得反向改变本方法的 observed-gate 定义。
