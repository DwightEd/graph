# P-Cut：把“异常检测”改写成“证据依赖检验”

## 1. 一句话

> **不要问这张 attention 图像不像异常；要问当前回答在切断 prompt 证据路径后，是否仍然可以靠自己已经生成的内容继续运转。**

暂定方法名：**P-Cut（Prompt-Provenance Cut）**。

可用的论文标题：

> **Can the Answer Run Without the Evidence? Unsupervised Hallucination Detection by Prompt-Provenance Cuts**

这不是对当前 HoloRoute 再加一个模块。HoloRoute 保留为图编码基线；P-Cut 改的是研究问题和最终分数的含义。

---

## 2. 为什么不再以“异常”作为起点

传统无监督检测默认：正常数据形成稳定分布，异常数据稀有、难预测或难重构。这个假设在很多工业图数据上合理，但在语言模型内部不一定成立。

一个无依据的实体或数字一旦被写进回答，后续 token 可以围绕它形成很顺畅的句子。此时：

- response-to-response attention 很稳定；
- 局部图可能很规则；
- 下一步路由可能很好预测；
- reconstruction error 甚至会下降。

因此，“稳定”可能表示正常，也可能表示模型已经稳定地进入错误续写。我们过去的结果正好符合这一点：RR 结构有信号，但 dynamic prediction、manifold-kNN 和一般图恢复没有可靠区分力。

真正与上下文幻觉相关的问题不是“是否稳定”，而是：

\[
\boxed{\text{这种稳定状态还需不需要 prompt 证据？}}
\]

---

## 3. 中心假设：证据旁路

### 3.1 Response-closure hypothesis

一个 response token 看历史 response 有两种完全不同的含义。

第一种是合法中继：

```text
prompt evidence -> earlier response token -> current token
```

虽然当前 token 没直接看 prompt，但它使用的历史 token 是从 prompt 继承来的。

第二种是回答内闭环：

```text
earlier response claim -> later response claim -> current token
```

这条路径主要由回答自身支撑，prompt 只剩很弱或不必要的作用。

P-Cut 的核心假设是：

> **上下文幻觉可以是一个稳定的 response-closed 状态。它不一定难恢复，却会表现为：切掉 prompt-rooted 路径影响很小，切掉 response-closed 路径影响很大。**

这比“prompt attention 少”更严格，也比“图重构误差大”更贴近当前实证。

### 3.2 为什么这个假设有依据

- 一跳 prompt provenance centroid/spread 是当前最强单结构筛查，说明 response source 的来源比直接 prompt 比例重要。
- RR 联合谱残差明显强于 RP residual，说明回答历史的组织方式携带主要信号。
- onset 中 prompt share 反而上升，说明直接 prompt mass 不能代表真正依赖。
- 来源重连和 recovery 失败，说明“真实/常见/易恢复”与正确性不是同一个概念。
- Lookback、RFS-Guard、StepFlow 和 CoDA 都从不同角度表明：信息流向哪里、是否保持桥接、是否被后续计算采用，比单个 attention 数值更重要。

---

## 4. 图的作用只有一个：追溯 response 路径从哪里来

P-Cut 不要求一个很复杂的新 GNN。图存在的理由非常具体：

> 当前 token 对一个 response source 的依赖是否 grounded，取决于这个 source 在更早层、更早 token 中从哪里取得信息。

单看当前 attention row 无法回答这个问题；必须沿有向因果图传播 provenance。

### 4.1 基础图

一条 prompt-response 样本构造一张图。token 是语义节点；每条 retained attention incidence 保留：

```text
source token
response target token
layer
head
attention weight
```

现有 HoloRoute 的 event node `(source,target,layer)` 也可以作为实现载体，但它不是论文创新。P-Cut 的定义对 token graph 或 event graph 都成立。

### 4.2 Sparse cache 的不确定性不能当零

设某行已知 retained off-diagonal mass、diagonal 和 unresolved mass。未保存边只知道低于阈值。因此 provenance 不是一个精确值，而是一个区间：

\[
\underline g_{t}^{\ell}\le g_t^{\ell}\le \overline g_t^{\ell}.
\]

- `lower` 假设 unresolved mass 都不来自 prompt；
- `upper` 假设 unresolved mass 都可能来自 prompt。

这样不会把 cache 截断错误地解释成“没有证据”。

---

## 5. Prompt provenance 如何传播

prompt token 的 provenance 固定为 1。response token 在进入第一层前设为 0。

对 response token `t`、layer `l`、head `h`，设：

- `A_{t,p}^{l,h}`：指向 prompt token 的 retained attention；
- `A_{t,j}^{l,h}`：指向历史 response token `j` 的 retained attention；
- `d_t^{l,h}`：self diagonal；
- `u_t^{l,h}`：unresolved mass。

下界：

\[
\underline g_{t,h}^{\ell+1}
=
\sum_{p\in P}A_{t,p}^{\ell,h}
+d_t^{\ell,h}\underline g_t^\ell
+\sum_{j<t}A_{t,j}^{\ell,h}\underline g_j^\ell.
\]

上界：

\[
\overline g_{t,h}^{\ell+1}
=
\sum_{p\in P}A_{t,p}^{\ell,h}
+d_t^{\ell,h}\overline g_t^\ell
+\sum_{j<t}A_{t,j}^{\ell,h}\overline g_j^\ell
+u_t^{\ell,h}.
\]

再对 heads 做预注册的聚合，并截断到 `[0,1]`。

直观上，它只是不断问：

```text
这个 response token 读到的质量中，至少有多少能追溯到 prompt？
最多有多少可能追溯到 prompt？
```

不需要先把 token 判为“grounded / ungrounded”，也不需要训练一个 provenance 分类器。

---

## 6. 每条 response edge 分成三部分

对一条 response-source edge `e=(r_j -> r_t,l,h)`，source `r_j` 的 provenance 区间为：

\[
[\underline g_j^\ell,\overline g_j^\ell].
\]

把已知 edge weight `A_e` 精确拆成：

\[
w_e^P=A_e\underline g_j^\ell,
\]

\[
w_e^R=A_e(1-\overline g_j^\ell),
\]

\[
w_e^Q=A_e(\overline g_j^\ell-\underline g_j^\ell).
\]

三者相加仍为 `A_e`。

- `P`：至少可以确认是 prompt-rooted 的质量；
- `R`：即使把 unresolved mass 全算给 prompt，也无法追溯到 prompt 的 response-closed 质量；
- `Q`：受 sparse censoring 影响，来源不确定的质量。

prompt edge 直接属于 `P`。这一步是整个方法中最关键的图操作。

---

## 7. 两次切割，而不是六个手工特征

### 7.1 Full view

完整图记为 `G^F`。用一个无标签图编码器得到 token 表征和一个预先定义的自监督目标：

\[
z_t^F=Enc(G^F,t),
\qquad
L_t^F=L(z_t^F,y_t^{self}).
\]

`y_t^{self}` 不能是 hallucination label。第一版可以使用当前 HoloRoute-Base 的 high-dimensional routing reconstruction target；更重要的是同一个 decoder 在所有视图中共享。

### 7.2 No-prompt view

将 `P` 质量切掉，保留 `R+Q`。为了避免“总质量变小”本身造成 loss 上升，对剩余已知 off-diagonal 质量做行内重标定，使它与 full view 的已知质量相同；diagonal 和 unresolved mass保持不变。

得到：

\[
L_t^{-P}=L(Enc(G^{-P},t),y_t^{self}).
\]

定义 prompt necessity：

\[
\Delta_t^P=L_t^{-P}-L_t^F.
\]

值大说明 prompt-rooted 路径对当前 token 的 routing state 很重要。

### 7.3 No-response-closed view

同理切掉 `R` 质量，保留 `P+Q`，并做相同的质量守恒重标定：

\[
\Delta_t^R=L_t^{-R}-L_t^F.
\]

值大说明 response-closed 路径很重要。

### 7.4 唯一主分数

\[
\boxed{C_t=\Delta_t^R-\Delta_t^P}
\]

高 `C_t` 的含义非常直接：

> 当前 token 更依赖回答内闭合路径，而不是 prompt-rooted 路径。

它不是把 entropy、degree、holonomy、reconstruction 和 position 随意相加。整个方法只有一个主比较。

必要时还可以报告 embedding shift：

\[
D_t^P=\|z_t^F-z_t^{-P}\|,
\qquad
D_t^R=\|z_t^F-z_t^{-R}\|,
\]

但它们是诊断，不另行拼成主分数。

---

## 8. 为什么这不是普通图异常检测

普通无监督图异常检测比较：

```text
这个节点是否不像训练集中的正常节点？
```

P-Cut 比较的是同一个 token 的三个 matched graph views：

```text
完整图
切掉 prompt-rooted 路径
切掉 response-closed 路径
```

因此它不要求幻觉是 population outlier。即使一个错误 token 的图非常常见、非常容易恢复，只要它对 response-closed 路径的依赖明显超过 prompt-rooted 路径，仍可能被检测到。

这是一种从 **anomaly likelihood** 转向 **within-token predictive sufficiency** 的范式变化。

必须用准确措辞：当前仅重跑图编码器时，它是“图视图上的预测充分性”，不是语言模型输出上的因果效应。只有后续真正修改基础模型 attention/value 并观察 logits，才能使用更强的因果表述。

---

## 9. 当前 HoloRoute 在新范式中的位置

HoloRoute 不删除，也不再承担“核心 insight”。它改名或定位为：

> **HoloRoute-Base：attention event graph masked autoencoder。**

它负责：

1. 读取 sparse attention；
2. 构造 layer/head-aware graph；
3. 输出 event / token embedding；
4. 提供一个共享的无标签 decoder；
5. 与 Flat-1024 做图结构对照。

P-Cut 只在其前后增加两个明确对象：

```text
prompt provenance
mass-preserving graph cuts
```

因此第一版代码不需要重写复杂消息传递，也不需要加入更多 residual heads。

---

## 10. 最小实现：四个文件足够

建议在当前项目中新增：

```text
experiments/holoroute/pcut/
├── provenance.py   # 传播 prompt-origin lower/upper bounds
├── cuts.py         # 构造 full / no-P / no-R 质量守恒视图
├── score.py        # 计算 ΔP、ΔR、C，并做 train-only 条件校准
└── export.py       # 保存 event、token-layer、token embeddings 和三视图结果
```

现有：

```text
graph.py
model.py
learning.py
baseline.py
```

继续作为共享底座。第一版不要加入新 GNN、flow、GRU、sheaf 或 holonomy 模块。

### 10.1 输出数据

每条样本至少保存：

```text
event_embedding          [E, d]
token_layer_embedding    [R, L, d]
token_embedding          [R, d]
prompt_origin_lower      [R, L]
prompt_origin_upper      [R, L]
prompt_necessity         [R]
response_closed_necessity[R]
closure_score            [R]
```

这样 P-Cut 既能直接检测，也能为后续无监督方法、可视化和 hidden-state 扩展提供节点表征。

---

## 11. 无监督训练和校准

训练阶段不读取 hallucination label。

1. 用 fit source groups 训练 HoloRoute-Base 或一个更简单的图编码器。
2. 用独立 validation groups 选择 checkpoint，只看自监督 loss。
3. 用 calibration groups 拟合 `C_t` 在 task、相对位置、response length、retained coverage 和 provenance interval width 条件下的经验分布。
4. test 先冻结 `C_t` 和 p-value，再打开标签。

最终无监督 score 可以直接是条件上尾：

\[
S_t=-\log \hat P(C\ge C_t\mid condition_t).
\]

不要再和五个辅助分数做 Fisher fusion。若单一 closure score 不够，先判定假设是否失败，而不是用更多分量把结果救回来。

---

## 12. 必须通过的实验门槛

### 12.1 先证明 provenance 计算不是 prompt mass 换皮

- P-Cut 必须优于直接 prompt share 和 Lookback。
- 一跳 provenance 的冻结方向结果要在多个 seed、任务和模型上复现。
- 将 response source 的 provenance 随机置换后，P-Cut 应明显退化。

### 12.2 证明 exact graph endpoints 有用

构造 matched rewire：保持 layer、head、role、lag、row mass、provenance bin 和 degree，只替换真实 source endpoint。真实图的 closure detection 必须优于 rewire。

若不优于，说明分数主要来自 role/mass，而不是图拓扑。

### 12.3 证明不是删质量造成的伪效应

- full、no-P、no-R 的 retained row mass 必须一致。
- `C_t` 与被删除原始质量、event count、position 的相关性要单独报告。
- 使用 equal-mass random cut 作为对照。

### 12.4 证明优于当前异常基线

至少比较：

```text
prompt mass / Lookback
RR raw spectral residual
Flat-1024
HoloRoute reconstruction residual
CoLA / GraphMAE adaptation
P-Cut
```

P-Cut 只有在 AUPRC、source-bootstrap CI 和位置控制上都胜出，才有资格成为主方法。

### 12.5 证明机制而不是只证明检测

后续拿到可重跑模型后，对高 closure token 做 attention/value 干预：

- 恢复 prompt-rooted 质量；
- 削弱 response-closed 质量；
- 与 layer/head/role/lag/mass 匹配的随机边干预比较；
- 观察正确 token logit、错误 token logit或答案事实性是否改善。

若图分数能检测但干预没有功能效应，论文只能写“predictive routing signature”，不能写“causal mechanism”。

---

## 13. 什么时候应当放弃这个假设

出现以下任一结果，就不再继续包装 P-Cut：

1. `C_t` 不优于直接 prompt mass 或 Lookback；
2. real endpoint 与 matched rewire 没差别；
3. 效果主要由 removed mass、position 或 response length解释；
4. 只有标签后验选择 layer/head/方向后才有效；
5. QA 有效但 Summary/Data2txt 方向完全相反且无法由任务条件解释；
6. 加入 value/hidden state 后发现 attention provenance 与实际 contribution 几乎无关。

这些是 falsification，不是“实验还不够复杂”。

---

## 14. Hidden state 和 value 以后怎样加入

不建议现在直接把 hidden state 拼到节点属性里。那样可能提高结果，但会让 attention 图的贡献无法判断。

合理顺序是：

### 阶段 A：attention-only

只验证 P-Cut 的 routing provenance 是否成立，并完成 Flat-1024、rewire 和位置控制。

### 阶段 B：value-aware transport

把 edge weight 从 `A_{t,s}` 扩展为 source-resolved contribution 的近似：

\[
A_{t,s}^{l,h}W_O^{l,h}V_s^{l,h}.
\]

这时 provenance 不只是“看了谁”，还开始描述运输了什么。

### 阶段 C：hidden-state adoption

比较 attention-derived prompt provenance 与 hidden/residual 中实际保留的 prompt/evidence direction，研究“routing 与 adoption 的分离”。

这样每一步都有独立问题和消融，不会把更多数据维度带来的收益误写成图机制。

---

## 15. 这条路线的新意在哪里

P-Cut 不以新 GNN 层作为卖点。它提出的是一个新的检测问题：

> **无监督幻觉检测不一定要找“异常内部状态”；可以检验同一个 token 在证据路径和自生成路径被分别切断时，哪一类路径对它更必要。**

它与已有方法的边界很清楚。

- 比 Lookback 多了一步 provenance：response source 可以是 grounded relay。
- 比 CHARM 少依赖标签：图编码器和分数都不读幻觉标签。
- 比 TOHA/HalluZig 更局部：输出逐 token 的证据依赖，而不是全局拓扑签名。
- 比 RFS-Guard 更适合当前数据：不依赖 hidden-state semantic similarity 和 step phase。
- 比 CausalGaze 更严格无监督：不使用 hallucination classifier gradient。
- 比 GraphMAE/CoLA 更换了问题：不假设错误一定难恢复或与邻域不一致。

“fancy”的地方不在模块多，而在把研究范式从：

```text
find an unusual graph
```

改成：

```text
test whether the answer can bypass its evidence
```

---

## 16. 允许写进论文的 claim

在 attention-only、未做基础模型干预的阶段，最多可以说：

> P-Cut traces attention-derived prompt provenance through response relays and measures within-token predictive dependence on prompt-rooted versus response-closed graph components.

不能说：

- 它恢复了真实事实因果链；
- 某条 attention edge 导致了输出；
- response-closed 一定等于 hallucination；
- P-Cut 已经优于所有方法；
- prompt provenance 等于模型实际采用的证据。

“优于现有方法”必须是多个任务、多个模型、多个 seed 和完整对照之后的实验结论，而不是方法设计阶段的前提。

---

## 17. 最简实验顺序

第一轮只做四个分数：

```text
Lookback
RR spectral residual
HoloRoute-Base reconstruction residual
P-Cut closure C_t
```

第一轮只做四个控制：

```text
position-only
equal-mass random cut
matched endpoint rewire
Flat-1024
```

第一轮只回答一个问题：

> **在不使用标签、不依赖“异常难恢复”的前提下，prompt-provenance cut 是否比直接 attention 比例和一般图异常分数更能定位幻觉 token？**

这个问题若回答不了，就不应继续增加 fragility、holonomy、GRU、flow 或更多融合分数。
