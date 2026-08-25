# P-Cut 方法

## 1. 图

一条 prompt-response 样本构造一张 multiplex token graph。token 是节点；每条 retained attention incidence 保留 exact source、response target、layer、head 和 weight。未保存边只计入 row-level unresolved mass，不当作零。

## 2. Prompt provenance

prompt token 的来源下界和上界都设为 1，response token 初始为 0。沿 layer 和 causal response edges 传播：

\[
\underline g_{t,h}^{\ell+1}
=\sum_{p\in P}A_{t,p}^{\ell,h}
+d_t^{\ell,h}\underline g_t^\ell
+\sum_{j<t}A_{t,j}^{\ell,h}\underline g_j^\ell,
\]

\[
\overline g_{t,h}^{\ell+1}
=\underline{\text{same retained terms}}
+u_t^{\ell,h}.
\]

这样 sparse cache 的未知质量形成 provenance 区间，而不是假零。

## 3. Edge partition

response-source edge `e=(j -> t,l,h)` 被精确拆成：

\[
w_e^P=A_e\underline g_j^l,
\qquad
w_e^R=A_e(1-\overline g_j^l),
\qquad
w_e^Q=A_e(\overline g_j^l-\underline g_j^l).
\]

它们分别表示至少可确认的 prompt-rooted mass、即使把未知质量全给 prompt 也无法追溯到 prompt 的 response-closed mass，以及不确定 mass。

## 4. Matched cuts

构造 full、no-prompt-rooted 和 no-response-closed 三个图视图。切割后对剩余 retained edges 在每个 `(target,layer,head)` 行内重新缩放，使 retained row mass 与 full view 相同。若某行没有可保留边，被删质量转入 unknown state，避免总质量变化成为捷径。

## 5. Token representation

每个 token 先获得固定的 deterministic identity embedding。attention 在每层、每 head 上传播这些身份表示；随后用固定 DCT-like head projection 压成 token-layer embedding：

```text
token_layer_embedding [response, layer, dimension]
token_embedding       [response, dimension]
```

这不是训练得到的分类表示，而是 exact endpoint、layer、head 和路径共同决定的 routing-state embedding。

## 6. 唯一主分数

比较 full 表示与两种 cut 表示：

\[
\Delta_t^P=d(z_t^F,z_t^{-P}),
\qquad
\Delta_t^R=d(z_t^F,z_t^{-R}),
\]

\[
C_t=\Delta_t^R-\Delta_t^P.
\]

高 `C_t` 表示 token 对 response-closed 路径的依赖超过 prompt-rooted 路径。train split 只用于拟合位置、长度、unresolved mass、provenance interval width 和 cut fallback 条件下的经验上尾；hallucination labels 不参与拟合。
