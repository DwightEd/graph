# Hub-aware evidence mechanism audit

## 1. 目标：定位信息转换，而不是 attention 峰

对 predictor 位置 (q)，所有 screen 和 rerun 使用同一个冻结目标：

\[
F_t=z_q(a)-z_q(b).
\]

native audit 中，(a) 是 teacher-forced observed token，(b) 是未干预 native run 中排除
(a) 后的 top runner。受控 pair 中，(a,b) 必须在实验前注册为 factual candidates。两种
estimand 不混报。

审计分成四个问题：

1. selected-source operator 的影响从哪里进入，是否经过 prompt/response hub；
2. 该影响是否改变 residual write，并推动固定 target；
3. attention、MLP 和后续层是保留、增强、抵消还是覆盖它；
4. 删除、恢复、阻断真实 message 后，target margin 是否按预测改变。

这一定义允许 `source → hub → target`。target 缺少 direct prompt attention 不足以否定间接路径；
只有 selected-source intervention 改变 hub state，且 hub→target patch/block 闭合 mediation，才称
该 operator 下的 causal relay。`grounded route` 还要求 matched factual contrast。

## 2. 真实消息与逐 head 图

attention (A) 只是 QK gate。实际写入 residual 的逐 head message 是

\[
m^{l,h}_{j\to i}
=W_O^{l,h}\left(A^{l,h}_{i,j}V^{l,g(h)}_j\right).
\]

因此图节点是 `(layer boundary, token position)`，显式边保存
`(layer, head, source, target)`。residual continuation 和 MLP 是同位置的层内转换。GQA head
映射在 message capture 时处理，不在分析后补近似。

不同 head 可能做 global retrieval、local copying 或相反方向的 write。向量合成遵循 Transformer
真实的 residual addition；诊断统计允许计算 signed sum/coherence，但数据契约始终保留
(L\times H\times P) 轴，不保存“平均 head”。

## 3. 四账分析模型

`HeadResolvedRouteModel` 接收一次 native capture 和已经选定的 `root_unit_id`。它不重新计算
attention，也不充当 detector。

### 3.1 Provenance / routing ledger

每个节点维护四通道来源：

```text
selected evidence root | other prompt | response origin | unobserved
```

对一个 destination row，以 residual norm 和逐边真实 message norm 构造候选 transport law：

\[
p_e=\frac{\lVert m_e\rVert_2}
{\lVert r_{l,i}\rVert_2+\sum_{h,j}\lVert m^{l,h}_{j\to i}\rVert_2},
\qquad
p_r=\frac{\lVert r_{l,i}\rVert_2}
{\lVert r_{l,i}\rVert_2+\sum_{h,j}\lVert m^{l,h}_{j\to i}\rVert_2}.
\]

未捕获/裁剪质量进入 `unobserved`，不重新分配给保留边。`carrier_scope` 默认是 `all`；若 reduced
scope 没有表示某个 prompt destination，它在下一层的来源也转为 `unobserved`，不会把初始 source
身份跨层硬拷贝。
节点 provenance 递推为

\[
Z_{l+1,i}=p_rZ_{l,i}+\sum_{h,j}p^{l,h}_{j\to i}Z_{l,j}
+p_{sink}e_{unobserved}.
\]

`edge_register` 和 `head_transport[L,H,P,4]` 因而能区分 direct evidence、经 response hub
中继的 evidence、other prompt 以及原生 response history。它只是 norm-based routing model，
不是向量信息的概率守恒或因果分解。

### 3.2 Target-action ledger

每条 native message 对冻结 margin 的一阶作用为

\[
\phi_e=\left\langle\nabla_{c_e}F_t,c_e\right\rangle .
\]

`head_gradient_action[L,H,P,4]` 按 source-node provenance 记录有符号 action。正值支持当前
target contrast，负值反对。它回答“这条可用路径在当前点能否推动 target”，但仍是局部一阶
screen，不能替代 rerun。

`reanchor_events` 在每个 head 的 selected-root action 轨迹上找局部峰。峰可以由 hub-carried
lineage 产生；`direct_fraction` 是 direct-root norm transport / total root-lineage norm transport，
不是语义内容比例或因果效应比例。local response
复用和 evidence-origin 复用分别保存，避免把带有 evidence lineage 的 response hub 错分成纯粹
self-history。

### 3.3 Source-cut integration ledger

native world 与 selected-root Value-message-cut world 使用同样的 token 坐标。对 residual input、
attention write 和 MLP write 分别保存

\[
P_k=\lVert u_k^{native}-u_k^{cut}\rVert_2,
\qquad
C_k=\left\langle\nabla_{u_k}F_t,
u_k^{native}-u_k^{cut}\right\rangle .
\]

(P_k) 只是 source-cut state displacement；(C_k) 是当前 target 下的 first-order action screen。
它们都不证明存在可分离的事实表征或该改变对 target 有因果作用。相同 token 与位置能抵消许多
共享的词法/语法结构，但 source cut 仍会改变后续非线性交互。

逐 head integration 保存 message-delta budget、合成后 net norm、vector coherence 和 signed
action。逐层 stage ledger 同时保存 residual/attention/MLP displacement 与 action。

### 3.4 Exact causal ledger

最终结论来自真实 forward rerun，而不是前三账：

- root necessity：删掉 selected source 的 Value messages；
- root conditional sufficiency：只保留 selected source；
- corridor rescue/block：恢复或删除选中的 pre-`W_O` messages；
- carrier patch/block：恢复 hub state 后，再切断 hub 到 target 的 downstream path；
- restoration check：原位删掉 message 后补回，必须重建相应世界的 margin。

只有 root、corridor、carrier 的方向和 restoration check 一致，才命名为 confirmed native causal
chain；matched factual pair 中才可进一步命名 confirmed factual corridor。attention、message norm、
gradient、throughput 都只是候选选择。

## 4. 一致性与“稳定状态”的严格命名

多个 heads 指向相似来源并不表示其 residual writes 一致。代码同时计算：

\[
\text{vector coherence}
=\frac{\left\lVert\sum_h\Delta m_h\right\rVert_2}
{\sum_h\lVert\Delta m_h\rVert_2+\epsilon},
\]

\[
\text{functional agreement}
=\frac{\left|\sum_h\phi_h\right|}
{\sum_h|\phi_h|+\epsilon}.
\]

attention/MLP 另存 vector cosine 与 functional agreement；相邻 residual state 的
source-cut delta cosine 作为 `state_continuity`。这些量保留时间/位置和 head 结构。

这些量只有在对应 displacement/action budget 非零且跨连续层成立时才有意义。高 coherence、高
agreement、高 continuity 只表示当前 target contrast 下的 `functional alignment`；它可能保留
source-conditioned 影响，也可能与 response-origin action 同时出现。
稳定性不是 correctness，也不能直接叫“推理谷底”或 attractor。真正的 attractor 主张需要
free-running 多次生成和双向扰动恢复实验；当前 teacher-forced DAG 不识别它。

用于 hallucination 的预注册机制对照是两个轴：

\[
G_t=\text{native source-mediated observed margin},\qquad
R_t=\text{response-origin supporting-action candidate}.
\]

(R_t) 高是正常生成也会出现的现象；(G_t) 低、(R_t) 高只能称 response-reliance candidate。
真正的生成自我强化主张需要独立的 response-history cut/patch。捕获冻结后才加入 labels。

## 5. Verbal-confidence cache → retrieve 的启发

[*How do LLMs Compute Verbal Confidence?*](https://arxiv.org/html/2603.17839v3) 用 steering、
corrupt-restore patching、swap、probe 和 attention blocking 得到一个可迁移的实验模板：与
confidence 相关的 answer state 先汇入 post-answer newline cache，后续 confidence cue 再取回；
更长模板中的多跳中继是论文对 direct block null result 的可能解释，而非已经定位的固定路径。

本方法迁移的是以下逻辑：先定位 cache candidate，再分别验证 state displacement、target action、
必要性和 downstream mediation。不能迁移的是语义结论：verbal-confidence 论文没有
操纵外部 supporting evidence，也没有证明 cache 表示 grounded truth。这里必须始终把
evidence lineage、response-history/commit state 和最终 correctness 分开。

## 6. 两类实验与可识别边界

### Native RAGTruth source cut

native 流程可声称：在 label-free teacher-forced 单世界中，observed-token vs frozen-runner margin
对指定 source Value-message-cut operator 的依赖，以及通过精确 rerun 门槛的 native causal chain。

该 cut 不直接 mask Q/K，也不重归一 attention；source self-message 删除后，source state 和后续
Q/K 会在 cut world 中自然演化。它不能证明 observed token 正确，不能排除词法/语法贡献，也不能
把 source-conditioned residual delta 称作独立“证据语义向量”。

### Controlled clean/counterfactual pair

grounded factual-effect 结论需要预先构造 matched worlds：

- 相同 tokenizer、token 长度、position coordinates 和 teacher-forced response；
- 只改变注册的 factual value/entity/relation unit；
- 固定 grounded/counterfactual candidates (a,b)；
- 两个方向都做 patch/block，并加入 grammar/template、随机同位置 span、other-prompt controls；
- 无法 token-align 的 corruption 不进入原位 patching 主分析。

这让共享结构最大程度抵消，但仍不声称获得“纯语义”。

## 7. 可视化判读

四联图与四账一一对应：

| 面板 | 读法 | 限制 |
|---|---|---|
| route DAG | connected root→target backbone、root-lineage allocated action、head 和 hub | 候选 topology |
| head map | evidence 与 response-origin action，不平均 head | first-order use |
| integration | residual/attention/MLP displacement、action、一致性和 continuity | source-conditioned screen |
| intervention ladder | root/corridor/carrier 的真实 margin effect | causal confirmation |

判定 verified evidence hub 还要求 matched factual contrast：route 连通、state displacement 高于 controls、hub
patch 能 rescue、hub block 能破坏、hub→target path mediation 成立且没有被下游反向覆盖。native
source-cut 通过相同结构时只命名为 native carrier。
