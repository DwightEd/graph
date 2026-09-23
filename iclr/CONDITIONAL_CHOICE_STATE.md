# 条件选择与来源状态：实现及四答诊断

本轮新增 `main.py transport-state`。复用完整 value-path 缓存或 compact ZIP，
CPU 评分，不加载大模型，不消融，不训练，不覆盖旧结果。
默认 `risk` 仍为 `raw_route`；新读出单独比较，没有自动选方法、改符号或调系数。

## 1. 哪个条件概率属于大模型

令生成前状态为完整 prefix KV、位置和当前残差表示 `S_t`：

\[
p_\theta(y_t\mid x,y_{<t})=p_\theta(y_t\mid S_t),\qquad
S_{t+1}=F_\theta(S_t,y_t).
\]

给定实际 token，原生状态转移是确定的；采样的不确定性来自词分布。
不能把归因图的转移矩阵说成大模型的原生状态转移概率。

对当前实际词 `y_t` 和每个竞争词 `c`，缓存给出条件 log-odds：

\[
\Lambda_{tc}=\log\frac{p_\theta(y_t\mid S_t)}{p_\theta(c\mid S_t)},\qquad
\sum_k R_{tkc}=\Lambda_{tc}+\epsilon_{tc}.
\]

`R` 是输入根在指定值路径分解下的作用。正负号只说明相对这个竞争词的方向，
不自动说明事实支持/反驳。FFN、残差和后续层已经进入 `R`；不再次相加。
`group_choice_factor=softmax([0,-sum_group(R)])` 是归一化因子，
所有组因子的乘积归一化后重建捕获候选上的词分布（受账本误差影响）。
单个因子不是“仅给这一来源时”的模型预测，也不是来源独立的似然。

继承现有 [DecompX](https://aclanthology.org/2023.acl-long.149/) 等分解思想的范围见
[VALUE_PATH_TRANSPORT.md](VALUE_PATH_TRANSPORT.md)。本实现固定 attention/RMS，
使用 SiLU 割线与乘法等分，不是包含 Q/K 路由变化的原生 Jacobian。

## 2. 先保留来源、符号和候选，再读出

实际观测仍是每个根、每个候选的 `R`，没有 SVD 或训练压缩。
竞争词权重为原生条件概率：

\[
w_{tc}=p_\theta(c\mid S_t,y\ne y_t),\qquad
u_t=1-\sum_{c\in\mathcal C_t}w_{tc}.
\]

`u_t` 是未捕获备选词的条件概率质量。候选以实际 ID 保存，不跨位置匹配“第几个候选”。
未捕获质量进入 unresolved；不会因为缺数据而成为负证据。
原生全词表 entropy、来源读取 entropy、状态 entropy 分别记录，不能互相替代。

缓存的 float32 `1-sum(probability)` 曾出现 `-1.19e-7`，经条件化放大为负质量。
现在利用保存的 logits 和 surprisal 在 float64 对数域重建尾部：
`log(captured_mass)=logsumexp(logits-logits[0])-surprisal`，
`tail=-expm1(min(log(captured_mass),0))`。超过 `1e-5` 的正 log-mass 拒绝为不一致输入，
更小的归一化舍入修正和尾部差异逐 token 保存，原始捕获不修改。

每个候选先按 `Z_tc=sum_k(abs(R_tkc))` 归一化正负作用。
prompt 根按自动来源块、other、special 和正负号进入终止分布 `D_t`。
历史根的负作用也终止；历史根的正作用形成严格向过去的边：

\[
K_{tj}=\sum_c w_{tc}\frac{[R_{t,P+j,c}]_+}{Z_{tc}},\quad j<t.
\]

当前 predictor self 是 `j=t-1`，它是已经生成的上一 token，因此包括在历史边中。
这里连接的是历史输入 token 的生成决策 `j`，不是消费它时的 query 状态 `j+1`。
使用完整 `root_token_choice`，不把 compact ZIP 的 top-8 attention 当完整图。
所有历史正作用质量从终止分布移到边；负作用不跨两个不同的词汇对比相乘。
零作用的候选进入 unresolved。这样每行 `sum(D_t)+sum(K_t)=1`，只有舍入误差。

状态是来源/符号分布：

\[
A_t=D_t+\sum_{j<t}K_{tj}A_j,\qquad
A=(I-K)^{-1}D.
\]

`K` 严格下三角，递推有限终止，不需要收敛迭代、训练或滑动窗口。
逐候选 `candidate_state` 也保存；只有读出时按本位置的候选概率求期望。
复杂度主要是缓存 I/O 和 `O(T²CB)` 状态传递，其中 T 为回答长度，B 为来源组数，
C 为捕获的备选词数；没有第二个语言模型。矩阵逆只用于软件核验，实际实现按 token 递推。

**这一步跨过了离散 token 的生成边界，是额外检测假设。**
原生根归因已经贯穿固定 prefix 的 KV 值路径；`A_j` 再传递不能宣称是另一份精确根归因。
它表示我们定义的归因路径终点分布，不是幻觉后验，也不证明复制了此前的语义。
没有把内生的证据重复读取当成独立观测，再乘一次贝叶斯似然。

## 3. 读出与边界

定义 `s_t` 为来源正向终止质量，`v_t` 为 unresolved 质量。
主探索读出是：

\[
\mathrm{lineage\_source\_deficit}_t
=1-\frac{s_t}{1-v_t}.
\]

实现使用实际已解析质量作分母，避免舍入后与 `1-v_t` 的微小差异。
全部 unresolved 时返回未评分，而非高风险或零风险。
同时给出未解析质量对应的路径事件上下界；它们不是事实错误率的置信区间。
`local_source_deficit` 不递推：把历史正作用终止在历史根，形成直接根读出对照。
它们衡量的是“没有来源正向终点”，所以纯历史延续并不必然高分：
若此前状态有来源支持，递推可以继承它。但来源正作用本身仍不保证事实正确。

`local_opposition`、`lineage_opposition` 保留失败假设的数据。
`descendant_opposition` 用同一图的反向递推给出离线后续来源反向作用暴露，
并保存路径访问量作为分母。它使用未来 token，是另一个诊断读出，不混入主分数。

`head_group_attention` 保存全部层/头的当前地址读取，self 单独分组。
`source_read_mass/concentration/entropy` 描述是否集中回看来源；
`source_support_terminal` 描述当前最终选择的来源根作用。
二者必须分开：根作用可以经过历史 KV 到达当前输出，不一定是此时新读取了 prompt。
本轮没有把任何高集中度位置自动标注为语义 reanchor，也没有识别“所有头协同一致”。

## 4. 实测：收益来自哪里，哪里没有通过

836 token，81 标错，4 个反复使用的诊断回答。标签只在所有分数落盘后读取。
生成者是 Llama-2-7b-chat，观察者是 Llama-3.1-8B-Instruct；这里的“原生”指观察者
在保存 prefix 上的 teacher-forcing 前向，不能当成原始生成者当时的内部状态。
先冻结“反向作用继承”假设，发现 AUROC 仅 0.413；随后依据同一批诊断提出来源支持缺口。
因此以下是**探索结果，不是独立测试效果**。没有搜索参数或按标签反转分数。

| 读出 | AUROC | AP | 最高 84 分中的错误 token |
|---|---:|---:|---:|
| raw_route | 0.754934 | 0.199525 | 20 |
| route_offline_mean | 0.784400 | 0.193617 | 12 |
| local_source_deficit | 0.769226 | 0.218225 | 19 |
| lineage_source_deficit | 0.752383 | 0.203478 | 17 |
| lineage_opposition | 0.413294 | 0.080582 | 2 |

直接条件读出比 raw 的 AUROC/AP 增加 0.014292/0.018700；
尚未胜过离线均值 AUROC，top-84 精度也未胜过 raw。
历史继承没有改善直接读出，因此不提升为默认方法，也不能宣称已经解决错误状态维持。
状态质量最大误差约 `6.46e-8`；原生归因账本最大绝对误差 `0.251721`，二者不是同一指标。

四答 ZIP 的 CPU 评分只需数秒（见实际输出 `scoring_seconds`），没有执行模型。
更细的分阶段、within-answer、top-rank、位置相关数据见
[CHOICE_STATE_PILOT.json](CHOICE_STATE_PILOT.json)。

## 5. 交付数据与下一步

| 输出 | 内容 |
|---|---|
| scoring_protocol.json | 公式方向、条件概率范围、尾部处理、探索顺序、是否使用未来 |
| responses/*/state.npz | 来源/符号状态、逐候选状态、实际候选 ID、条件词概率、来源因子、历史边、全部头读取 |
| responses/*/scores.npz | 各独立分数、原始基线、未解析质量、账本与状态质量误差 |
| evaluation.json / comparisons.csv | 总体、首错、延续、同回答和 source-balanced 比较 |
| ranking_audit.json / state_audit.json | 头部排序、位置相关、按回答/标签分组的分位数；非匹配对照 |
| onsets.csv / high_risk_normals.csv / recovery.csv | 首错、高分正常、恢复位置的数据 |

下一步需要核验历史词支持当前选择时，承载的是相同断言还是一般语言组织作用。
只凭正向归因不能区分二者；当前继承反而降低 AP，说明这项桥接假设需要额外可观测量。
不在这四答上继续选窗口、混合权重或宣布新状态架构有效。
先把直接读出和原始基线放到未参与设计的既有缓存上复核，再决定是否扩大采集。
