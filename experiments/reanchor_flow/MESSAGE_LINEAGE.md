# DAG 消息路由建模与幻觉机制发现

研究核心：**建立保留来源身份、原生消息变换与聚合规则的 DAG，从图中发现正常与幻觉的可重复传递结构差异，再检验这些结构是否提供检测信息。**

不能预设“幻觉就是少读 prompt”“高流量中继一定有效”或“材料支持不足就是核心机制”。
这些是可以被否定的假设；当前 `material_deficit` 是其中一个检测对照，不能代替图上的机制发现。

**实现状态：** `message_lineage.py` 已实现材料边界来源的条件向量传播和直接读出，尚未实现下文 §6.1 的
逐来源单位、同目标全路径贡献图。原始 v3 的 `audit train/QA` 是逐 head／两跳结构统计，不是这张全路径图。
没有新增真实 8B 机制或性能结论。已有缓存可复用；完整新图实现和小规模成本／可辨识性验证前，不推荐再追加全量采集。

## 1. 为什么回看峰不能定义关键节点

WAAD 峰只是局部距离极大值。正常与幻觉都可能回看；稳定地读取材料、在远处来源间切换、有效信息在同一来源中变化，都不一定产生峰。
v2 的峰交集和最大 attention 来源选择还使 36 条展示路径全部落到 BOS。这不是事实中继的证据。
v3 已经去掉峰交集；但其 `sum_b A_reader(q,b) * E_writer(b)` 仍是标量结构组合，不能确认中继传出的消息来自材料。

论文中的 `max(0, norm(y)-norm(y-a))` 是非负几何贡献规则，与时间上的回看峰不是一回事。
本实现保留正、负贡献。只有用于有限页面展示的 `top_k(abs(contribution))`，不影响传播、评分、总体统计。
不存在“没有峰就没有中继”“排名第一就确认关键”“没有材料支持就自动标幻觉”的规则。

## 2. 已有结果支持到什么程度

- 历史 QA 的 `functional_route_collapse` AUROC 约 .7337，而其 attention 对照约 .7333；Data2txt 约 .4961。
  这提示部分 QA 数据有路由区分信号，但没有证明引入消息算子就增加了信息，也没有跨任务稳定性。
- v2 全量覆盖 2,946 个样本、509,834 个 response token；42,587 个标为幻觉。
  部分逐 head 差异可复现，不能从“都经常回看”推出完全没有区分信号；也不能从某一 head 差异推出事实中继机制。
- 受控 `fact_flip` 的约 +16 logit-margin 变化证明模型对相关输入条件敏感；不能定位该响应由哪个自然回答中继传递。
- 新 v3 的最终真实数据结果尚未全部取得。统计显著、预测增益和因果解释分别报告，不能互相替代。

以上实验的样本范围不完全相同；历史 AUROC 不能替代新流程同一 token 集上的对照。

## 3. DAG 对应什么运算

节点必须有 **位置、层、阶段**，例如 `(b, layer l, input residual)`，而不只是一个静态 token。
边记录物理 head 的真实消息：

\[
m^{l,h}_{s\to q}=W_O^{l,h}\big[A^{l,h}_{q,s}V^{l,h}_s\big].
\]

同层所有 head 从更新前的状态读取；完成 attention 加法之后才进入 MLP。
因此同层 head 之间不存在先写后读的中继；严格更深层才可能读取刚更新的载体。
代码按层推进完整向量，再按原始 attention 汇集来源向量，正好遵循这项层序。
把同一 token 的不同层状态合成一个节点，会抹掉这个约束。

本实现没有训练额外的 GNN。它使用 Llama 原本的 W_V、W_O、W_gate、W_up、W_down 做条件分解。
head 在残差里相加是模型真实的聚合；所有物理 head 的读出和展示边另外保存，没有在采集前平均 head。

[FlowTracer](https://arxiv.org/pdf/2606.10646) 用 attention 标量图、答案条件化和守恒传播定义节点通量；
其实现平均中间层的 heads，用于 RL credit。它启发我们明确全局起点和终点；它没有追踪原生 value／MLP 的事实内容。
我们的设计需同时保留消息算子与来源，并检查全路径对同一目标的贡献。两者的“流”定义不能混用。

## 4. 材料向量如何进入并经过载体

以 `p_l(q)` 表示当前状态中归到材料边界输入的向量；初始为零。
只在 response-query 子图中追踪，所有 prompt V 状态视为每层的边界输入。
根集合是 v3 `evidence_mask & ~special_mask`，它表示数据材料位置，并未确认与当前 claim 的语义相关性。

冻结已观察 attention 与 RMSNorm 分母，令

\[
N_l(q)v=\frac{\gamma_l\odot v}{\sqrt{\mathrm{mean}(x_l(q)^2)+\epsilon}}.
\]

在每个 head，传递两部分：

\[
o^{M}_{l,h}(q)=W_O^{l,h}\left[
\sum_{s\in\mathrm{material}} A^{l,h}_{q,s}V^{l,h}_s
+\sum_{b\ge P,\ b\le q}A^{l,h}_{q,b}W_V^{l,g(h)}N_l(b)p_l(b)
\right].
\]

第一项是当前层直接读取材料；第二项使用载体已经传播到本层的材料向量。
它不能被 `attention * material_fraction` 替代：W_V 可以接受、旋转或消去这个向量中的方向。
第二项的 `b=q` 是同位置的跨层自读，不作为跨 token 中继展示；真正的历史中继要求 `P<=b<q`。

随后按照模型顺序更新：

\[
p_l^{post}(q)=p_l(q)+\sum_h o^M_{l,h}(q),\qquad
p_{l+1}(q)=p_l^{post}(q)+M_l(q)p_l^{post}(q).
\]

所有普通与特殊 token 仍参与原模型计算。特殊 token 不作为材料根、显示中继或主要 N/H 比较目标；其直接来源贡献独立保存。
材料经过特殊状态的影响没有从模型计算中删除，不能把普通中继解释为“整个祖先路径从未经过特殊 token”。

### MLP 的分配规则

在缓存的原生 post-attention 状态上计算 `z=RMSNorm(x)`、`g=W_gate z`、`u=W_up z`、`a=SiLU(g)`。
令 `k=a/g`，零点用极限 1/2。对归一化后的材料分量 `v` 使用

\[
M(v)=W_{down}\left[\tfrac12 a\odot(W_{up}v)
+\tfrac12 u\odot k\odot(W_{gate}v)\right].
\]

该算子线性于分量，包含 gate 和 up 两条支路；在原参考输入上求和重构原生 MLP 输出（浮点误差另计）。
它采用乘积的对称贡献分配，不是普通损失梯度，也不是把材料向量单独送进非线性 MLP。
后者不满足 `MLP(p)+MLP(x-p)=MLP(x)`，不能直接作为来源分解。

分配不是唯一的。`--mlp-rule up` 提供固定 gate、只分配 up 支路的敏感性对照，必须使用另一输出目录；不能在 test 上挑更好的规则。
两个规则都可以闭合，但来源归因不同，因此“能闭合”本身不是事实身份或因果可信度证明。
这里与固定 attention 的 CP-LRP、归一化处理及乘积对称分配有联系，**不是完整 AttnLRP 复现**；Q/K 的来源归因不在本实现范围内。
依据见 [AttnLRP / CP-LRP 比较与乘积规则](https://arxiv.org/html/2402.05602v2)。

## 5. 为什么同一张 attention 图可能给出不同结论

例如材料进入载体时沿 e1 方向，后续 reader 的 W_V 只接收 e2。
即使两条 attention 都为 1，没有产生 e2 分量就不会沿该 value 通道传递材料。
若中间 MLP 将部分材料归因映射到 e2，后续读取才能传出该部分。
本项目测试包含此反例；标量 `A1*A2` 在两种情况下相同。

因此 DAG 的用途是约束“哪次写入可能进入哪次读取”，向量与节点算子决定“这个通道传出了什么分量”。
只换一种图形布局、做标量随机游走或阈值剪边，不能自动获得这项能力。

## 6. 中继如何定位，能用来做什么

对目标 `q+1` 固定 observed-vs-runner 的读出方向 `d_q`，给每个真实 reader head 的历史边保存

\[
c^{native}_{l,h,b\to q}=d_q^T W_O^{l,h}A^{l,h}_{q,b}V_b,\qquad
c^{M}_{l,h,b\to q}=d_q^T W_O^{l,h}A^{l,h}_{q,b}W_V^{l,g(h)}N_l(b)p_l(b).
\]

这可以定位 **具有材料来源读出贡献的候选载体**：不是只被多次关注，而是它传出的指定分量在当前目标上有非零作用。
展示沿目标向前查这些边，不要求载体处有峰，不限制输入来源只取最大 attention，也不丢弃负读出。
候选可以用于检查具体文本、位置／层分布、N/H 差异及少量预先冻结的功能确认。
排名只是展示，不能据此命名“已确认关键事实 token”。

这里的边读出是对最终固定方向的直接贡献；后续计算造成的整个来源分量变化由 `material_residual_margin` 的逐层轨迹记录。
不能把 `b` 处对 `b+1` 的投影与 `q` 处对 `q+1` 的投影相乘当成同一事实的贡献；页面不这样拼接两条边。
本轮将材料向量按模型原生加法汇集，不逐个保存“每一材料 token × 每一 writer × 每一 reader”的独立向量分解。
所以它定位来源类别的载体，尚未唯一定位载体中某个具体事实来自哪个 writer。

### 6.1 完整方法应增加的全路径定义（设计，尚未实现）

以节点 `v=(position, layer, stage)` 表示计算状态，保持 residual-input、post-attention、post-MLP 分开。
attention 边 `e=(u,v,head)` 的算子为

\[
B_e=W_O^{l,h} A^{l,h}_{q,s} W_V^{l,g(h)} N_l(s).
\]

残差直通边是恒等算子，MLP 边用 §4 明示的条件分配算子。所有算子在同一次原生计算的参考状态上固定。
这些是已观察计算的分配规则，不是任意输入下 Transformer 的等价线性模型，也不是删边后的反事实模型。
浮点舍入差异作为显式外部余项；不能把它隐含归给材料。

**来源与前向聚合。** 对互斥且完备的来源组 `c`，保留

\[
x_v^{(c)}=b_v^{(c)}+\sum_{u\to v}B_{u\to v}x_u^{(c)},\qquad
x_v=\sum_c x_v^{(c)}.
\]

`b` 是外部注入：各层 prompt 边界、response 初始 embedding、特殊边界及数值余项。
第一步可按材料单位、其他 prompt、response 初始状态、特殊来源、余项划分；不是把“非材料余项”改名为错误历史。
经过材料位置的 prompt V 已含更早层的上下文混合，因此是**边界位置来源**，不是原始输入事实的纯净编码。
若要追到输入事实身份，需要完整 prompt 子图，或独立事实条件的配对；已有 response-only 状态不足以恢复前者。

**同目标的反向贡献。** 在最终 predictor `q=t-1` 固定目标 `F_t=d_t^T x_T`，由同一个 sink 定义

\[
\lambda_T=d_t,\qquad
\lambda_u=\sum_{u\to v}B_{u\to v}^{T}\lambda_v,\qquad
C_{u\to v}^{(c,t)}=\lambda_v^{T}B_{u\to v}x_u^{(c)}.
\]

这是固定算子的伴随传播，无需训练 GNN 或逐边消融；它把后续所有合法层序路径计入当前边的贡献。
它仍依赖 attention／归一化／MLP 分配规则，并不是原生非线性模型的因果效应。
当前实现的 `d_t^T message` 只是直接投影；一般不等于上述 `lambda_v^T message`，不能把两者混报。

对没有外部注入、也不是目标出口的内部节点，有**带符号的局部平衡**：

\[
\sum_{u\to v}C_{u\to v}^{(c,t)}
=\lambda_v^T x_v^{(c)}
=\sum_{v\to w}C_{v\to w}^{(c,t)}.
\]

有外部注入时左侧加 `lambda_v^T b_v^(c)`；有目标读出时右侧加对应出口。
守恒对象是固定目标下的代数贡献，不是物理信息量、非负概率或事实真值。
求绝对值后不再守恒；正负通量、净通量和抵消必须同时保存。所有边的贡献不能相加当成一次总输出，
否则同一路径会按长度重复计数；总输出只在完整切面或源／汇边界求和检查。

**关键节点从整张图识别。** 所有普通节点先获得入／出通量、来源组成、净贡献与抵消、后续目标覆盖。
高通量、汇聚、再次分发和路由集中是不同属性；排名仅用于展示，不把一次回看峰设为必要条件。
高通量可以出现在错误内容或格式处理上，是否与幻觉有关要由后续比较决定。
“收敛”需检查同一载体后续被哪些目标／head 读取、分布是否持续集中，并控制可见位置数；单次低熵不足以确认。

### 6.2 要从 DAG 中比较的结构，而不是预先宣布的机制

| 待检验结构 | 必须联合观察 | 单凭什么不能认定 |
|---|---|---|
| 有材料来源的压缩与复用 | 来源进入载体、经过变换仍可传出、多个后续目标经该载体获得贡献 | 后续少读 prompt，或载体被频繁读取 |
| 路由绕过相关约束 | 输出路径集中，同时独立确定的相关材料来源在该路径上缺失或作用改变 | 路径短、边少、总材料份额低 |
| 历史中的持续局部复用 | 多个相继目标依赖重叠的局部载体，新增相关来源注入弱，且旧来源成分未保留约束 | local attention 大，或出现重复文本 |
| 聚合与下游作用失配 | 指定来源确实进入，MLP／value 变换后分量仍存在但在同一目标上的贡献减弱或被抵消 | head／MLP 的直接读出为负 |
| 高流量瓶颈或多路径补偿 | 来源分支汇聚、向后分发、替代分支覆盖及正负贡献关系 | 入度大、某一 head 强、单条路径效应小 |

因果展开的 DAG 没有有向环。“循环利用”在这里应表达为随位置／层向前的反复复用，
或某个局部集合的持续占用；折叠 token 节点后画出的环不是模型中的真实计算环。
相关材料需独立的来源映射或约束配对；普通材料 mask 无法区分相关事实、干扰材料和错误关系。
暂时缺少这种映射的样本，只报告结构与边界来源，不硬填“正确关联”“错误捷径”等标签。

“聚合接受”拆成可核查的三个问题：消息是否写入（原生加法）、指定来源是否通过后续算子继续传播（条件分解）、
它怎样作用于固定判断（同目标全路径贡献）。事实约束是否正确是第四个语义问题，不能由前三项的范数或正号推出。

### 6.3 发现、验证与成本约束

1. 在不读幻觉标签的条件下构图，覆盖稳定读取、无峰、无中继、强／弱材料来源和正负贡献。
2. 保存逐节点与完整目标窗口的图状态，再接 N/H 标签比较；按来源汇总，控制位置、词法、任务、logprob。
   同一 source 的正常／幻觉回答及同回答内部对照应分别报告。正常数据也可能有相同结构，比较频率和作用，而非只找有无。
3. 分开报告三类证据：结构差异及区间；跨来源／任务重复性；同 token 集上的检测增益。
   探索后选择的结构在另留的来源上确认，不把已用于设计的 test 当成新的确认集。
4. 先验证全图比直接来源读出、度／attention 通量、位置与 logprob 增加了什么。
   只对冻结的少量具体路径补充事实条件变化或功能验证，不把逐路由消融作为全量发现入口。
5. 目标级伴随传播的成本随目标数增加。精确的每目标图不能声称只需一次常数成本传播。
   先在固定小批缓存记录耗时／峰值内存；再选择目标分块或明确的多目标窗口。
   多目标窗口得到的是该窗口固定函数的贡献，不能冒充每个 token 的独立归因。
6. 路径不逐条枚举，用 DAG 动态传播；稀疏化只用于显示并报告遗漏量。层内/逐 source 完成即保存，
   明确“当前阶段、目标／层、已完成数、耗时”，通过验证后再决定全量预算。

当前缓存提供了必要的 response 子图数据，但 §6.1 的来源单位传播、全目标伴随和节点结构报告尚待实现。
当前测试验证的是已有材料分解；不能据此宣称上面的完整 DAG 方案已经跑通。

## 7. 固定检测对照：材料支持不足假设

最终材料归因读出 `P_t=d_q^T p_L(q)`，其余部分 `U_t=d_q^T(x_L(q)-p_L(q))`。
固定待检验假设：相对缺少材料来源的正向输出支持，更可能出现幻觉。分数定义为

\[
S_t=-\frac{P_t}{|P_t|+|U_t|}.
\]

高分方向预先固定，不训练，不按标签选 head，不按 test 翻转符号。分母为零或没有材料根时缺失，不补零。
`U_t` 含其他 prompt、原始回答 embedding、未归到材料的计算及数值余项，**不能直接叫参数先验或错误历史**。
支持 observed token 不保证事实正确，材料来源也不保证内容相关；这个分数是可证伪的检测假设。

同一批有效 token 上比较：

1. `material_deficit`：包含多跳向量、残差和 MLP 传播的候选分数；
2. `direct_deficit`：只用目标位置直接材料消息读出，使用同一归一化和方向；
3. `negative_logprob`；
4. `position`。

AUROC、AP 和候选分数减对照的配对区间全部输出；source_id 整组 bootstrap。
所有分数共用同一有限值、已知标签、非特殊 query/target 集合。保存原始样本与 token 坐标，不把未知标签当正常。
另外保留三个预定义诊断的全部 layer/head：直接材料读出、继承材料的历史读出、历史剩余读出。
N/H 在同回答、同粗 token 类型的两侧正常位置插值后比较；先按 source 汇总，报告区间和 BY 校正。
train/test 相同 head 的重复性单独报告；单个随机 tiny 模型的测试结果不进入研究实证表。

若该分数未优于直接读出或位置对照，应报告这个分数未显示额外检测价值；不能据此推断整张 DAG 不含任何区分信息，
也不能因为图更复杂就声称成功。图模型是否有用还需由来源、聚合与传递的联合结构检验回答。
四类“事实未进入／未整合／被覆盖／读出沉默”的语义判断仍需独立事实定义，不能从上述数值直接命名。

## 8. 复用中断前数据，一键传播并评估

在仓库根目录执行：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_lineage_run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 \
  --completed-only --device cuda:0 --query-chunk 16
```

默认读取原 index 中的模型路径；移动模型后传 `--model /实际/model/目录`。
使用全部已完成的任务、split、样本和 response token，不做抽样，不依赖峰。原始 index 和原生状态保留。
逐样本、逐层显示进度；最终 NPZ 原子提交，同一命令自动复用已经完成的来源分解。
需要原生 v3 `.states.npz`、`.qk.npz`、`.history.npz` 和对应模型 safetensors 权重；v2 摘要不能补出这些状态。
捕获时使用 `--discard-states` 的样本无法执行该传播；程序不会自动补采或把不完整状态当零。

输出默认位于 `attention_audit_v3/message_lineage/`：

| 文件 | 用途 |
|---|---|
| `summary.md/json` | 固定检测分数、同 token 对照、source 区间、匹配和部分覆盖 |
| `train/test_tokens.npz` | 每个被评估 token 的分数、标签、sample/source/task、query 和 target 坐标 |
| `cohorts/*.npz/png` | 所有物理 layer/head 的 N、H、位置配对差异，不平均 head |
| `train/test_detection.png` | ROC/PR 图；AP 用 average precision 定义 |
| `split/task/id.npz` | 全 head 来源读出、继承材料分量、逐层材料/MLP读出、原生重建误差、显示边 |
| `gallery.html`、样本 HTML | 正负平衡示例、完整标签文本、固定 reader head 的目标选择器 |
| `view_message_lineage.ipynb` | 任意样本、layer/head 和目标；同一目标下的逐层材料读出轨迹 |

不再生成第二份巨大 `[layer,head,query,source,hidden]` 张量：只传播当前层材料向量，计算来源边时先把读出方向乘 W_O，利用 head_dim 空间收缩。
原生 attention 仍覆盖全来源，原始完整状态可用于以后重新分解；top-k 只限制另存的显示边。
这仍有按层矩阵运算和大缓存读取成本，不能承诺即时完成，也不是“只算几个标量”。

来源分解完成后，仅重评估／绘图：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_lineage_run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 \
  --phase evaluate --completed-only
```

此阶段只需要原始 compact NPZ/labels/index 与派生的来源分解 NPZ；不加载模型权重、Q/K、history 或 states。
在 CPU 上可单独完成；下载查看时无需附带大型状态缓存。notebook 的 AUDIT 路径可改为下载后的位置。

## 9. 代码与验证

- `message_lineage.py`：真实 V/W_O 边、RMS 分配、SwiGLU 分配、层序传播、固定分数。
- `message_lineage_report.py`：事后标签比较、同集合检测、source 区间、逐 head 图与文本。
- `message_lineage_run.py`：复用缓存、按层权重加载、中断续跑与离线评估。
- `attention_audit.py` 额外保存 runner IDs，省去未来离线查找 runner 的词表计算；旧 v3 兼容。

验证包含真实 tiny Llama 的 float32/bfloat16 原生写入重建、GQA、分块不变、根分量可加性、相同标量路由但不同 MLP 传输、负贡献、严格过去载体、特殊 token、追加文本前缀不变、旧 runner 恢复、无材料缺失、中断数据及无权重/无大型状态的再次评估。
尚未运行用户服务器上的新 8B 分解，没有新的真实 AUROC 或显著性结论。

参考：[Information Flow Reveals When to Trust Language Models](https://github.com/rxu0112/RAG-information-flow)，
[ALTI-Logit](https://aclanthology.org/2023.acl-long.301/)，[AttnLRP](https://arxiv.org/abs/2402.05602)。
